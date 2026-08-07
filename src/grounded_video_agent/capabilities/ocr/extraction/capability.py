from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance, manifest_ref, write_json
from grounded_video_agent.capabilities.ocr.extraction.contracts import OCRExtractionRequest
from grounded_video_agent.domain import (
    BoundingBox,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ManifestKind,
    OCRManifest,
    OCRObservation,
    OCRSpan,
    TimeRange,
)
from grounded_video_agent.infrastructure.ocr import OCRBackend, OCRFrameInput, OCRFrameResult


@dataclass(slots=True)
class _Track:
    observations: list[OCRObservation]

    @property
    def last(self) -> OCRObservation:
        return self.observations[-1]


class OCRExtractionCapability:
    VERSION = "1.0.0"

    def __init__(
        self,
        backend: OCRBackend,
        output_root: str | Path = "artifacts",
    ) -> None:
        self._backend = backend
        self._output_root = Path(output_root).resolve()

    def execute(self, request: OCRExtractionRequest) -> CapabilityResult[OCRManifest]:
        started = perf_counter()
        frames = request.frames.frames
        frame_limit = request.context.limits.max_frames
        limited = frame_limit is not None and len(frames) > frame_limit
        if frame_limit is not None:
            frames = frames[:frame_limit]
        try:
            model_info = self._backend.get_model_info()
            results = self._backend.recognize(
                tuple(OCRFrameInput(frame.frame_id, frame.image.uri) for frame in frames)
            )
            observations, conversion_warnings = self._convert(request, frames, results)
            spans = self._merge_spans(observations, request)
        except Exception as error:
            return CapabilityResult(
                status=CapabilityStatus.FAILED,
                data=None,
                error=CapabilityError("OCR_EXTRACTION_FAILED", str(error), "ocr", True),
                usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
            )

        provenance = make_provenance(
            "ocr-extraction",
            self.VERSION,
            {"request": request, "model": model_info},
            video_id=request.frames.video_id,
            source_artifact_ids=(request.frames.ref.artifact.artifact_id,),
        )
        manifest_id = f"ocr_{request.context.operation_id}"
        path = self._output_root / "ocr" / request.frames.video_id / f"{manifest_id}.json"
        ref = manifest_ref(
            path,
            manifest_id=manifest_id,
            kind=ManifestKind.OCR,
            video_id=request.frames.video_id,
            item_count=len(observations),
            provenance=provenance,
        )
        manifest = OCRManifest(ref, request.frames.video_id, observations, spans)
        write_json(path, manifest)
        warnings = list(conversion_warnings)
        if limited:
            warnings.append(f"OCR input was limited to {len(frames)} frame(s).")
        if not observations:
            warnings.append("OCR found no text in the submitted frames.")
        status = (
            CapabilityStatus.SUCCESS
            if observations and not warnings
            else CapabilityStatus.PARTIAL
        )
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(ref.artifact,),
            warnings=tuple(warnings),
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(frames),
                output_items=len(observations),
                decoded_frames=len(frames),
                returned_frames=len(frames),
                model_calls=len(frames),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _convert(
        request: OCRExtractionRequest,
        frames: tuple,
        results: tuple[OCRFrameResult, ...],
    ) -> tuple[tuple[OCRObservation, ...], tuple[str, ...]]:
        frame_by_id = {frame.frame_id: frame for frame in frames}
        result_ids = tuple(result.frame_id for result in results)
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("OCR backend returned duplicate frame results")
        unknown = set(result_ids).difference(frame_by_id)
        if unknown:
            raise ValueError("OCR backend returned unknown frame results")
        warnings = tuple(
            f"OCR backend returned no result for frame {frame_id}."
            for frame_id in frame_by_id
            if frame_id not in set(result_ids)
        )
        observations: list[OCRObservation] = []
        for result in results:
            frame = frame_by_id[result.frame_id]
            for detection in result.detections:
                if detection.confidence < request.min_confidence:
                    continue
                normalized = _normalize_text(detection.text)
                bbox = _normalized_bbox(result, detection.polygon)
                if not normalized or bbox is None:
                    continue
                index = len(observations)
                observations.append(
                    OCRObservation(
                        observation_id=f"ocr_{request.context.operation_id}_{index:06d}",
                        video_id=request.frames.video_id,
                        frame_id=frame.frame_id,
                        timestamp_ms=frame.timestamp_ms,
                        raw_text=detection.text.strip(),
                        normalized_text=normalized,
                        bbox=bbox,
                        confidence=detection.confidence,
                        language=request.language,
                    )
                )
        observations.sort(key=lambda item: (item.timestamp_ms, item.observation_id))
        return tuple(observations), warnings

    @staticmethod
    def _merge_spans(
        observations: tuple[OCRObservation, ...],
        request: OCRExtractionRequest,
    ) -> tuple[OCRSpan, ...]:
        tracks: list[_Track] = []
        for observation in observations:
            candidates = [
                track
                for track in tracks
                if 0 <= observation.timestamp_ms - track.last.timestamp_ms
                <= request.max_merge_gap_ms
                and _text_similarity(
                    observation.normalized_text,
                    track.last.normalized_text,
                )
                >= request.min_text_similarity
                and _bbox_iou(observation.bbox, track.last.bbox) >= request.min_bbox_iou
            ]
            if candidates:
                best = max(
                    candidates,
                    key=lambda track: (
                        _text_similarity(
                            observation.normalized_text,
                            track.last.normalized_text,
                        ),
                        _bbox_iou(observation.bbox, track.last.bbox),
                    ),
                )
                best.observations.append(observation)
            else:
                tracks.append(_Track([observation]))
        spans: list[OCRSpan] = []
        for track in tracks:
            if len(track.observations) < request.min_span_occurrences:
                continue
            representative = max(track.observations, key=lambda item: item.confidence)
            first = track.observations[0]
            last = track.observations[-1]
            spans.append(
                OCRSpan(
                    span_id=f"ocr_span_{request.context.operation_id}_{len(spans):06d}",
                    video_id=first.video_id,
                    time_range=TimeRange(
                        first.timestamp_ms,
                        max(first.timestamp_ms + 1, last.timestamp_ms + 1),
                    ),
                    text=representative.normalized_text,
                    observation_ids=tuple(item.observation_id for item in track.observations),
                    confidence=sum(item.confidence for item in track.observations)
                    / len(track.observations),
                )
            )
        spans.sort(key=lambda item: (item.time_range, item.span_id))
        return tuple(spans)


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _normalized_bbox(
    result: OCRFrameResult,
    polygon: tuple[tuple[float, float], ...],
) -> BoundingBox | None:
    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    left = min(max(0.0, min(x_values)), float(result.width))
    right = min(max(0.0, max(x_values)), float(result.width))
    top = min(max(0.0, min(y_values)), float(result.height))
    bottom = min(max(0.0, max(y_values)), float(result.height))
    if right <= left or bottom <= top:
        return None
    return BoundingBox(
        left / result.width,
        top / result.height,
        (right - left) / result.width,
        (bottom - top) / result.height,
    )


def _text_similarity(first: str, second: str) -> float:
    return SequenceMatcher(None, first.casefold(), second.casefold()).ratio()


def _bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0:
        return 0.0
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union

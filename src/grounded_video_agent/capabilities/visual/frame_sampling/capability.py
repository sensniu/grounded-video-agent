from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from grounded_video_agent.capabilities._support import (
    file_artifact,
    make_provenance,
    manifest_ref,
    write_json,
)
from grounded_video_agent.capabilities.visual.frame_sampling.contracts import FrameSamplingRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    FrameManifest,
    FrameRef,
    FrameSamplingStrategy,
    ManifestKind,
    Provenance,
    TimeRange,
)


class FrameSamplingCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts") -> None:
        self._output_root = Path(output_root).resolve()

    def execute(self, request: FrameSamplingRequest) -> CapabilityResult[FrameManifest]:
        started = perf_counter()
        limit = request.context.limits.max_frames
        max_frames = min(request.max_frames, limit) if limit is not None else request.max_frames
        requested_timestamps = self._requested_timestamps(request, max_frames)
        provenance = make_provenance(
            "frame-sampling",
            self.VERSION,
            request,
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        try:
            frames, artifacts, decoded, dropped = self._decode(
                request,
                requested_timestamps,
                provenance,
            )
        except Exception as error:
            return CapabilityResult(
                status=CapabilityStatus.FAILED,
                data=None,
                error=CapabilityError("FRAME_SAMPLING_FAILED", str(error), "opencv"),
                usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
            )

        manifest_id = f"frames_{request.context.operation_id}"
        path = (
            self._output_root
            / "manifests"
            / request.video_asset.video_id
            / f"{manifest_id}.json"
        )
        ref = manifest_ref(
            path,
            manifest_id=manifest_id,
            kind=ManifestKind.FRAMES,
            video_id=request.video_asset.video_id,
            item_count=len(frames),
            provenance=provenance,
        )
        manifest = FrameManifest(
            ref=ref,
            video_id=request.video_asset.video_id,
            strategy=request.strategy,
            requested_ranges=request.ranges,
            frames=frames,
            decoded_frames=decoded,
            dropped_duplicates=dropped,
        )
        write_json(path, manifest)
        status = CapabilityStatus.SUCCESS if frames else CapabilityStatus.PARTIAL
        warnings = () if frames else ("No frames could be decoded for the requested ranges.",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(*artifacts, ref.artifact),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(request.ranges),
                output_items=len(frames),
                processed_duration_ms=sum(item.duration_ms for item in request.ranges),
                decoded_frames=decoded,
                returned_frames=len(frames),
            ),
            provenance=provenance,
        )

    def _decode(
        self,
        request: FrameSamplingRequest,
        timestamps: tuple[int, ...],
        provenance: Provenance,
    ) -> tuple[tuple[FrameRef, ...], tuple[ArtifactRef, ...], int, int]:
        import cv2

        capture = cv2.VideoCapture(request.video_asset.source.uri)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {request.video_asset.source.uri}")
        output_dir = self._output_root / "frames" / request.video_asset.video_id
        output_dir /= request.context.operation_id
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[FrameRef] = []
        artifacts: list[ArtifactRef] = []
        previous_signature: Any = None
        decoded = 0
        dropped = 0
        try:
            for requested_ms in timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, requested_ms)
                ok, image = capture.read()
                if not ok:
                    continue
                decoded += 1
                actual_ms = max(0, round(capture.get(cv2.CAP_PROP_POS_MSEC)))
                signature = cv2.resize(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
                if (
                    request.deduplicate
                    and previous_signature is not None
                    and float(cv2.absdiff(signature, previous_signature).mean()) < 2.0
                ):
                    dropped += 1
                    continue
                previous_signature = signature
                index = len(frames)
                frame_id = f"frame_{request.context.operation_id}_{index:06d}"
                image_path = output_dir / f"{frame_id}.jpg"
                if not cv2.imwrite(
                    str(image_path),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, request.jpeg_quality],
                ):
                    raise RuntimeError(f"Failed to write frame: {image_path}")
                artifact = file_artifact(
                    image_path,
                    artifact_id=f"{frame_id}_image",
                    kind=ArtifactKind.FRAME_IMAGE,
                    provenance=provenance,
                )
                segment_ids = self._segment_ids(request, actual_ms)
                frames.append(
                    FrameRef(
                        frame_id=frame_id,
                        video_id=request.video_asset.video_id,
                        timestamp_ms=actual_ms,
                        requested_timestamp_ms=requested_ms,
                        image=artifact,
                        segment_ids=segment_ids,
                    )
                )
                artifacts.append(artifact)
        finally:
            capture.release()
        ordered = tuple(sorted(frames, key=lambda frame: frame.timestamp_ms))
        return ordered, tuple(artifacts), decoded, dropped

    @staticmethod
    def _segment_ids(request: FrameSamplingRequest, timestamp_ms: int) -> tuple[str, ...]:
        if request.shots is None:
            return ()
        return tuple(
            shot.shot_id
            for shot in request.shots.shots
            if shot.time_range.contains_timestamp(timestamp_ms)
        )

    @classmethod
    def _requested_timestamps(
        cls,
        request: FrameSamplingRequest,
        max_frames: int,
    ) -> tuple[int, ...]:
        if request.strategy is FrameSamplingStrategy.SHOT_KEYFRAME:
            assert request.shots is not None
            timestamps = [
                (overlap.start_ms + overlap.end_ms) // 2
                for shot in request.shots.shots
                for requested_range in request.ranges
                if (overlap := shot.time_range.intersection(requested_range)) is not None
            ]
            return tuple(sorted(set(timestamps))[:max_frames])
        if request.strategy in {
            FrameSamplingStrategy.FIXED_FPS,
            FrameSamplingStrategy.DENSE_WINDOW,
        }:
            assert request.fps is not None
            step_ms = max(1, round(1000 / request.fps))
            timestamps = [
                timestamp
                for requested_range in request.ranges
                for timestamp in range(
                    requested_range.start_ms,
                    requested_range.end_ms,
                    step_ms,
                )
            ]
            return tuple(timestamps[:max_frames])
        return cls._uniform_timestamps(request.ranges, max_frames)

    @staticmethod
    def _uniform_timestamps(ranges: tuple[TimeRange, ...], count: int) -> tuple[int, ...]:
        total_duration = sum(item.duration_ms for item in ranges)
        count = min(count, total_duration)
        offsets = [round((index + 0.5) * total_duration / count) for index in range(count)]
        timestamps: list[int] = []
        for offset in offsets:
            remaining = min(offset, total_duration - 1)
            for item in ranges:
                if remaining < item.duration_ms:
                    timestamps.append(item.start_ms + remaining)
                    break
                remaining -= item.duration_ms
        return tuple(timestamps)

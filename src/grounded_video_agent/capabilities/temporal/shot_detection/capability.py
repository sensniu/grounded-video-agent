from __future__ import annotations

from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance, manifest_ref, write_json
from grounded_video_agent.capabilities.temporal.shot_detection.contracts import ShotDetectionRequest
from grounded_video_agent.domain import (
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ManifestKind,
    Shot,
    ShotManifest,
    TimeRange,
)


class ShotDetectionCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts") -> None:
        self._output_root = Path(output_root).resolve()

    def execute(self, request: ShotDetectionRequest) -> CapabilityResult[ShotManifest]:
        started = perf_counter()
        try:
            from scenedetect import (  # type: ignore[import-untyped]
                ContentDetector,
                SceneManager,
                open_video,
            )

            video = open_video(request.video_asset.source.uri)
            minimum_frames = max(
                1,
                round(video.frame_rate * request.min_shot_duration_ms / 1000),
            )
            manager = SceneManager()
            manager.add_detector(
                ContentDetector(threshold=request.threshold, min_scene_len=minimum_frames)
            )
            video.seek(request.source_range.start_ms / 1000)
            manager.detect_scenes(
                video,
                end_time=request.source_range.end_ms / 1000,
                show_progress=False,
            )
            detected = manager.get_scene_list(start_in_scene=True)
        except Exception as error:
            return CapabilityResult(
                status=CapabilityStatus.FAILED,
                data=None,
                error=CapabilityError("SHOT_DETECTION_FAILED", str(error), "pyscenedetect"),
                usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
            )

        ranges: list[TimeRange] = []
        for start, end in detected:
            start_ms = max(0, round(start.seconds * 1000))
            end_ms = max(start_ms + 1, round(end.seconds * 1000))
            intersection = TimeRange(start_ms, end_ms).intersection(request.source_range)
            if intersection is not None:
                ranges.append(intersection)
        if not ranges:
            ranges = [request.source_range]

        shots = tuple(
            Shot(
                shot_id=f"shot_{request.context.operation_id}_{index:06d}",
                video_id=request.video_asset.video_id,
                time_range=time_range,
            )
            for index, time_range in enumerate(ranges)
        )
        provenance = make_provenance(
            "shot-detection",
            self.VERSION,
            request,
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        manifest_id = f"shots_{request.context.operation_id}"
        path = (
            self._output_root
            / "manifests"
            / request.video_asset.video_id
            / f"{manifest_id}.json"
        )
        ref = manifest_ref(
            path,
            manifest_id=manifest_id,
            kind=ManifestKind.SHOTS,
            video_id=request.video_asset.video_id,
            item_count=len(shots),
            provenance=provenance,
        )
        manifest = ShotManifest(ref=ref, video_id=request.video_asset.video_id, shots=shots)
        write_json(path, manifest)
        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            data=manifest,
            artifacts=(ref.artifact,),
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=1,
                output_items=len(shots),
                processed_duration_ms=request.source_range.duration_ms,
            ),
            provenance=provenance,
        )

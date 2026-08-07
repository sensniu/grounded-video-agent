from __future__ import annotations

from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import (
    make_provenance,
    manifest_ref,
    write_json,
)
from grounded_video_agent.capabilities.visual.content_analysis.contracts import (
    VisualContentAnalysisRequest,
)
from grounded_video_agent.domain import (
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ManifestKind,
    VisualDescription,
    VisualDescriptionManifest,
)
from grounded_video_agent.infrastructure.visual_model import (
    VisualModelBackend,
    VisualModelFrame,
    VisualModelObservation,
    VisualModelRequest,
    VisualModelTarget,
)


class VisualContentAnalysisCapability:
    VERSION = "1.0.0"

    def __init__(
        self,
        backend: VisualModelBackend,
        output_root: str | Path = "artifacts",
    ) -> None:
        self._backend = backend
        self._output_root = Path(output_root).resolve()

    def execute(
        self,
        request: VisualContentAnalysisRequest,
    ) -> CapabilityResult[VisualDescriptionManifest]:
        started = perf_counter()
        model_request = VisualModelRequest(
            operation_id=request.context.operation_id,
            mode=request.mode.value,
            question=request.question,
            frames=tuple(
                VisualModelFrame(frame.frame_id, frame.image.uri, frame.timestamp_ms)
                for frame in request.frames.frames
                if any(frame.frame_id in target.frame_ids for target in request.targets)
            ),
            targets=tuple(
                VisualModelTarget(
                    target.target_id,
                    target.time_range.start_ms,
                    target.time_range.end_ms,
                    target.frame_ids,
                )
                for target in request.targets
            ),
        )
        try:
            response = self._backend.analyze(model_request)
            observations = self._validate_response(request, response.observations)
        except Exception as error:
            return CapabilityResult(
                status=CapabilityStatus.FAILED,
                data=None,
                error=CapabilityError("VISUAL_ANALYSIS_FAILED", str(error), "visual_model", True),
                usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
            )
        target_by_id = {target.target_id: target for target in request.targets}
        descriptions = tuple(
            VisualDescription(
                description_id=f"visual_{request.context.operation_id}_{index:06d}",
                video_id=request.frames.video_id,
                time_range=target_by_id[item.target_id].time_range,
                text=item.text,
                mode=request.mode,
                frame_ids=item.frame_ids,
                tags=item.tags,
                confidence=item.confidence,
                question=request.question,
            )
            for index, item in enumerate(observations)
        )
        provenance = make_provenance(
            "visual-content-analysis",
            self.VERSION,
            {"request": request, "model": response.model},
            video_id=request.frames.video_id,
            source_artifact_ids=(request.frames.ref.artifact.artifact_id,),
        )
        manifest_id = f"visual_descriptions_{request.context.operation_id}"
        path = (
            self._output_root
            / "visual_descriptions"
            / request.frames.video_id
            / f"{manifest_id}.json"
        )
        ref = manifest_ref(
            path,
            manifest_id=manifest_id,
            kind=ManifestKind.VISUAL_DESCRIPTIONS,
            video_id=request.frames.video_id,
            item_count=len(descriptions),
            provenance=provenance,
        )
        manifest = VisualDescriptionManifest(ref, request.frames.video_id, descriptions)
        write_json(path, manifest)
        missing = len(request.targets) - len(descriptions)
        status = CapabilityStatus.SUCCESS if missing == 0 else CapabilityStatus.PARTIAL
        warnings = (*response.warnings,)
        if missing:
            warnings += (f"Visual model returned no observation for {missing} target(s).",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(ref.artifact,),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(model_request.frames),
                output_items=len(descriptions),
                processed_duration_ms=sum(item.time_range.duration_ms for item in request.targets),
                returned_frames=len(model_request.frames),
                decoded_frames=len(model_request.frames),
                model_calls=response.model_calls,
            ),
            provenance=provenance,
        )

    @staticmethod
    def _validate_response(
        request: VisualContentAnalysisRequest,
        observations: tuple[VisualModelObservation, ...],
    ) -> tuple[VisualModelObservation, ...]:
        known_targets = {target.target_id: target for target in request.targets}
        seen: set[str] = set()
        validated: list[VisualModelObservation] = []
        for observation in observations:
            if observation.target_id not in known_targets or observation.target_id in seen:
                raise ValueError("visual backend returned an unknown or duplicate target")
            target = known_targets[observation.target_id]
            if not observation.text.strip():
                raise ValueError("visual observations must contain text")
            if not set(observation.frame_ids).issubset(target.frame_ids):
                raise ValueError("visual observation references frames outside its target")
            if any(not tag.strip() for tag in observation.tags):
                raise ValueError("visual observation tags must not be empty")
            if len(set(observation.tags)) != len(observation.tags):
                raise ValueError("visual observation tags must be unique")
            if observation.confidence is not None and not 0 <= observation.confidence <= 1:
                raise ValueError("visual observation confidence must be between zero and one")
            seen.add(observation.target_id)
            validated.append(observation)
        return tuple(validated)

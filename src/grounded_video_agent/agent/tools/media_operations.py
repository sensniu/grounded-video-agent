"""Shared frame, VLM, and OCR operations used by multiple high-level tools."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.agent.tools._support import add_usage
from grounded_video_agent.agent.tools.contracts import VisualDetail
from grounded_video_agent.agent.tools.dependencies import VideoToolDependencies
from grounded_video_agent.agent.tools.resolution import (
    ResolvedTarget,
    load_shots,
    sampling_ranges,
)
from grounded_video_agent.agent.tools.runtime import ToolRuntimeContext, fingerprint
from grounded_video_agent.capabilities.ocr.extraction import OCRExtractionRequest
from grounded_video_agent.capabilities.visual.content_analysis import (
    VisualContentAnalysisRequest,
)
from grounded_video_agent.capabilities.visual.frame_sampling import FrameSamplingRequest
from grounded_video_agent.domain import (
    CapabilityError,
    CapabilityStatus,
    CapabilityUsage,
    FrameManifest,
    FrameSamplingStrategy,
    ManifestRef,
    OCRManifest,
    TimeRange,
    VisualAnalysisTarget,
    VisualDescriptionManifest,
    VisualDescriptionMode,
)
from grounded_video_agent.workspace.catalog import CatalogError


@dataclass(frozen=True, slots=True)
class FrameRun:
    manifest: FrameManifest | None
    status: CapabilityStatus
    usage: CapabilityUsage
    warnings: tuple[str, ...]
    error: CapabilityError | None
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class VisualRun:
    frames: FrameManifest | None
    descriptions: VisualDescriptionManifest | None
    status: CapabilityStatus
    usage: CapabilityUsage
    warnings: tuple[str, ...]
    error: CapabilityError | None
    reused_frames: bool
    reused_analysis: bool


@dataclass(frozen=True, slots=True)
class OCRRun:
    frames: FrameManifest | None
    ocr: OCRManifest | None
    status: CapabilityStatus
    usage: CapabilityUsage
    warnings: tuple[str, ...]
    error: CapabilityError | None
    reused_frames: bool
    reused_ocr: bool


class FrameProvider:
    def __init__(self, dependencies: VideoToolDependencies) -> None:
        self._dependencies = dependencies

    def sample(
        self,
        runtime: ToolRuntimeContext,
        call_id: str,
        targets: tuple[ResolvedTarget, ...],
        detail: VisualDetail,
    ) -> FrameRun:
        ranges = sampling_ranges(targets)
        key = f"frames:{fingerprint((runtime.video_id, ranges, detail))}"
        cached = runtime.memory.get(key, FrameManifest)
        if cached is not None:
            return FrameRun(
                cached,
                CapabilityStatus.SUCCESS,
                CapabilityUsage(),
                (),
                None,
                True,
            )
        try:
            snapshot = runtime.catalog.get_snapshot(runtime.video_id)
            shots = load_shots(runtime)
        except CatalogError as error:
            return FrameRun(
                None,
                CapabilityStatus.FAILED,
                CapabilityUsage(),
                (),
                CapabilityError(
                    "FRAME_SOURCE_UNAVAILABLE",
                    str(error),
                    "catalog",
                ),
                False,
            )
        strategy, fps, max_frames = self._sampling(detail, len(targets))
        result = self._dependencies.frame_sampler.execute(
            FrameSamplingRequest(
                snapshot.video_asset,
                ranges,
                strategy,
                runtime.capability_context(call_id, "frames"),
                max_frames=max_frames,
                fps=fps,
                shots=shots,
            )
        )
        if result.data is not None:
            runtime.memory.put(key, result.data)
        return FrameRun(
            result.data,
            result.status,
            result.usage,
            result.warnings,
            result.error,
            False,
        )

    @staticmethod
    def _sampling(
        detail: VisualDetail,
        target_count: int,
    ) -> tuple[FrameSamplingStrategy, float | None, int]:
        if detail is VisualDetail.COARSE:
            return FrameSamplingStrategy.SHOT_KEYFRAME, None, min(16, max(4, target_count * 2))
        if detail is VisualDetail.DETAILED:
            return FrameSamplingStrategy.DENSE_WINDOW, 1.0, min(48, max(12, target_count * 8))
        return FrameSamplingStrategy.SHOT_KEYFRAME, None, min(24, max(8, target_count * 4))


class VisualOperation:
    def __init__(self, dependencies: VideoToolDependencies, frames: FrameProvider) -> None:
        self._dependencies = dependencies
        self._frames = frames

    def inspect(
        self,
        runtime: ToolRuntimeContext,
        call_id: str,
        targets: tuple[ResolvedTarget, ...],
        question: str,
        detail: VisualDetail,
    ) -> VisualRun:
        if self._dependencies.visual_analyzer is None:
            return VisualRun(
                None,
                None,
                CapabilityStatus.FAILED,
                CapabilityUsage(),
                (),
                CapabilityError(
                    "VISUAL_ANALYZER_UNAVAILABLE",
                    "No visual model backend is configured.",
                    "tool_configuration",
                ),
                False,
                False,
            )
        sampled = self._frames.sample(runtime, call_id, targets, detail)
        if sampled.status is CapabilityStatus.FAILED or sampled.manifest is None:
            return VisualRun(
                sampled.manifest,
                None,
                sampled.status,
                sampled.usage,
                sampled.warnings,
                sampled.error,
                sampled.cache_hit,
                False,
            )
        known_frames = sampled.manifest.frames
        analysis_targets = tuple(
            VisualAnalysisTarget(
                target.target_id,
                runtime.video_id,
                target.time_range,
                tuple(
                    frame.frame_id
                    for frame in known_frames
                    if target.time_range.contains_timestamp(frame.timestamp_ms)
                ),
                chunk_id=target.chunk_id,
            )
            for target in targets
            if any(
                target.time_range.contains_timestamp(frame.timestamp_ms)
                for frame in known_frames
            )
        )
        if not analysis_targets:
            return VisualRun(
                sampled.manifest,
                None,
                CapabilityStatus.PARTIAL,
                sampled.usage,
                (*sampled.warnings, "No sampled frame belongs to a requested target."),
                None,
                sampled.cache_hit,
                False,
            )
        analysis_key = "visual:" + fingerprint(
            (
                runtime.video_id,
                tuple(
                    (item.target_id, item.time_range, item.frame_ids)
                    for item in analysis_targets
                ),
                question,
            )
        )
        cached = runtime.memory.get(analysis_key, VisualDescriptionManifest)
        if cached is not None:
            return VisualRun(
                sampled.manifest,
                cached,
                CapabilityStatus.SUCCESS,
                sampled.usage,
                sampled.warnings,
                None,
                sampled.cache_hit,
                True,
            )
        analyzed = self._dependencies.visual_analyzer.execute(
            VisualContentAnalysisRequest(
                sampled.manifest,
                analysis_targets,
                VisualDescriptionMode.QUESTION_CONDITIONED,
                runtime.capability_context(call_id, "visual"),
                question=question,
            )
        )
        if analyzed.data is not None:
            runtime.memory.put(analysis_key, analyzed.data)
        return VisualRun(
            sampled.manifest,
            analyzed.data,
            analyzed.status,
            add_usage(sampled.usage, analyzed.usage),
            (*sampled.warnings, *analyzed.warnings),
            analyzed.error,
            sampled.cache_hit,
            False,
        )


class OCROperation:
    def __init__(self, dependencies: VideoToolDependencies, frames: FrameProvider) -> None:
        self._dependencies = dependencies
        self._frames = frames

    def read(
        self,
        runtime: ToolRuntimeContext,
        call_id: str,
        targets: tuple[ResolvedTarget, ...],
        detail: VisualDetail,
        *,
        frame_ids: tuple[str, ...] = (),
        language: str | None = None,
        min_confidence: float = 0.5,
    ) -> OCRRun:
        if self._dependencies.ocr_extractor is None:
            return OCRRun(
                None,
                None,
                CapabilityStatus.FAILED,
                CapabilityUsage(),
                (),
                CapabilityError(
                    "OCR_EXTRACTOR_UNAVAILABLE",
                    "No OCR backend is configured.",
                    "tool_configuration",
                ),
                False,
                False,
            )
        if frame_ids:
            manifest = runtime.memory.frame_manifest_for(frame_ids)
            if manifest is not None:
                manifest = self._select_frames(manifest, frame_ids)
            sampled = FrameRun(
                manifest,
                CapabilityStatus.SUCCESS if manifest is not None else CapabilityStatus.FAILED,
                CapabilityUsage(),
                (),
                (
                    None
                    if manifest is not None
                    else CapabilityError(
                        "UNKNOWN_FRAME_IDS",
                        "Requested frame ids are not available in runtime memory.",
                        "target_resolution",
                    )
                ),
                True,
            )
        else:
            sampled = self._frames.sample(runtime, call_id, targets, detail)
        if sampled.status is CapabilityStatus.FAILED or sampled.manifest is None:
            return OCRRun(
                sampled.manifest,
                None,
                sampled.status,
                sampled.usage,
                sampled.warnings,
                sampled.error,
                sampled.cache_hit,
                False,
            )
        ocr_key = "ocr:" + fingerprint(
            (sampled.manifest.ref.manifest_id, frame_ids, language, min_confidence)
        )
        cached = runtime.memory.get(ocr_key, OCRManifest)
        if cached is not None:
            return OCRRun(
                sampled.manifest,
                cached,
                CapabilityStatus.SUCCESS,
                sampled.usage,
                sampled.warnings,
                None,
                sampled.cache_hit,
                True,
            )
        extracted = self._dependencies.ocr_extractor.execute(
            OCRExtractionRequest(
                sampled.manifest,
                runtime.capability_context(call_id, "ocr"),
                language=language,
                min_confidence=min_confidence,
            )
        )
        if extracted.data is not None:
            runtime.memory.put(ocr_key, extracted.data)
        return OCRRun(
            sampled.manifest,
            extracted.data,
            extracted.status,
            add_usage(sampled.usage, extracted.usage),
            (*sampled.warnings, *extracted.warnings),
            extracted.error,
            sampled.cache_hit,
            False,
        )

    @staticmethod
    def _select_frames(
        manifest: FrameManifest,
        frame_ids: tuple[str, ...],
    ) -> FrameManifest:
        wanted = set(frame_ids)
        frames = tuple(frame for frame in manifest.frames if frame.frame_id in wanted)
        requested_ranges = tuple(
            TimeRange(frame.timestamp_ms, frame.timestamp_ms + 1) for frame in frames
        )
        ref = ManifestRef(
            manifest_id=f"{manifest.ref.manifest_id}_selection_{fingerprint(frame_ids)[:12]}",
            kind=manifest.ref.kind,
            artifact=manifest.ref.artifact,
            source_video_id=manifest.video_id,
            item_count=len(frames),
        )
        return FrameManifest(
            ref,
            manifest.video_id,
            manifest.strategy,
            requested_ranges,
            frames,
            decoded_frames=len(frames),
        )

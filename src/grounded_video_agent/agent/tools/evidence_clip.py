from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from grounded_video_agent.agent.tools._support import (
    SCHEMA_VERSION,
    add_usage,
    failed_result,
    start_tool,
)
from grounded_video_agent.agent.tools.contracts import (
    ClipExportFailure,
    ClipGrouping,
    EvidenceClipDelivery,
    EvidenceDelta,
    ExportEvidenceClipInput,
    ExportEvidenceClipOutput,
    ToolProgress,
    ToolResult,
    ToolStatus,
)
from grounded_video_agent.agent.tools.dependencies import VideoToolDependencies
from grounded_video_agent.agent.tools.runtime import (
    DeliveryState,
    ToolRuntimeContext,
    fingerprint,
    stable_id,
)
from grounded_video_agent.capabilities.visual.clip_export import ClipExportRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityStatus,
    CapabilityUsage,
    ProducerInfo,
    Provenance,
    TimeRange,
    VideoClipArtifact,
)
from grounded_video_agent.pipelines.preprocessing.keys import (
    MEDIA_INSPECTION_KEY,
    SOURCE_KEY,
)
from grounded_video_agent.workspace.catalog import (
    CatalogDocumentKind,
    CatalogDocumentRef,
    CatalogError,
    CatalogKey,
    CatalogRegistration,
    CatalogResourceType,
    DocumentCodecRegistry,
    MediaInspectionDocument,
    VideoClipDocument,
)


@dataclass(frozen=True, slots=True)
class _ClipSelection:
    time_range: TimeRange
    evidence_ids: tuple[str, ...]


class ExportEvidenceClipTool:
    name = "export_evidence_clip"
    description = (
        "Export downloadable video clips for evidence that has already been verified. "
        "This tool is available only when the user explicitly requested evidence clips."
    )
    input_type = ExportEvidenceClipInput
    enabled = True
    runtime_guarded = True

    def __init__(
        self,
        dependencies: VideoToolDependencies,
        artifact_root: str | Path = "artifacts",
        *,
        document_codecs: DocumentCodecRegistry | None = None,
    ) -> None:
        self._dependencies = dependencies
        self._artifact_root = Path(artifact_root).resolve()
        self._document_codecs = document_codecs or DocumentCodecRegistry()

    def is_available(self, runtime: ToolRuntimeContext) -> bool:
        policy = runtime.delivery_policy
        return policy.evidence_clip_requested and bool(policy.verified_evidence_ids)

    def execute(
        self,
        request: ExportEvidenceClipInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[ExportEvidenceClipOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[ExportEvidenceClipOutput], early)
        assert call_id is not None
        authorization_error = self._authorization_error(request, runtime)
        if authorization_error is not None:
            return cast(
                ToolResult[ExportEvidenceClipOutput],
                failed_result(
                    call_id,
                    authorization_error[0],
                    authorization_error[1],
                    suggested_action=authorization_error[2],
                ),
            )
        try:
            inspection = runtime.catalog.load_document(
                runtime.video_id,
                MEDIA_INSPECTION_KEY,
                MediaInspectionDocument,
            )
            source_entry = runtime.catalog.resolve(runtime.video_id, SOURCE_KEY).entry
            selections = self._resolve_selections(request, runtime, inspection)
        except CatalogError as error:
            return cast(
                ToolResult[ExportEvidenceClipOutput],
                failed_result(
                    call_id,
                    f"CATALOG_{error.code.value.upper()}",
                    str(error),
                ),
            )
        except KeyError as error:
            return cast(
                ToolResult[ExportEvidenceClipOutput],
                failed_result(call_id, "EVIDENCE_NOT_FOUND", str(error)),
            )
        except ValueError as error:
            return cast(
                ToolResult[ExportEvidenceClipOutput],
                failed_result(call_id, "CONTEXT_NOT_RELATED", str(error)),
            )
        limit_error = self._limit_error(request, selections, runtime)
        if limit_error is not None:
            return cast(
                ToolResult[ExportEvidenceClipOutput],
                failed_result(
                    call_id,
                    "EXPORT_LIMIT_EXCEEDED",
                    limit_error,
                    suggested_action="Select fewer evidence items or narrower context windows.",
                ),
            )

        include_audio = request.include_audio and inspection.basic_flags.has_audio
        warnings: list[str] = []
        if request.include_audio and not include_audio:
            warnings.append("The source video has no audio stream; clips will be silent.")
        deliveries: list[EvidenceClipDelivery] = []
        failures: list[ClipExportFailure] = []
        usages: list[CapabilityUsage] = []
        for index, selection in enumerate(selections):
            try:
                document, entry_id, reused, usage, item_warnings = self._export_one(
                    runtime,
                    call_id,
                    index,
                    selection,
                    include_audio,
                    source_entry.entry_id,
                )
            except CatalogError as error:
                failures.append(
                    ClipExportFailure(
                        selection.evidence_ids,
                        f"CATALOG_{error.code.value.upper()}",
                        str(error),
                    )
                )
                continue
            except RuntimeError as error:
                failures.append(
                    ClipExportFailure(
                        selection.evidence_ids,
                        "CLIP_EXPORT_FAILED",
                        str(error),
                    )
                )
                continue
            except (OSError, ValueError) as error:
                failures.append(
                    ClipExportFailure(
                        selection.evidence_ids,
                        "CLIP_PUBLICATION_FAILED",
                        str(error),
                    )
                )
                continue
            usages.append(usage)
            warnings.extend(item_warnings)
            clip = document.video_clip
            delivery_id = stable_id(
                "delivery",
                (runtime.video_id, clip.artifact.artifact_id, selection.evidence_ids),
            )
            runtime.deliveries.put(
                DeliveryState(
                    delivery_id,
                    entry_id,
                    clip.artifact,
                    selection.evidence_ids,
                )
            )
            deliveries.append(
                EvidenceClipDelivery(
                    delivery_id,
                    clip.artifact.artifact_id,
                    entry_id,
                    Path(clip.artifact.uri).name,
                    clip.requested_range,
                    clip.actual_range,
                    clip.actual_range.duration_ms,
                    clip.includes_audio,
                    selection.evidence_ids,
                    clip.artifact.size_bytes or 0,
                    reused,
                )
            )

        usage = add_usage(*usages)
        runtime.record_usage(usage)
        if not deliveries:
            message = "; ".join(item.message for item in failures) or "No clips were exported."
            return cast(
                ToolResult[ExportEvidenceClipOutput],
                failed_result(call_id, "CLIP_EXPORT_FAILED", message, usage=usage),
            )
        output = ExportEvidenceClipOutput(
            stable_id(
                "export",
                (runtime.video_id, tuple(item.delivery_id for item in deliveries)),
            ),
            tuple(deliveries),
            tuple(failures),
            sum(item.duration_ms for item in deliveries),
        )
        status = ToolStatus.PARTIAL if failures else ToolStatus.SUCCESS
        return ToolResult(
            SCHEMA_VERSION,
            call_id,
            status,
            output,
            EvidenceDelta(reused_evidence_ids=request.evidence_ids),
            ToolProgress(cache_hit=all(item.reused for item in deliveries)),
            tuple(dict.fromkeys(warnings)),
            usage=usage,
        )

    @staticmethod
    def _authorization_error(
        request: ExportEvidenceClipInput,
        runtime: ToolRuntimeContext,
    ) -> tuple[str, str, str] | None:
        policy = runtime.delivery_policy
        if not policy.evidence_clip_requested:
            return (
                "CLIP_EXPORT_NOT_AUTHORIZED",
                "The user did not explicitly request an evidence video clip.",
                "Do not export a clip; answer with normal citations.",
            )
        unverified = tuple(
            item for item in request.evidence_ids if item not in policy.verified_evidence_ids
        )
        if unverified:
            return (
                "EVIDENCE_NOT_VERIFIED",
                f"Evidence has not been verified: {', '.join(unverified)}",
                "Run deterministic evidence verification before exporting.",
            )
        return None

    def _resolve_selections(
        self,
        request: ExportEvidenceClipInput,
        runtime: ToolRuntimeContext,
        inspection: MediaInspectionDocument,
    ) -> tuple[_ClipSelection, ...]:
        evidence = {item: runtime.evidence.get(item) for item in request.evidence_ids}
        if any(item.video_id != runtime.video_id for item in evidence.values()):
            raise ValueError("all evidence must belong to the current video")
        duration_ms = inspection.media_probe.container.duration_ms
        if duration_ms is None or duration_ms <= 0:
            raise ValueError("source video duration is unavailable")

        selected: list[_ClipSelection] = []
        covered_evidence_ids: set[str] = set()
        for context_id in request.context_window_ids:
            context = runtime.contexts.get(context_id)
            related = tuple(
                sorted(item for item in request.evidence_ids if item in context.evidence_ids)
            )
            if not related:
                raise ValueError(
                    f"context_window_id does not reference selected evidence: {context_id}"
                )
            covered_evidence_ids.update(related)
            selected.extend(_ClipSelection(item, related) for item in context.ranges)
        selected.extend(
            _ClipSelection(item.time_range, (evidence_id,))
            for evidence_id, item in evidence.items()
            if evidence_id not in covered_evidence_ids
        )
        padded = tuple(
            _ClipSelection(
                TimeRange(
                    max(0, item.time_range.start_ms - request.padding_before_ms),
                    min(duration_ms, item.time_range.end_ms + request.padding_after_ms),
                ),
                item.evidence_ids,
            )
            for item in selected
        )
        ordered = tuple(sorted(padded, key=lambda item: item.time_range))
        if request.grouping is ClipGrouping.SEPARATE:
            return ordered
        return self._merge(ordered, runtime.delivery_policy.merge_gap_ms)

    @staticmethod
    def _merge(
        selections: tuple[_ClipSelection, ...],
        merge_gap_ms: int,
    ) -> tuple[_ClipSelection, ...]:
        merged: list[_ClipSelection] = []
        for item in selections:
            if not merged or item.time_range.start_ms > (
                merged[-1].time_range.end_ms + merge_gap_ms
            ):
                merged.append(item)
                continue
            previous = merged[-1]
            merged[-1] = _ClipSelection(
                TimeRange(
                    previous.time_range.start_ms,
                    max(previous.time_range.end_ms, item.time_range.end_ms),
                ),
                tuple(sorted(set((*previous.evidence_ids, *item.evidence_ids)))),
            )
        return tuple(merged)

    @staticmethod
    def _limit_error(
        request: ExportEvidenceClipInput,
        selections: tuple[_ClipSelection, ...],
        runtime: ToolRuntimeContext,
    ) -> str | None:
        policy = runtime.delivery_policy
        if (
            request.padding_before_ms > policy.max_padding_ms
            or request.padding_after_ms > policy.max_padding_ms
        ):
            return f"Clip padding cannot exceed {policy.max_padding_ms} ms."
        if len(selections) > policy.max_clips:
            return f"At most {policy.max_clips} clips may be exported."
        oversized = tuple(
            item for item in selections if item.time_range.duration_ms > policy.max_clip_duration_ms
        )
        if oversized:
            return f"A clip cannot exceed {policy.max_clip_duration_ms} ms."
        total = sum(item.time_range.duration_ms for item in selections)
        if total > policy.max_total_duration_ms:
            return f"Total exported duration cannot exceed {policy.max_total_duration_ms} ms."
        return None

    def _export_one(
        self,
        runtime: ToolRuntimeContext,
        call_id: str,
        index: int,
        selection: _ClipSelection,
        include_audio: bool,
        source_entry_id: str,
    ) -> tuple[VideoClipDocument, str, bool, CapabilityUsage, tuple[str, ...]]:
        derivation_key = fingerprint(
            (
                runtime.video_id,
                selection,
                include_audio,
                True,
                self._dependencies.clip_exporter.VERSION,
            )
        )
        key = CatalogKey(
            CatalogResourceType.DOCUMENT,
            variant=f"evidence_{derivation_key[:24]}",
            document_kind=CatalogDocumentKind.VIDEO_CLIP,
        )
        dependencies = (source_entry_id,)
        reusable = runtime.catalog.find_reusable(
            runtime.video_id,
            key,
            derivation_key,
            dependencies,
        )
        if reusable is not None:
            runtime.catalog.activate(
                runtime.video_id,
                key,
                reusable.entry.entry_id,
            )
            document = runtime.catalog.load_document(
                runtime.video_id,
                key,
                VideoClipDocument,
            )
            return document, reusable.entry.entry_id, True, CapabilityUsage(), ()

        snapshot = runtime.catalog.get_snapshot(runtime.video_id)
        result = self._dependencies.clip_exporter.execute(
            ClipExportRequest(
                snapshot.video_asset,
                selection.time_range,
                runtime.capability_context(call_id, f"clip_{index:03d}"),
                include_audio=include_audio,
                reencode=True,
            )
        )
        if result.status is CapabilityStatus.FAILED or result.data is None:
            message = result.error.message if result.error is not None else "Clip export failed."
            raise RuntimeError(message)
        document, entry_id = self._publish(
            runtime,
            key,
            result.data,
            selection.evidence_ids,
            call_id,
            index,
            derivation_key,
            dependencies,
        )
        return document, entry_id, False, result.usage, result.warnings

    def _publish(
        self,
        runtime: ToolRuntimeContext,
        key: CatalogKey,
        clip: VideoClipArtifact,
        evidence_ids: tuple[str, ...],
        call_id: str,
        index: int,
        derivation_key: str,
        dependencies: tuple[str, ...],
    ) -> tuple[VideoClipDocument, str]:
        provenance = clip.artifact.provenance or Provenance(
            ProducerInfo("clip-export", self._dependencies.clip_exporter.VERSION),
            derivation_key,
            runtime.video_id,
            (runtime.catalog.get_snapshot(runtime.video_id).video_asset.source.artifact_id,),
        )
        document_id = f"clip_document_{call_id}_{index:03d}"
        path = self._artifact_root / "metadata" / runtime.video_id / f"{document_id}.json"
        ref = CatalogDocumentRef(
            document_id,
            CatalogDocumentKind.VIDEO_CLIP,
            ArtifactRef(
                f"{document_id}_artifact",
                ArtifactKind.METADATA,
                str(path),
                provenance=provenance,
            ),
            runtime.video_id,
        )
        document = VideoClipDocument(ref, clip, evidence_ids)
        self._document_codecs.dump(path, document)
        runtime.catalog.register(
            runtime.video_id,
            CatalogRegistration(
                key,
                ref,
                call_id,
                dependencies,
                producer_name="evidence-clip-tool",
                producer_version="1.0.0",
                parameters_hash=derivation_key,
                derivation_key=derivation_key,
            ),
        )
        entry = runtime.catalog.resolve(runtime.video_id, key).entry
        return document, entry.entry_id

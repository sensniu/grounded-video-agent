from __future__ import annotations

from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import (
    file_artifact,
    make_provenance,
    manifest_ref,
    write_json,
)
from grounded_video_agent.capabilities.indexing._bm25 import (
    TextIndexDocument,
    unique_values,
    write_index,
)
from grounded_video_agent.capabilities.indexing.visual_index.contracts import VisualIndexingRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    IndexManifest,
    IndexModality,
    ManifestKind,
    VisualDescription,
)


class VisualIndexingCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts") -> None:
        self._output_root = Path(output_root).resolve()

    def execute(self, request: VisualIndexingRequest) -> CapabilityResult[IndexManifest]:
        started = perf_counter()
        video_id = request.descriptions.video_id
        documents = tuple(
            TextIndexDocument(
                item_id=item.description_id,
                text=item.text,
                time_range=item.time_range,
                source_ids=unique_values(
                    (
                        item.description_id,
                        *item.frame_ids,
                        *self._related_ids(item, request),
                    )
                ),
                tags=item.tags,
            )
            for item in request.descriptions.descriptions
        )
        provenance = make_provenance(
            "visual-indexing",
            self.VERSION,
            request,
            video_id=video_id,
            source_artifact_ids=(request.descriptions.ref.artifact.artifact_id,),
        )
        index_id = f"visual_index_{request.context.operation_id}"
        directory = self._output_root / "indexes" / video_id
        index_path = directory / f"{index_id}.json"
        write_index(
            index_path,
            video_id=video_id,
            modality=IndexModality.VISUAL_DESCRIPTION.value,
            documents=documents,
        )
        index_artifact = file_artifact(
            index_path,
            artifact_id=f"{index_id}_data",
            kind=ArtifactKind.INDEX,
            provenance=provenance,
        )
        source_manifest_ids = [request.descriptions.ref.manifest_id]
        source_manifest_ids.extend(
            manifest.ref.manifest_id
            for manifest in (request.chunks, request.shots)
            if manifest is not None
        )
        manifest_path = directory / f"{index_id}.manifest.json"
        ref = manifest_ref(
            manifest_path,
            manifest_id=index_id,
            kind=ManifestKind.INDEX,
            video_id=video_id,
            item_count=len(documents),
            provenance=provenance,
        )
        manifest = IndexManifest(
            ref=ref,
            video_id=video_id,
            modality=IndexModality.VISUAL_DESCRIPTION,
            index_kind=request.index_kind,
            source_manifest_ids=tuple(source_manifest_ids),
            index_artifact=index_artifact,
        )
        write_json(manifest_path, manifest)
        status = CapabilityStatus.SUCCESS if documents else CapabilityStatus.PARTIAL
        warnings = () if documents else ("No visual descriptions were available to index.",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(index_artifact, ref.artifact),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(request.descriptions.descriptions),
                output_items=len(documents),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _related_ids(
        item: VisualDescription,
        request: VisualIndexingRequest,
    ) -> tuple[str, ...]:
        identifiers: list[str] = []
        if request.chunks is not None:
            identifiers.extend(
                chunk.chunk_id
                for chunk in request.chunks.chunks
                if chunk.time_range.overlaps(item.time_range)
            )
        if request.shots is not None:
            identifiers.extend(
                shot.shot_id
                for shot in request.shots.shots
                if shot.time_range.overlaps(item.time_range)
            )
        return tuple(identifiers)

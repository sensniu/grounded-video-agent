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
from grounded_video_agent.capabilities.indexing.transcript_index.contracts import (
    TranscriptIndexingRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    IndexManifest,
    IndexModality,
    ManifestKind,
)


class TranscriptIndexingCapability:
    VERSION = "2.0.0"

    def __init__(self, output_root: str | Path = "artifacts") -> None:
        self._output_root = Path(output_root).resolve()

    def execute(self, request: TranscriptIndexingRequest) -> CapabilityResult[IndexManifest]:
        started = perf_counter()
        video_id = request.transcript.video_id
        documents = self._documents(request)
        provenance = make_provenance(
            "transcript-indexing",
            self.VERSION,
            request,
            video_id=video_id,
            source_artifact_ids=(request.transcript.ref.artifact.artifact_id,),
        )
        index_id = f"transcript_index_{request.context.operation_id}"
        directory = self._output_root / "indexes" / video_id
        index_path = directory / f"{index_id}.json"
        write_index(
            index_path,
            video_id=video_id,
            modality=IndexModality.TRANSCRIPT.value,
            documents=documents,
        )
        index_artifact = file_artifact(
            index_path,
            artifact_id=f"{index_id}_data",
            kind=ArtifactKind.INDEX,
            provenance=provenance,
        )
        manifest_path = directory / f"{index_id}.manifest.json"
        source_manifest_ids = [request.transcript.ref.manifest_id]
        if request.chunks is not None:
            source_manifest_ids.append(request.chunks.ref.manifest_id)
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
            modality=IndexModality.TRANSCRIPT,
            index_kind=request.index_kind,
            source_manifest_ids=tuple(source_manifest_ids),
            index_artifact=index_artifact,
        )
        write_json(manifest_path, manifest)
        status = CapabilityStatus.SUCCESS if documents else CapabilityStatus.PARTIAL
        warnings = () if documents else ("Transcript contains no indexable text.",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(index_artifact, ref.artifact),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(request.transcript.segments),
                output_items=len(documents),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _documents(request: TranscriptIndexingRequest) -> tuple[TextIndexDocument, ...]:
        if request.chunks is not None:
            chunk_documents = tuple(
                TextIndexDocument(
                    item_id=chunk.chunk_id,
                    text=chunk.text,
                    time_range=chunk.time_range,
                    source_ids=unique_values(
                        (chunk.chunk_id, *chunk.transcript_segment_ids, *chunk.shot_ids)
                    ),
                    tags=unique_values(
                        tuple(
                            value
                            for segment in request.transcript.segments
                            if segment.segment_id in chunk.transcript_segment_ids
                            for value in (segment.language, segment.source.value)
                            if value
                        )
                    ),
                )
                for chunk in request.chunks.chunks
                if chunk.text is not None
            )
            if chunk_documents:
                return chunk_documents
        return tuple(
            TextIndexDocument(
                item_id=segment.segment_id,
                text=segment.normalized_text,
                time_range=segment.time_range,
                source_ids=unique_values(
                    (
                        segment.segment_id,
                        *TranscriptIndexingCapability._chunk_ids(segment.segment_id, request),
                    )
                ),
                tags=tuple(item for item in (segment.language, segment.source.value) if item),
            )
            for segment in request.transcript.segments
        )

    @staticmethod
    def _chunk_ids(segment_id: str, request: TranscriptIndexingRequest) -> tuple[str, ...]:
        if request.chunks is None:
            return ()
        return tuple(
            chunk.chunk_id
            for chunk in request.chunks.chunks
            if segment_id in chunk.transcript_segment_ids
        )

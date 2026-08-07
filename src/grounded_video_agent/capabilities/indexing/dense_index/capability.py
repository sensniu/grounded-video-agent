from __future__ import annotations

from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import (
    file_artifact,
    make_provenance,
    manifest_ref,
    write_json,
)
from grounded_video_agent.capabilities.indexing._bm25 import TextIndexDocument, unique_values
from grounded_video_agent.capabilities.indexing._dense import write_dense_index
from grounded_video_agent.capabilities.indexing.dense_index.contracts import DenseIndexingRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    EmbeddingManifest,
    IndexKind,
    IndexManifest,
    IndexModality,
    ManifestKind,
    OCRManifest,
    TimeRange,
    TranscriptManifest,
    VisualDescriptionManifest,
)
from grounded_video_agent.infrastructure.embeddings import EmbeddingModelInfo, TextEmbeddingBackend


class DenseIndexingCapability:
    VERSION = "2.0.0"

    def __init__(
        self,
        backend: TextEmbeddingBackend,
        output_root: str | Path = "artifacts",
    ) -> None:
        self._backend = backend
        self._output_root = Path(output_root).resolve()

    def get_model_info(self) -> EmbeddingModelInfo:
        return self._backend.get_model_info()

    def execute(self, request: DenseIndexingRequest) -> CapabilityResult[IndexManifest]:
        started = perf_counter()
        try:
            model_info = self._backend.get_model_info()
            modality, documents = self._documents(request)
            vectors = self._backend.embed_documents(tuple(item.text for item in documents))
        except Exception as error:
            return self._failure(str(error), started)
        video_id = request.source.video_id
        provenance = make_provenance(
            "dense-indexing",
            self.VERSION,
            {"request": request, "model": model_info},
            video_id=video_id,
            source_artifact_ids=(request.source.ref.artifact.artifact_id,),
        )
        index_id = f"{modality.value}_dense_{request.context.operation_id}"
        directory = self._output_root / "indexes" / video_id
        embedding_path = directory / f"{index_id}.npy"
        index_path = directory / f"{index_id}.json"
        try:
            write_dense_index(
                index_path,
                embedding_path,
                video_id=video_id,
                modality=modality.value,
                embedding_space=model_info.embedding_space,
                dimensions=model_info.dimensions,
                documents=documents,
                vectors=vectors,
            )
        except (OSError, ValueError) as error:
            return self._failure(str(error), started)
        embedding_artifact = file_artifact(
            embedding_path,
            artifact_id=f"{index_id}_vectors",
            kind=ArtifactKind.EMBEDDING,
            provenance=provenance,
        )
        embedding_manifest_id = f"{index_id}_embeddings"
        embedding_manifest_path = directory / f"{embedding_manifest_id}.manifest.json"
        embedding_ref = manifest_ref(
            embedding_manifest_path,
            manifest_id=embedding_manifest_id,
            kind=ManifestKind.EMBEDDINGS,
            video_id=video_id,
            item_count=len(documents),
            provenance=provenance,
        )
        embedding_manifest = EmbeddingManifest(
            embedding_ref,
            video_id,
            modality,
            model_info.embedding_space,
            model_info.dimensions,
            tuple(item.item_id for item in documents),
            embedding_artifact,
        )
        write_json(embedding_manifest_path, embedding_manifest)
        index_artifact = file_artifact(
            index_path,
            artifact_id=f"{index_id}_data",
            kind=ArtifactKind.INDEX,
            provenance=provenance,
        )
        index_manifest_path = directory / f"{index_id}.manifest.json"
        source_manifest_ids = [request.source.ref.manifest_id]
        if request.chunks is not None:
            source_manifest_ids.append(request.chunks.ref.manifest_id)
        index_ref = manifest_ref(
            index_manifest_path,
            manifest_id=index_id,
            kind=ManifestKind.INDEX,
            video_id=video_id,
            item_count=len(documents),
            provenance=provenance,
        )
        index_manifest = IndexManifest(
            index_ref,
            video_id,
            modality,
            IndexKind.DENSE,
            tuple(source_manifest_ids),
            index_artifact,
            embedding_manifest_id,
            embedding_manifest,
        )
        write_json(index_manifest_path, index_manifest)
        status = CapabilityStatus.SUCCESS if documents else CapabilityStatus.PARTIAL
        warnings = () if documents else ("Source contains no dense-indexable text.",)
        return CapabilityResult(
            status=status,
            data=index_manifest,
            artifacts=(
                embedding_artifact,
                embedding_ref.artifact,
                index_artifact,
                index_ref.artifact,
            ),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(documents),
                output_items=len(documents),
                model_calls=1 if documents else 0,
            ),
            provenance=provenance,
        )

    def _documents(
        self,
        request: DenseIndexingRequest,
    ) -> tuple[IndexModality, tuple[TextIndexDocument, ...]]:
        source = request.source
        if isinstance(source, TranscriptManifest):
            if request.chunks is not None:
                chunk_documents = tuple(
                    TextIndexDocument(
                        item.chunk_id,
                        item.text,
                        item.time_range,
                        unique_values(
                            (item.chunk_id, *item.transcript_segment_ids, *item.shot_ids)
                        ),
                        self._transcript_chunk_tags(item.transcript_segment_ids, source),
                    )
                    for item in request.chunks.chunks
                    if item.text is not None
                )
                if chunk_documents:
                    return IndexModality.TRANSCRIPT, chunk_documents
            return IndexModality.TRANSCRIPT, tuple(
                TextIndexDocument(
                    item.segment_id,
                    item.normalized_text,
                    item.time_range,
                    unique_values(
                        (
                            item.segment_id,
                            *self._chunk_ids(item.time_range, request),
                        )
                    ),
                    tuple(value for value in (item.language, item.source.value) if value),
                )
                for item in source.segments
            )
        if isinstance(source, OCRManifest):
            observations = {item.observation_id: item for item in source.observations}
            if source.spans:
                return IndexModality.OCR, tuple(
                    TextIndexDocument(
                        item.span_id,
                        item.text,
                        item.time_range,
                        unique_values(
                            (
                                item.span_id,
                                *item.observation_ids,
                                *(observations[value].frame_id for value in item.observation_ids),
                                *self._chunk_ids(item.time_range, request),
                            )
                        ),
                        unique_values(
                            tuple(
                                language
                                for value in item.observation_ids
                                if (language := observations[value].language) is not None
                            )
                        ),
                    )
                    for item in source.spans
                )
            return IndexModality.OCR, tuple(
                TextIndexDocument(
                    item.observation_id,
                    item.normalized_text,
                    TimeRange(item.timestamp_ms, item.timestamp_ms + 1),
                    unique_values(
                        (
                            item.observation_id,
                            item.frame_id,
                            *self._chunk_ids(
                                TimeRange(item.timestamp_ms, item.timestamp_ms + 1),
                                request,
                            ),
                        )
                    ),
                    (item.language,) if item.language is not None else (),
                )
                for item in source.observations
            )
        if isinstance(source, VisualDescriptionManifest):
            return IndexModality.VISUAL_DESCRIPTION, tuple(
                TextIndexDocument(
                    item.description_id,
                    item.text,
                    item.time_range,
                    unique_values(
                        (
                            item.description_id,
                            *item.frame_ids,
                            *self._chunk_ids(item.time_range, request),
                        )
                    ),
                    item.tags,
                )
                for item in source.descriptions
            )
        raise TypeError("unsupported dense-index source")

    @staticmethod
    def _transcript_chunk_tags(
        segment_ids: tuple[str, ...],
        transcript: TranscriptManifest,
    ) -> tuple[str, ...]:
        return unique_values(
            tuple(
                value
                for segment in transcript.segments
                if segment.segment_id in segment_ids
                for value in (segment.language, segment.source.value)
                if value
            )
        )

    @staticmethod
    def _chunk_ids(time_range: TimeRange, request: DenseIndexingRequest) -> tuple[str, ...]:
        if request.chunks is None:
            return ()
        return tuple(
            chunk.chunk_id
            for chunk in request.chunks.chunks
            if chunk.time_range.overlaps(time_range)
        )

    @staticmethod
    def _failure(message: str, started: float) -> CapabilityResult[IndexManifest]:
        return CapabilityResult(
            status=CapabilityStatus.FAILED,
            data=None,
            error=CapabilityError("DENSE_INDEXING_FAILED", message, "embedding"),
            usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
        )

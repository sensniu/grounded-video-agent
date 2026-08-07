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
from grounded_video_agent.capabilities.indexing.ocr_index.contracts import OCRIndexingRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    IndexManifest,
    IndexModality,
    ManifestKind,
    TimeRange,
)


class OCRIndexingCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts") -> None:
        self._output_root = Path(output_root).resolve()

    def execute(self, request: OCRIndexingRequest) -> CapabilityResult[IndexManifest]:
        started = perf_counter()
        observations = {item.observation_id: item for item in request.ocr.observations}
        if request.ocr.spans:
            documents = tuple(
                TextIndexDocument(
                    span.span_id,
                    span.text,
                    span.time_range,
                    unique_values(
                        (
                            span.span_id,
                            *span.observation_ids,
                            *(observations[item].frame_id for item in span.observation_ids),
                            *self._chunk_ids(span.time_range, request),
                        )
                    ),
                    unique_values(
                        tuple(
                            language
                            for item in span.observation_ids
                            if (language := observations[item].language) is not None
                        )
                    ),
                )
                for span in request.ocr.spans
            )
        else:
            documents = tuple(
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
                for item in request.ocr.observations
            )
        provenance = make_provenance(
            "ocr-indexing",
            self.VERSION,
            request,
            video_id=request.ocr.video_id,
            source_artifact_ids=(request.ocr.ref.artifact.artifact_id,),
        )
        index_id = f"ocr_index_{request.context.operation_id}"
        directory = self._output_root / "indexes" / request.ocr.video_id
        index_path = directory / f"{index_id}.json"
        write_index(
            index_path,
            video_id=request.ocr.video_id,
            modality=IndexModality.OCR.value,
            documents=documents,
        )
        index_artifact = file_artifact(
            index_path,
            artifact_id=f"{index_id}_data",
            kind=ArtifactKind.INDEX,
            provenance=provenance,
        )
        manifest_path = directory / f"{index_id}.manifest.json"
        source_manifest_ids = [request.ocr.ref.manifest_id]
        if request.chunks is not None:
            source_manifest_ids.append(request.chunks.ref.manifest_id)
        ref = manifest_ref(
            manifest_path,
            manifest_id=index_id,
            kind=ManifestKind.INDEX,
            video_id=request.ocr.video_id,
            item_count=len(documents),
            provenance=provenance,
        )
        manifest = IndexManifest(
            ref,
            request.ocr.video_id,
            IndexModality.OCR,
            request.index_kind,
            tuple(source_manifest_ids),
            index_artifact,
        )
        write_json(manifest_path, manifest)
        status = CapabilityStatus.SUCCESS if documents else CapabilityStatus.PARTIAL
        warnings = () if documents else ("OCR records contain no indexable text.",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(index_artifact, ref.artifact),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(request.ocr.observations),
                output_items=len(documents),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _chunk_ids(time_range: TimeRange, request: OCRIndexingRequest) -> tuple[str, ...]:
        if request.chunks is None:
            return ()
        return tuple(
            chunk.chunk_id
            for chunk in request.chunks.chunks
            if chunk.time_range.overlaps(time_range)
        )

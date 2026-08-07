from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar
from uuid import uuid4

from grounded_video_agent.capabilities.audio.extraction import AudioExtractionRequest
from grounded_video_agent.capabilities.indexing.dense_index import DenseIndexingRequest
from grounded_video_agent.capabilities.indexing.transcript_index import TranscriptIndexingRequest
from grounded_video_agent.capabilities.media_inspection import NextAction
from grounded_video_agent.capabilities.subtitles.embedded_extraction import (
    EmbeddedSubtitleExtractionRequest,
)
from grounded_video_agent.capabilities.subtitles.speech_transcription import (
    SpeechTranscriptionRequest,
)
from grounded_video_agent.capabilities.temporal.chunking import TemporalChunkingRequest
from grounded_video_agent.capabilities.temporal.shot_detection import ShotDetectionRequest
from grounded_video_agent.domain import (
    CapabilityRequestContext,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ChunkManifest,
    EmbeddingManifest,
    IndexManifest,
    ShotManifest,
    TimeRange,
    TranscriptManifest,
    VideoAsset,
)
from grounded_video_agent.input import RegistrationStatus
from grounded_video_agent.pipelines.preprocessing.config import (
    DenseIndexPolicy,
    PreprocessingConfig,
    SubtitlePolicy,
)
from grounded_video_agent.pipelines.preprocessing.contracts import (
    PipelineCatalogEntries,
    PipelineError,
    PipelineReadiness,
    PipelineStage,
    PipelineStageReport,
    PipelineStageStatus,
    PipelineStatus,
    PreprocessingRequest,
    PreprocessingResult,
)
from grounded_video_agent.pipelines.preprocessing.dependencies import (
    PreprocessingDependencies,
)
from grounded_video_agent.pipelines.preprocessing.derivation import DerivationSpec
from grounded_video_agent.pipelines.preprocessing.keys import (
    AUDIO_KEY,
    CHUNKS_KEY,
    DENSE_INDEX_KEY,
    MEDIA_INSPECTION_KEY,
    SHOTS_KEY,
    SOURCE_KEY,
    SPARSE_INDEX_KEY,
    TRANSCRIPT_EMBEDDINGS_KEY,
    TRANSCRIPT_KEY,
)
from grounded_video_agent.pipelines.preprocessing.publication import CatalogPublisher
from grounded_video_agent.workspace.catalog import (
    ArtifactCatalog,
    AudioAssetDocument,
    CatalogEntry,
    CatalogKey,
    MediaInspectionDocument,
)

T = TypeVar("T")


class PreprocessingPipeline:
    """Fixed, framework-owned preprocessing flow for one registered source video."""

    VERSION = "1.0.0"

    def __init__(
        self,
        catalog: ArtifactCatalog,
        publisher: CatalogPublisher,
        dependencies: PreprocessingDependencies,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self._catalog = catalog
        self._publisher = publisher
        self._dependencies = dependencies
        self._config = config or PreprocessingConfig()

    def run(self, request: PreprocessingRequest | str) -> PreprocessingResult:
        normalized = (
            request
            if isinstance(request, PreprocessingRequest)
            else PreprocessingRequest(request)
        )
        run_id = f"preprocess_{uuid4().hex}"
        started_at = datetime.now(UTC)
        return self._run(normalized, run_id, started_at)

    def _run(
        self,
        request: PreprocessingRequest,
        run_id: str,
        started_at: datetime,
    ) -> PreprocessingResult:
        reports: list[PipelineStageReport] = []
        warnings: list[str] = []

        registration = self._dependencies.registrar.register(request.filename)
        if registration.status is RegistrationStatus.FAILED:
            assert registration.error is not None
            error = PipelineError(
                registration.error.code.value,
                registration.error.message,
                PipelineStage.REGISTRATION,
                registration.error.code.value == "file_changed_during_registration",
            )
            reports.append(self._failure_report(error))
            return self._result(
                run_id,
                started_at,
                PipelineStatus.FAILED,
                None,
                None,
                reports,
                warnings,
                PipelineReadiness(),
                PipelineCatalogEntries(),
                error,
            )

        assert registration.video_asset is not None
        try:
            snapshot = self._catalog.create_video(registration.video_asset)
            video_asset = snapshot.video_asset
            source_entry = self._catalog.resolve(video_asset.video_id, SOURCE_KEY).entry
        except Exception as exception:
            error = PipelineError(
                "CATALOG_REGISTRATION_FAILED",
                str(exception),
                PipelineStage.REGISTRATION,
            )
            reports.append(self._failure_report(error))
            return self._result(
                run_id,
                started_at,
                PipelineStatus.FAILED,
                registration.video_asset.video_id,
                None,
                reports,
                warnings,
                PipelineReadiness(),
                PipelineCatalogEntries(),
                error,
            )
        reports.append(
            PipelineStageReport(
                PipelineStage.REGISTRATION,
                PipelineStageStatus.SUCCEEDED,
                (source_entry.entry_id,),
            )
        )

        inspection_spec = DerivationSpec(
            "media-inspection",
            self._version(self._dependencies.inspector),
            self._cache_identity(self._dependencies.inspector),
        )
        cached_inspection = self._cached_document(
            video_asset.video_id,
            MEDIA_INSPECTION_KEY,
            inspection_spec,
            (source_entry.entry_id,),
            MediaInspectionDocument,
            request.force_refresh,
        )
        if cached_inspection is not None:
            inspection, inspection_entry = cached_inspection
            reports.append(
                self._cache_report(PipelineStage.MEDIA_INSPECTION, inspection_entry)
            )
        else:
            inspection_result = self._dependencies.inspector.inspect_registered(registration)
            if inspection_result.video_context is None:
                assert inspection_result.error is not None
                error = PipelineError(
                    inspection_result.error.code.value,
                    inspection_result.error.message,
                    PipelineStage.MEDIA_INSPECTION,
                    inspection_result.error.retryable,
                )
                reports.append(self._failure_report(error))
                return self._result(
                    run_id,
                    started_at,
                    PipelineStatus.FAILED,
                    video_asset.video_id,
                    self._catalog.get_snapshot(video_asset.video_id).revision,
                    reports,
                    warnings,
                    PipelineReadiness(),
                    PipelineCatalogEntries(),
                    error,
                )
            inspection, inspection_entry = self._publisher.publish_inspection(
                inspection_result,
                video_asset,
                MEDIA_INSPECTION_KEY,
                (source_entry.entry_id,),
                inspection_spec,
            )
            reports.append(
                PipelineStageReport(
                    PipelineStage.MEDIA_INSPECTION,
                    PipelineStageStatus.SUCCEEDED,
                    (inspection_entry.entry_id,),
                    usage=CapabilityUsage(
                        wall_time_ms=inspection_result.execution.duration_ms,
                        input_items=1,
                        output_items=1,
                    ),
                )
            )

        if inspection.next_action.value in {
            NextAction.REJECT.value,
            NextAction.NORMALIZE_AND_REINSPECT.value,
        }:
            error = PipelineError(
                "MEDIA_NOT_PROCESSABLE",
                f"Media inspection requires action: {inspection.next_action.value}.",
                PipelineStage.MEDIA_INSPECTION,
            )
            reports[-1] = PipelineStageReport(
                PipelineStage.MEDIA_INSPECTION,
                PipelineStageStatus.FAILED,
                (inspection_entry.entry_id,),
                error=error,
                usage=reports[-1].usage,
            )
            return self._result(
                run_id,
                started_at,
                PipelineStatus.FAILED,
                video_asset.video_id,
                self._catalog.get_snapshot(video_asset.video_id).revision,
                reports,
                warnings,
                PipelineReadiness(),
                PipelineCatalogEntries(
                    media_inspection_entry_id=inspection_entry.entry_id,
                ),
                error,
            )
        if inspection.next_action.value == NextAction.PROCEED_WITH_LIMITATIONS.value:
            warnings.append("Media inspection reported processing limitations.")

        source_range = self._source_range(inspection)
        if source_range is None:
            error = PipelineError(
                "MEDIA_DURATION_UNKNOWN",
                "A positive source duration could not be determined.",
                PipelineStage.MEDIA_INSPECTION,
            )
            reports[-1] = PipelineStageReport(
                PipelineStage.MEDIA_INSPECTION,
                PipelineStageStatus.FAILED,
                (inspection_entry.entry_id,),
                error=error,
                usage=reports[-1].usage,
            )
            return self._result(
                run_id,
                started_at,
                PipelineStatus.FAILED,
                video_asset.video_id,
                self._catalog.get_snapshot(video_asset.video_id).revision,
                reports,
                warnings,
                PipelineReadiness(media_ready=True),
                PipelineCatalogEntries(
                    media_inspection_entry_id=inspection_entry.entry_id,
                ),
                error,
            )

        shots, shots_entry = self._shots(
            video_asset,
            source_range,
            source_entry,
            request,
            run_id,
            reports,
            warnings,
        )
        transcript, transcript_entry, audio_entry = self._transcript(
            video_asset,
            inspection,
            source_range,
            source_entry,
            request,
            run_id,
            reports,
            warnings,
        )

        chunks, chunks_entry = self._chunks(
            video_asset.video_id,
            source_range,
            shots,
            shots_entry,
            transcript,
            transcript_entry,
            request,
            run_id,
            reports,
            warnings,
        )
        if chunks is None or chunks_entry is None:
            error = PipelineError(
                "TIMELINE_UNAVAILABLE",
                "No logical video timeline could be produced.",
                PipelineStage.CHUNKING,
            )
            entries = PipelineCatalogEntries(
                inspection_entry.entry_id,
                shots_entry.entry_id if shots_entry else None,
                transcript_entry.entry_id if transcript_entry else None,
                audio_entry.entry_id if audio_entry else None,
            )
            return self._result(
                run_id,
                started_at,
                PipelineStatus.FAILED,
                video_asset.video_id,
                self._catalog.get_snapshot(video_asset.video_id).revision,
                reports,
                warnings,
                PipelineReadiness(
                    media_ready=True,
                    shots_ready=shots is not None,
                    transcript_ready=transcript is not None,
                    limitations=("No chunk timeline is available.",),
                ),
                entries,
                error,
            )

        sparse_index, sparse_entry = self._sparse_index(
            transcript,
            transcript_entry,
            chunks,
            chunks_entry,
            request,
            run_id,
            reports,
            warnings,
        )
        dense_index, embedding_entry, dense_entry = self._dense_index(
            transcript,
            transcript_entry,
            chunks,
            chunks_entry,
            request,
            run_id,
            reports,
            warnings,
        )

        audit = self._catalog.audit(video_asset.video_id, deep=True)
        audit_error: PipelineError | None = None
        if audit.is_valid:
            reports.append(
                PipelineStageReport(
                    PipelineStage.CATALOG_AUDIT,
                    PipelineStageStatus.SUCCEEDED,
                )
            )
        else:
            audit_error = PipelineError(
                "CATALOG_AUDIT_FAILED",
                "Catalog contains missing or corrupt active resources.",
                PipelineStage.CATALOG_AUDIT,
            )
            reports.append(self._failure_report(audit_error))

        limitations: list[str] = []
        if shots is None:
            limitations.append("Shot detection is unavailable; chunks are not shot-aligned.")
        if transcript is None:
            limitations.append("No transcript is available; subtitle search is disabled.")
        if sparse_index is None:
            limitations.append("Sparse transcript search is unavailable.")
        dense_expected = self._config.dense_index_policy is not DenseIndexPolicy.DISABLED
        if dense_expected and dense_index is None:
            limitations.append("Dense transcript search is unavailable.")
        if inspection.next_action.value == NextAction.PROCEED_WITH_LIMITATIONS.value:
            limitations.append("Media inspection reported processing limitations.")
        readiness = PipelineReadiness(
            media_ready=True,
            shots_ready=shots is not None,
            transcript_ready=transcript is not None,
            timeline_ready=True,
            sparse_search_ready=sparse_index is not None,
            dense_search_ready=dense_index is not None,
            limitations=tuple(dict.fromkeys(limitations)),
        )
        entries = PipelineCatalogEntries(
            inspection_entry.entry_id,
            shots_entry.entry_id if shots_entry else None,
            transcript_entry.entry_id if transcript_entry else None,
            audio_entry.entry_id if audio_entry else None,
            chunks_entry.entry_id,
            sparse_entry.entry_id if sparse_entry else None,
            embedding_entry.entry_id if embedding_entry else None,
            dense_entry.entry_id if dense_entry else None,
        )
        required_dense_missing = (
            self._config.dense_index_policy is DenseIndexPolicy.REQUIRED
            and dense_index is None
        )
        complete = (
            shots is not None
            and transcript is not None
            and sparse_index is not None
            and (not dense_expected or dense_index is not None)
        )
        final_error = audit_error
        if required_dense_missing:
            final_error = PipelineError(
                "REQUIRED_DENSE_INDEX_UNAVAILABLE",
                "The configured required dense transcript index is unavailable.",
                PipelineStage.DENSE_INDEXING,
            )
        if final_error is not None:
            status = PipelineStatus.FAILED
        else:
            status = PipelineStatus.READY if complete else PipelineStatus.PARTIAL
        return self._result(
            run_id,
            started_at,
            status,
            video_asset.video_id,
            self._catalog.get_snapshot(video_asset.video_id).revision,
            reports,
            warnings,
            readiness,
            entries,
            final_error,
        )

    def _shots(
        self,
        video_asset: VideoAsset,
        source_range: TimeRange,
        source_entry: CatalogEntry,
        request: PreprocessingRequest,
        run_id: str,
        reports: list[PipelineStageReport],
        warnings: list[str],
    ) -> tuple[ShotManifest | None, CatalogEntry | None]:
        spec = DerivationSpec(
            "shot-detection",
            self._version(self._dependencies.shot_detector),
            {
                "source_range": source_range,
                "threshold": self._config.shot_threshold,
                "min_shot_duration_ms": self._config.min_shot_duration_ms,
            },
        )
        cached = self._cached_manifest(
            video_asset.video_id,
            SHOTS_KEY,
            spec,
            (source_entry.entry_id,),
            ShotManifest,
            request.force_refresh,
        )
        if cached is not None:
            shots, entry = cached
            reports.append(self._cache_report(PipelineStage.SHOT_DETECTION, entry))
            return shots, entry
        operation_id = f"{run_id}_shots"
        result = self._dependencies.shot_detector.execute(
            ShotDetectionRequest(
                video_asset,
                source_range,
                self._context(operation_id, request),
                self._config.shot_threshold,
                self._config.min_shot_duration_ms,
            )
        )
        if result.data is None:
            reports.append(self._capability_report(PipelineStage.SHOT_DETECTION, result))
            warnings.append("Shot detection failed; transcript-only chunking will be attempted.")
            return None, None
        entry = self._publisher.register_manifest(
            video_asset.video_id,
            SHOTS_KEY,
            result.data.ref,
            operation_id,
            (source_entry.entry_id,),
            spec,
        )
        reports.append(
            self._capability_report(PipelineStage.SHOT_DETECTION, result, (entry.entry_id,))
        )
        warnings.extend(result.warnings)
        return result.data, entry

    def _transcript(
        self,
        video_asset: VideoAsset,
        inspection: MediaInspectionDocument,
        source_range: TimeRange,
        source_entry: CatalogEntry,
        request: PreprocessingRequest,
        run_id: str,
        reports: list[PipelineStageReport],
        warnings: list[str],
    ) -> tuple[TranscriptManifest | None, CatalogEntry | None, CatalogEntry | None]:
        transcript: TranscriptManifest | None = None
        transcript_entry: CatalogEntry | None = None
        audio_entry: CatalogEntry | None = None
        subtitle_index = inspection.primary_streams.subtitle_stream_index
        try_embedded = (
            self._config.subtitle_policy is not SubtitlePolicy.ASR_ONLY
            and subtitle_index is not None
        )
        if try_embedded:
            assert subtitle_index is not None
            language = next(
                (
                    stream.language
                    for stream in inspection.media_probe.subtitle_streams
                    if stream.stream_index == subtitle_index
                ),
                self._config.language_hint,
            )
            spec = DerivationSpec(
                "embedded-subtitle-extraction",
                self._version(self._dependencies.embedded_subtitle_extractor),
                {"stream_index": subtitle_index, "language": language},
            )
            cached = self._cached_manifest(
                video_asset.video_id,
                TRANSCRIPT_KEY,
                spec,
                (source_entry.entry_id,),
                TranscriptManifest,
                request.force_refresh,
            )
            if cached is not None and cached[0].segments:
                transcript, transcript_entry = cached
                reports.append(
                    self._cache_report(PipelineStage.EMBEDDED_SUBTITLES, transcript_entry)
                )
            else:
                operation_id = f"{run_id}_embedded_subtitles"
                result = self._dependencies.embedded_subtitle_extractor.execute(
                    EmbeddedSubtitleExtractionRequest(
                        video_asset,
                        subtitle_index,
                        self._context(operation_id, request),
                        language,
                    )
                )
                if result.data is not None and result.data.segments:
                    transcript = result.data
                    transcript_entry = self._publisher.register_manifest(
                        video_asset.video_id,
                        TRANSCRIPT_KEY,
                        transcript.ref,
                        operation_id,
                        (source_entry.entry_id,),
                        spec,
                    )
                    reports.append(
                        self._capability_report(
                            PipelineStage.EMBEDDED_SUBTITLES,
                            result,
                            (transcript_entry.entry_id,),
                        )
                    )
                else:
                    reports.append(
                        self._capability_report(PipelineStage.EMBEDDED_SUBTITLES, result)
                    )
                    warnings.append("Embedded subtitles were unavailable or empty.")
        else:
            reports.append(self._skip_report(PipelineStage.EMBEDDED_SUBTITLES))

        try_asr = (
            transcript is None
            and self._config.subtitle_policy is not SubtitlePolicy.EMBEDDED_ONLY
        )
        audio_index = inspection.primary_streams.audio_stream_index
        if not try_asr or audio_index is None:
            reason = (
                "ASR fallback was not needed."
                if not try_asr
                else "No audio stream is available for ASR."
            )
            reports.append(self._skip_report(PipelineStage.AUDIO_EXTRACTION, reason))
            reports.append(self._skip_report(PipelineStage.SPEECH_TRANSCRIPTION, reason))
            if try_asr and audio_index is None:
                warnings.append(reason)
            return transcript, transcript_entry, audio_entry

        audio_spec = DerivationSpec(
            "audio-extraction",
            self._version(self._dependencies.audio_extractor),
            {
                "source_range": source_range,
                "stream_index": audio_index,
                "sample_rate_hz": self._config.audio_sample_rate_hz,
                "channels": self._config.audio_channels,
            },
        )
        cached_audio = self._cached_document(
            video_asset.video_id,
            AUDIO_KEY,
            audio_spec,
            (source_entry.entry_id,),
            AudioAssetDocument,
            request.force_refresh,
        )
        if cached_audio is not None:
            audio_document, audio_entry = cached_audio
            audio = audio_document.audio_asset
            reports.append(self._cache_report(PipelineStage.AUDIO_EXTRACTION, audio_entry))
        else:
            audio_operation = f"{run_id}_audio"
            audio_result = self._dependencies.audio_extractor.execute(
                AudioExtractionRequest(
                    video_asset,
                    source_range,
                    audio_index,
                    self._context(audio_operation, request),
                    self._config.audio_sample_rate_hz,
                    self._config.audio_channels,
                )
            )
            if audio_result.data is None:
                reports.append(
                    self._capability_report(PipelineStage.AUDIO_EXTRACTION, audio_result)
                )
                reports.append(
                    self._skip_report(
                        PipelineStage.SPEECH_TRANSCRIPTION,
                        "Audio extraction failed.",
                    )
                )
                warnings.append("ASR fallback failed during audio extraction.")
                return transcript, transcript_entry, None
            audio = audio_result.data
            _, audio_entry = self._publisher.publish_audio(
                audio,
                AUDIO_KEY,
                audio_operation,
                (source_entry.entry_id,),
                audio_spec,
            )
            reports.append(
                self._capability_report(
                    PipelineStage.AUDIO_EXTRACTION,
                    audio_result,
                    (audio_entry.entry_id,),
                )
            )

        asr_spec = DerivationSpec(
            "speech-transcription",
            self._version(self._dependencies.speech_transcriber),
            {
                "language_hint": self._config.language_hint,
                "word_timestamps": True,
                "backend": self._cache_identity(self._dependencies.speech_transcriber),
            },
        )
        assert audio_entry is not None
        cached_asr = self._cached_manifest(
            video_asset.video_id,
            TRANSCRIPT_KEY,
            asr_spec,
            (audio_entry.entry_id,),
            TranscriptManifest,
            request.force_refresh,
        )
        if cached_asr is not None and cached_asr[0].segments:
            transcript, transcript_entry = cached_asr
            reports.append(
                self._cache_report(PipelineStage.SPEECH_TRANSCRIPTION, transcript_entry)
            )
            return transcript, transcript_entry, audio_entry
        asr_operation = f"{run_id}_asr"
        asr_result = self._dependencies.speech_transcriber.execute(
            SpeechTranscriptionRequest(
                audio,
                self._context(asr_operation, request),
                self._config.language_hint,
                True,
            )
        )
        if asr_result.data is not None and asr_result.data.segments:
            transcript = asr_result.data
            transcript_entry = self._publisher.register_manifest(
                video_asset.video_id,
                TRANSCRIPT_KEY,
                transcript.ref,
                asr_operation,
                (audio_entry.entry_id,),
                asr_spec,
            )
            reports.append(
                self._capability_report(
                    PipelineStage.SPEECH_TRANSCRIPTION,
                    asr_result,
                    (transcript_entry.entry_id,),
                )
            )
        else:
            reports.append(
                self._capability_report(PipelineStage.SPEECH_TRANSCRIPTION, asr_result)
            )
            warnings.append("ASR did not produce a usable transcript.")
        return transcript, transcript_entry, audio_entry

    def _chunks(
        self,
        video_id: str,
        source_range: TimeRange,
        shots: ShotManifest | None,
        shots_entry: CatalogEntry | None,
        transcript: TranscriptManifest | None,
        transcript_entry: CatalogEntry | None,
        request: PreprocessingRequest,
        run_id: str,
        reports: list[PipelineStageReport],
        warnings: list[str],
    ) -> tuple[ChunkManifest | None, CatalogEntry | None]:
        dependencies = tuple(
            entry.entry_id
            for entry in (shots_entry, transcript_entry)
            if entry is not None
        )
        if not dependencies:
            error = PipelineError(
                "NO_CHUNK_SOURCE",
                "Neither shots nor transcript are available for chunking.",
                PipelineStage.CHUNKING,
            )
            reports.append(self._failure_report(error))
            return None, None
        config = self._config.chunking
        spec = DerivationSpec(
            "temporal-chunking",
            self._version(self._dependencies.chunker),
            config,
        )
        cached = self._cached_manifest(
            video_id,
            CHUNKS_KEY,
            spec,
            dependencies,
            ChunkManifest,
            request.force_refresh,
        )
        if cached is not None:
            chunks, entry = cached
            reports.append(self._cache_report(PipelineStage.CHUNKING, entry))
            return chunks, entry
        operation_id = f"{run_id}_chunks"
        result = self._dependencies.chunker.execute(
            TemporalChunkingRequest(
                video_id=video_id,
                source_range=source_range,
                shots=shots,
                transcript=transcript,
                context=self._context(operation_id, request),
                target_duration_ms=config.target_duration_ms,
                max_duration_ms=config.max_duration_ms,
                overlap_ms=0,
                target_characters=config.target_characters,
                max_characters=config.max_characters,
                max_silence_gap_ms=config.max_silence_gap_ms,
                context_padding_ms=config.context_padding_ms,
                max_inspection_duration_ms=config.max_inspection_duration_ms,
                align_to_shots=config.align_to_shots,
            )
        )
        if result.data is None:
            reports.append(self._capability_report(PipelineStage.CHUNKING, result))
            return None, None
        entry = self._publisher.register_manifest(
            video_id,
            CHUNKS_KEY,
            result.data.ref,
            operation_id,
            dependencies,
            spec,
        )
        reports.append(
            self._capability_report(PipelineStage.CHUNKING, result, (entry.entry_id,))
        )
        warnings.extend(result.warnings)
        return result.data, entry

    def _sparse_index(
        self,
        transcript: TranscriptManifest | None,
        transcript_entry: CatalogEntry | None,
        chunks: ChunkManifest,
        chunks_entry: CatalogEntry,
        request: PreprocessingRequest,
        run_id: str,
        reports: list[PipelineStageReport],
        warnings: list[str],
    ) -> tuple[IndexManifest | None, CatalogEntry | None]:
        if transcript is None or transcript_entry is None or not any(
            chunk.text for chunk in chunks.chunks
        ):
            reports.append(
                self._skip_report(
                    PipelineStage.SPARSE_INDEXING,
                    "No transcript chunks are available for indexing.",
                )
            )
            return None, None
        dependencies = (transcript_entry.entry_id, chunks_entry.entry_id)
        spec = DerivationSpec(
            "transcript-indexing",
            self._version(self._dependencies.transcript_indexer),
            {"index_kind": "bm25", "document_unit": "chunk"},
        )
        cached = self._cached_manifest(
            transcript.video_id,
            SPARSE_INDEX_KEY,
            spec,
            dependencies,
            IndexManifest,
            request.force_refresh,
        )
        if cached is not None:
            index, entry = cached
            reports.append(self._cache_report(PipelineStage.SPARSE_INDEXING, entry))
            return index, entry
        operation_id = f"{run_id}_sparse_index"
        result = self._dependencies.transcript_indexer.execute(
            TranscriptIndexingRequest(
                transcript,
                self._context(operation_id, request),
                chunks,
            )
        )
        if result.data is None:
            reports.append(self._capability_report(PipelineStage.SPARSE_INDEXING, result))
            warnings.append("Sparse transcript indexing failed.")
            return None, None
        entry = self._publisher.register_manifest(
            transcript.video_id,
            SPARSE_INDEX_KEY,
            result.data.ref,
            operation_id,
            dependencies,
            spec,
        )
        reports.append(
            self._capability_report(PipelineStage.SPARSE_INDEXING, result, (entry.entry_id,))
        )
        warnings.extend(result.warnings)
        return result.data, entry

    def _dense_index(
        self,
        transcript: TranscriptManifest | None,
        transcript_entry: CatalogEntry | None,
        chunks: ChunkManifest,
        chunks_entry: CatalogEntry,
        request: PreprocessingRequest,
        run_id: str,
        reports: list[PipelineStageReport],
        warnings: list[str],
    ) -> tuple[IndexManifest | None, CatalogEntry | None, CatalogEntry | None]:
        if self._config.dense_index_policy is DenseIndexPolicy.DISABLED:
            reports.append(self._skip_report(PipelineStage.DENSE_INDEXING))
            return None, None, None
        indexer = self._dependencies.dense_indexer
        if transcript is None or transcript_entry is None or indexer is None:
            message = "A transcript or dense embedding backend is unavailable."
            if self._config.dense_index_policy is DenseIndexPolicy.REQUIRED:
                reports.append(
                    self._failure_report(
                        PipelineError(
                            "REQUIRED_DENSE_INPUT_UNAVAILABLE",
                            message,
                            PipelineStage.DENSE_INDEXING,
                        )
                    )
                )
            else:
                reports.append(self._skip_report(PipelineStage.DENSE_INDEXING, message))
            warnings.append("Dense transcript indexing was requested but could not run.")
            return None, None, None
        dependencies = (transcript_entry.entry_id, chunks_entry.entry_id)
        try:
            model_info = indexer.get_model_info()
        except Exception as exception:
            error = PipelineError(
                "DENSE_MODEL_UNAVAILABLE",
                str(exception),
                PipelineStage.DENSE_INDEXING,
            )
            reports.append(self._failure_report(error))
            warnings.append("Dense embedding model information could not be loaded.")
            return None, None, None
        spec = DerivationSpec(
            "dense-indexing",
            self._version(indexer),
            {"model": model_info, "document_unit": "chunk"},
        )
        if not request.force_refresh:
            cached_embedding = self._catalog.find_reusable(
                transcript.video_id,
                TRANSCRIPT_EMBEDDINGS_KEY,
                spec.key,
                dependencies,
            )
            if cached_embedding is not None:
                self._catalog.activate(
                    transcript.video_id,
                    TRANSCRIPT_EMBEDDINGS_KEY,
                    cached_embedding.entry.entry_id,
                )
                embedding = self._catalog.load_manifest(
                    transcript.video_id,
                    TRANSCRIPT_EMBEDDINGS_KEY,
                    EmbeddingManifest,
                )
                dense_dependencies = (*dependencies, cached_embedding.entry.entry_id)
                cached_index = self._cached_manifest(
                    transcript.video_id,
                    DENSE_INDEX_KEY,
                    spec,
                    dense_dependencies,
                    IndexManifest,
                    False,
                )
                if cached_index is not None:
                    index, index_entry = cached_index
                    reports.append(
                        PipelineStageReport(
                            PipelineStage.DENSE_INDEXING,
                            PipelineStageStatus.CACHE_HIT,
                            (cached_embedding.entry.entry_id, index_entry.entry_id),
                        )
                    )
                    return index, cached_embedding.entry, index_entry
                del embedding
        operation_id = f"{run_id}_dense_index"
        result = indexer.execute(
            DenseIndexingRequest(
                transcript,
                self._context(operation_id, request),
                chunks,
            )
        )
        if result.data is None or result.data.embedding_manifest is None:
            reports.append(self._capability_report(PipelineStage.DENSE_INDEXING, result))
            warnings.append("Dense transcript indexing failed.")
            return None, None, None
        embedding = result.data.embedding_manifest
        embedding_entry = self._publisher.register_manifest(
            transcript.video_id,
            TRANSCRIPT_EMBEDDINGS_KEY,
            embedding.ref,
            operation_id,
            dependencies,
            spec,
        )
        dense_dependencies = (*dependencies, embedding_entry.entry_id)
        index_entry = self._publisher.register_manifest(
            transcript.video_id,
            DENSE_INDEX_KEY,
            result.data.ref,
            operation_id,
            dense_dependencies,
            spec,
        )
        reports.append(
            self._capability_report(
                PipelineStage.DENSE_INDEXING,
                result,
                (embedding_entry.entry_id, index_entry.entry_id),
            )
        )
        warnings.extend(result.warnings)
        return result.data, embedding_entry, index_entry

    def _cached_manifest(
        self,
        video_id: str,
        key: CatalogKey,
        spec: DerivationSpec,
        dependencies: tuple[str, ...],
        expected_type: type[T],
        force_refresh: bool,
    ) -> tuple[T, CatalogEntry] | None:
        if force_refresh:
            return None
        resolved = self._catalog.find_reusable(video_id, key, spec.key, dependencies)
        if resolved is None:
            return None
        self._catalog.activate(video_id, key, resolved.entry.entry_id)
        return self._catalog.load_manifest(video_id, key, expected_type), resolved.entry

    def _cached_document(
        self,
        video_id: str,
        key: CatalogKey,
        spec: DerivationSpec,
        dependencies: tuple[str, ...],
        expected_type: type[T],
        force_refresh: bool,
    ) -> tuple[T, CatalogEntry] | None:
        if force_refresh:
            return None
        resolved = self._catalog.find_reusable(video_id, key, spec.key, dependencies)
        if resolved is None:
            return None
        self._catalog.activate(video_id, key, resolved.entry.entry_id)
        return self._catalog.load_document(video_id, key, expected_type), resolved.entry

    def _context(
        self,
        operation_id: str,
        request: PreprocessingRequest,
    ) -> CapabilityRequestContext:
        return CapabilityRequestContext(
            operation_id,
            self._config.limits,
            request.trace_id,
            request.force_refresh,
        )

    @staticmethod
    def _source_range(inspection: MediaInspectionDocument) -> TimeRange | None:
        probe = inspection.media_probe
        candidates = (
            probe.container.duration_ms,
            probe.primary_video_stream.duration_ms if probe.primary_video_stream else None,
            probe.primary_audio_stream.duration_ms if probe.primary_audio_stream else None,
        )
        duration_ms = next((value for value in candidates if value is not None and value > 0), None)
        return TimeRange(0, duration_ms) if duration_ms is not None else None

    @staticmethod
    def _version(dependency: object) -> str:
        value = getattr(dependency, "VERSION", "1")
        return str(value)

    @staticmethod
    def _cache_identity(dependency: object) -> object:
        identity = getattr(dependency, "cache_identity", None)
        if callable(identity):
            return identity()
        dependency_type = type(dependency)
        return {
            "implementation": (
                f"{dependency_type.__module__}.{dependency_type.__qualname__}"
            )
        }

    @staticmethod
    def _cache_report(stage: PipelineStage, entry: CatalogEntry) -> PipelineStageReport:
        return PipelineStageReport(stage, PipelineStageStatus.CACHE_HIT, (entry.entry_id,))

    @staticmethod
    def _skip_report(stage: PipelineStage, warning: str | None = None) -> PipelineStageReport:
        return PipelineStageReport(
            stage,
            PipelineStageStatus.SKIPPED,
            warnings=(warning,) if warning is not None else (),
        )

    @staticmethod
    def _failure_report(error: PipelineError) -> PipelineStageReport:
        return PipelineStageReport(error.stage, PipelineStageStatus.FAILED, error=error)

    @staticmethod
    def _capability_report(
        stage: PipelineStage,
        result: CapabilityResult[T],
        entry_ids: tuple[str, ...] = (),
    ) -> PipelineStageReport:
        if result.status is CapabilityStatus.FAILED:
            assert result.error is not None
            error = PipelineError(
                result.error.code,
                result.error.message,
                stage,
                result.error.retryable,
            )
            return PipelineStageReport(
                stage,
                PipelineStageStatus.FAILED,
                warnings=result.warnings,
                error=error,
                usage=result.usage,
            )
        status = (
            PipelineStageStatus.PARTIAL
            if result.status is CapabilityStatus.PARTIAL
            else PipelineStageStatus.SUCCEEDED
        )
        return PipelineStageReport(
            stage,
            status,
            entry_ids,
            result.warnings,
            usage=result.usage,
        )

    @staticmethod
    def _result(
        run_id: str,
        started_at: datetime,
        status: PipelineStatus,
        video_id: str | None,
        revision: int | None,
        reports: list[PipelineStageReport],
        warnings: list[str],
        readiness: PipelineReadiness,
        entries: PipelineCatalogEntries,
        error: PipelineError | None,
    ) -> PreprocessingResult:
        return PreprocessingResult(
            run_id=run_id,
            status=status,
            video_id=video_id,
            catalog_revision=revision,
            readiness=readiness,
            stages=tuple(reports),
            entries=entries,
            warnings=tuple(dict.fromkeys(warnings)),
            error=error,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

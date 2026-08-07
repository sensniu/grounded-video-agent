"""Public facade combining FFprobe facts and deterministic validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from grounded_video_agent.capabilities.media_inspection.contracts import (
    BasicVideoFlags,
    InspectionDiagnostics,
    InspectionError,
    InspectionErrorCode,
    InspectionExecution,
    InspectionExecutionStatus,
    InspectionStage,
    NextAction,
    PrimaryStreams,
    VideoInspectionContext,
    VideoInspectionResult,
)
from grounded_video_agent.capabilities.media_inspection.ffprobe import (
    FFprobeError,
    FFprobeErrorCode,
    FFprobeRunner,
    ProbeRunner,
)
from grounded_video_agent.capabilities.media_inspection.mapper import (
    ProbeMappingError,
    map_ffprobe_payload,
)
from grounded_video_agent.capabilities.media_inspection.validator import MediaValidator
from grounded_video_agent.domain import ValidationStatus, VideoAsset
from grounded_video_agent.input import (
    RegistrationErrorCode,
    RegistrationStatus,
    VideoRegistrar,
    VideoRegistrationResult,
)


class MediaInspectionCapability:
    """Inspect basic media information without deriving any new media."""

    VERSION = "1.0.0"
    SCHEMA_VERSION = "1"

    def __init__(
        self,
        *,
        input_root: str | Path = "analyzed_video",
        registrar: VideoRegistrar | None = None,
        probe_runner: ProbeRunner | None = None,
        validator: MediaValidator | None = None,
        retain_raw_probe: bool = True,
    ) -> None:
        self._registrar = registrar or VideoRegistrar(input_root)
        self._probe_runner = probe_runner or FFprobeRunner()
        self._validator = validator or MediaValidator()
        self._retain_raw_probe = retain_raw_probe

    def cache_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "validator": self._validator.cache_identity(),
        }

    def inspect(self, filename: str) -> VideoInspectionResult:
        inspection_id = f"inspection_{uuid4().hex}"
        started_at = datetime.now(UTC)
        started = perf_counter()
        registration = self._registrar.register(filename)
        return self._inspect_registered(registration, inspection_id, started_at, started)

    def inspect_registered(
        self,
        registration: VideoRegistrationResult,
    ) -> VideoInspectionResult:
        """Inspect a previously registered video without hashing the source again."""

        return self._inspect_registered(
            registration,
            f"inspection_{uuid4().hex}",
            datetime.now(UTC),
            perf_counter(),
        )

    def _inspect_registered(
        self,
        registration: VideoRegistrationResult,
        inspection_id: str,
        started_at: datetime,
        started: float,
    ) -> VideoInspectionResult:
        if registration.status is RegistrationStatus.FAILED:
            assert registration.error is not None
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                self._registration_error(registration.error.code, registration.error.message),
            )

        assert registration.video_asset is not None
        video_asset = registration.video_asset
        source_path = Path(video_asset.source.uri)

        input_error = self._check_source(video_asset, source_path)
        if input_error is not None:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                input_error,
            )

        try:
            before = source_path.stat()
        except OSError:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                InspectionError(
                    code=InspectionErrorCode.SOURCE_CHANGED,
                    message="Registered video source changed before FFprobe could inspect it.",
                    stage=InspectionStage.INPUT_CHECK,
                    retryable=True,
                ),
            )
        try:
            raw_probe = self._probe_runner.probe(source_path)
        except FFprobeError as error:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                self._inspection_error(error),
                diagnostics=InspectionDiagnostics(ffprobe_stderr=error.stderr),
            )

        try:
            after = source_path.stat()
        except OSError:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                InspectionError(
                    code=InspectionErrorCode.SOURCE_CHANGED,
                    message="Video file disappeared or became unreadable during inspection.",
                    stage=InspectionStage.PROBING,
                    retryable=True,
                ),
                diagnostics=InspectionDiagnostics(
                    ffprobe_stderr=raw_probe.stderr,
                    raw_probe=raw_probe.payload if self._retain_raw_probe else None,
                ),
            )
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                InspectionError(
                    code=InspectionErrorCode.SOURCE_CHANGED,
                    message="Video file changed while FFprobe was inspecting it.",
                    stage=InspectionStage.PROBING,
                    retryable=True,
                ),
                diagnostics=InspectionDiagnostics(
                    ffprobe_stderr=raw_probe.stderr,
                    raw_probe=raw_probe.payload if self._retain_raw_probe else None,
                ),
            )

        try:
            media_probe = map_ffprobe_payload(video_asset.video_id, raw_probe.payload)
        except ProbeMappingError as error:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                InspectionError(
                    code=InspectionErrorCode.MAPPING_FAILED,
                    message=str(error),
                    stage=InspectionStage.MAPPING,
                ),
                diagnostics=InspectionDiagnostics(
                    ffprobe_stderr=raw_probe.stderr,
                    raw_probe=raw_probe.payload if self._retain_raw_probe else None,
                ),
            )

        try:
            validation = self._validator.validate(media_probe)
        except Exception as error:
            return self._failed_result(
                inspection_id,
                registration,
                started_at,
                started,
                InspectionError(
                    code=InspectionErrorCode.INTERNAL_ERROR,
                    message=f"Media validation failed: {error}",
                    stage=InspectionStage.VALIDATING,
                ),
                diagnostics=InspectionDiagnostics(
                    ffprobe_stderr=raw_probe.stderr,
                    raw_probe=raw_probe.payload if self._retain_raw_probe else None,
                ),
            )

        primary_video = media_probe.primary_video_stream
        primary_audio = media_probe.primary_audio_stream
        primary_subtitle = media_probe.primary_subtitle_stream
        context = VideoInspectionContext(
            video_asset=video_asset,
            media_probe=media_probe,
            validation=validation,
            primary_streams=PrimaryStreams(
                video_stream_index=primary_video.stream_index if primary_video else None,
                audio_stream_index=primary_audio.stream_index if primary_audio else None,
                subtitle_stream_index=(primary_subtitle.stream_index if primary_subtitle else None),
            ),
            basic_flags=BasicVideoFlags(
                has_video=primary_video is not None,
                has_audio=bool(media_probe.audio_streams),
                has_embedded_subtitles=bool(media_probe.subtitle_streams),
                has_multiple_video_streams=len(media_probe.video_streams) > 1,
                has_multiple_audio_streams=len(media_probe.audio_streams) > 1,
                is_variable_frame_rate=any(
                    stream.is_variable_frame_rate for stream in media_probe.video_streams
                ),
                has_rotation_metadata=any(
                    stream.rotation_degrees % 360 != 0 for stream in media_probe.video_streams
                ),
            ),
        )
        finished_at = datetime.now(UTC)
        return VideoInspectionResult(
            schema_version=self.SCHEMA_VERSION,
            inspection_id=inspection_id,
            registration=registration,
            execution=InspectionExecution(
                status=InspectionExecutionStatus.SUCCEEDED,
                stage=InspectionStage.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=round((perf_counter() - started) * 1000),
            ),
            next_action=self._next_action(validation.status),
            video_context=context,
            error=None,
            diagnostics=InspectionDiagnostics(
                ffprobe_stderr=raw_probe.stderr,
                raw_probe=raw_probe.payload if self._retain_raw_probe else None,
            ),
        )

    @staticmethod
    def _check_source(video_asset: VideoAsset, source_path: Path) -> InspectionError | None:
        try:
            stat_result = source_path.stat()
        except FileNotFoundError:
            return InspectionError(
                code=InspectionErrorCode.SOURCE_NOT_FOUND,
                message="Registered video source no longer exists.",
                stage=InspectionStage.INPUT_CHECK,
            )
        except PermissionError:
            return InspectionError(
                code=InspectionErrorCode.SOURCE_NOT_READABLE,
                message="Registered video source cannot be read.",
                stage=InspectionStage.INPUT_CHECK,
            )
        if not source_path.is_file():
            return InspectionError(
                code=InspectionErrorCode.SOURCE_NOT_READABLE,
                message="Registered video source is not a regular file.",
                stage=InspectionStage.INPUT_CHECK,
            )
        expected_size = video_asset.source.size_bytes
        if expected_size is not None and stat_result.st_size != expected_size:
            return InspectionError(
                code=InspectionErrorCode.SOURCE_CHANGED,
                message="Registered video size no longer matches the source artifact.",
                stage=InspectionStage.INPUT_CHECK,
                retryable=True,
            )
        return None

    @staticmethod
    def _inspection_error(error: FFprobeError) -> InspectionError:
        code_map = {
            FFprobeErrorCode.EXECUTABLE_NOT_FOUND: InspectionErrorCode.FFPROBE_NOT_FOUND,
            FFprobeErrorCode.TIMEOUT: InspectionErrorCode.FFPROBE_TIMEOUT,
            FFprobeErrorCode.PROCESS_FAILED: InspectionErrorCode.FFPROBE_FAILED,
            FFprobeErrorCode.EMPTY_OUTPUT: InspectionErrorCode.EMPTY_OUTPUT,
            FFprobeErrorCode.INVALID_JSON: InspectionErrorCode.INVALID_JSON,
        }
        return InspectionError(
            code=code_map[error.code],
            message=str(error),
            stage=InspectionStage.PROBING,
            retryable=error.retryable,
        )

    @staticmethod
    def _registration_error(
        code: RegistrationErrorCode,
        message: str,
    ) -> InspectionError:
        code_map = {
            RegistrationErrorCode.INVALID_FILENAME: InspectionErrorCode.INVALID_FILENAME,
            RegistrationErrorCode.INPUT_ROOT_NOT_FOUND: InspectionErrorCode.INPUT_ROOT_NOT_FOUND,
            RegistrationErrorCode.FILE_NOT_FOUND: InspectionErrorCode.SOURCE_NOT_FOUND,
            RegistrationErrorCode.NOT_A_FILE: InspectionErrorCode.SOURCE_NOT_A_FILE,
            RegistrationErrorCode.SYMLINK_NOT_ALLOWED: InspectionErrorCode.SYMLINK_NOT_ALLOWED,
            RegistrationErrorCode.PATH_NOT_ALLOWED: InspectionErrorCode.PATH_NOT_ALLOWED,
            RegistrationErrorCode.PERMISSION_DENIED: InspectionErrorCode.SOURCE_NOT_READABLE,
            RegistrationErrorCode.FILE_CHANGED: InspectionErrorCode.SOURCE_CHANGED,
            RegistrationErrorCode.IO_ERROR: InspectionErrorCode.SOURCE_NOT_READABLE,
        }
        return InspectionError(
            code=code_map[code],
            message=message,
            stage=InspectionStage.REGISTRATION,
            retryable=code is RegistrationErrorCode.FILE_CHANGED,
        )

    @staticmethod
    def _next_action(status: ValidationStatus) -> NextAction:
        return {
            ValidationStatus.VALID: NextAction.PROCEED,
            ValidationStatus.VALID_WITH_WARNINGS: NextAction.PROCEED,
            ValidationStatus.REQUIRES_NORMALIZATION: NextAction.NORMALIZE_AND_REINSPECT,
            ValidationStatus.PARTIALLY_SUPPORTED: NextAction.PROCEED_WITH_LIMITATIONS,
            ValidationStatus.INVALID: NextAction.REJECT,
        }[status]

    def _failed_result(
        self,
        inspection_id: str,
        registration: VideoRegistrationResult,
        started_at: datetime,
        started: float,
        error: InspectionError,
        *,
        diagnostics: InspectionDiagnostics | None = None,
    ) -> VideoInspectionResult:
        finished_at = datetime.now(UTC)
        return VideoInspectionResult(
            schema_version=self.SCHEMA_VERSION,
            inspection_id=inspection_id,
            registration=registration,
            execution=InspectionExecution(
                status=InspectionExecutionStatus.FAILED,
                stage=error.stage,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=round((perf_counter() - started) * 1000),
            ),
            next_action=(NextAction.RETRY_INSPECTION if error.retryable else NextAction.REJECT),
            video_context=None,
            error=error,
            diagnostics=diagnostics or InspectionDiagnostics(),
        )

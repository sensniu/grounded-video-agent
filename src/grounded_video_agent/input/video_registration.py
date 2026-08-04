"""Register a filename from a managed input directory as a video asset."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from grounded_video_agent.domain import ArtifactKind, ArtifactRef, VideoAsset
from grounded_video_agent.input.contracts import (
    RegisteredFileInfo,
    RegistrationError,
    RegistrationErrorCode,
    RegistrationStatus,
    VideoRegistrationResult,
)


class VideoRegistrar:
    """Resolve and fingerprint a plain filename under a managed directory."""

    def __init__(self, input_root: str | Path, *, hash_chunk_size: int = 1024 * 1024) -> None:
        if hash_chunk_size <= 0:
            raise ValueError("hash_chunk_size must be positive")
        self._input_root = Path(input_root).expanduser().resolve()
        self._hash_chunk_size = hash_chunk_size

    @property
    def input_root(self) -> Path:
        return self._input_root

    def register(self, filename: str) -> VideoRegistrationResult:
        registration_id = f"registration_{uuid4().hex}"
        registered_at = datetime.now(UTC)

        filename_error = self._validate_filename(filename)
        if filename_error is not None:
            return self._failure(registration_id, registered_at, filename_error)

        if not self._input_root.is_dir():
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.INPUT_ROOT_NOT_FOUND,
                    f"Input directory does not exist: {self._input_root}",
                ),
            )

        candidate = self._input_root / filename
        if candidate.is_symlink():
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.SYMLINK_NOT_ALLOWED,
                    "Symbolic links are not accepted as video inputs.",
                ),
            )

        try:
            resolved_path = candidate.resolve(strict=True)
        except FileNotFoundError:
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.FILE_NOT_FOUND,
                    f"Video file does not exist: {filename}",
                ),
            )
        except PermissionError:
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.PERMISSION_DENIED,
                    f"Video file cannot be accessed: {filename}",
                ),
            )
        except OSError as error:
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(RegistrationErrorCode.IO_ERROR, str(error)),
            )

        if not resolved_path.is_relative_to(self._input_root):
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.PATH_NOT_ALLOWED,
                    "Resolved video path is outside the managed input directory.",
                ),
            )
        if not resolved_path.is_file():
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.NOT_A_FILE,
                    f"Video input is not a regular file: {filename}",
                ),
            )

        try:
            digest, stat_result = self._fingerprint(resolved_path)
        except PermissionError:
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(
                    RegistrationErrorCode.PERMISSION_DENIED,
                    f"Video file cannot be read: {filename}",
                ),
            )
        except RuntimeError as error:
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(RegistrationErrorCode.FILE_CHANGED, str(error)),
            )
        except OSError as error:
            return self._failure(
                registration_id,
                registered_at,
                RegistrationError(RegistrationErrorCode.IO_ERROR, str(error)),
            )

        relative_uri = (Path(self._input_root.name) / filename).as_posix()
        artifact = ArtifactRef(
            artifact_id=f"source_video_{digest}",
            kind=ArtifactKind.SOURCE_VIDEO,
            uri=str(resolved_path),
            sha256=digest,
            size_bytes=stat_result.st_size,
        )
        video_asset = VideoAsset(
            video_id=f"video_{digest}",
            source=artifact,
            display_name=filename,
            registered_at=registered_at,
        )
        file_info = RegisteredFileInfo(
            filename=filename,
            relative_uri=relative_uri,
            resolved_path=str(resolved_path),
            size_bytes=stat_result.st_size,
            modified_at_ns=stat_result.st_mtime_ns,
        )
        return VideoRegistrationResult(
            registration_id=registration_id,
            status=RegistrationStatus.SUCCEEDED,
            registered_at=registered_at,
            video_asset=video_asset,
            file_info=file_info,
        )

    def _fingerprint(self, path: Path) -> tuple[str, os.stat_result]:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            before = os.fstat(file.fileno())
            while chunk := file.read(self._hash_chunk_size):
                digest.update(chunk)
            after = os.fstat(file.fileno())

        unchanged = (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        )
        if not unchanged:
            raise RuntimeError("Video file changed while its fingerprint was being calculated.")
        return digest.hexdigest(), after

    @staticmethod
    def _validate_filename(filename: str) -> RegistrationError | None:
        if not isinstance(filename, str) or not filename or not filename.strip():
            return RegistrationError(
                RegistrationErrorCode.INVALID_FILENAME,
                "Video filename must be a non-empty string.",
            )
        if "\x00" in filename:
            return RegistrationError(
                RegistrationErrorCode.INVALID_FILENAME,
                "Video filename must not contain null characters.",
            )
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            return RegistrationError(
                RegistrationErrorCode.INVALID_FILENAME,
                "Video input must be a plain filename without path components.",
            )
        if Path(filename).is_absolute() or Path(filename).name != filename:
            return RegistrationError(
                RegistrationErrorCode.INVALID_FILENAME,
                "Video input must be a plain filename without path components.",
            )
        return None

    @staticmethod
    def _failure(
        registration_id: str,
        registered_at: datetime,
        error: RegistrationError,
    ) -> VideoRegistrationResult:
        return VideoRegistrationResult(
            registration_id=registration_id,
            status=RegistrationStatus.FAILED,
            registered_at=registered_at,
            error=error,
        )

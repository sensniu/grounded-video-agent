from __future__ import annotations

import fcntl
import hashlib
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from grounded_video_agent.capabilities._support import json_value
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    IndexManifest,
    ManifestRef,
    VideoAsset,
)
from grounded_video_agent.workspace.catalog.codecs import (
    DocumentCodecRegistry,
    ManifestCodecRegistry,
)
from grounded_video_agent.workspace.catalog.contracts import (
    CatalogAuditIssue,
    CatalogAuditIssueCode,
    CatalogAuditReport,
    CatalogDocumentRef,
    CatalogEntry,
    CatalogEntryAudit,
    CatalogEntryState,
    CatalogError,
    CatalogErrorCode,
    CatalogKey,
    CatalogRegistration,
    CatalogResourceType,
    CatalogSelection,
    CatalogSnapshot,
    ResolvedCatalogEntry,
)

T = TypeVar("T")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FilesystemArtifactCatalog:
    SCHEMA_VERSION = "1"

    def __init__(
        self,
        catalog_root: str | Path = "artifacts/catalog",
        *,
        artifact_root: str | Path = "artifacts",
        input_roots: tuple[str | Path, ...] = ("analyzed_video",),
        codecs: ManifestCodecRegistry | None = None,
        document_codecs: DocumentCodecRegistry | None = None,
    ) -> None:
        if not input_roots:
            raise ValueError("input_roots must not be empty")
        self._catalog_root = Path(catalog_root).expanduser().resolve()
        self._artifact_root = Path(artifact_root).expanduser().resolve()
        self._input_roots = tuple(Path(root).expanduser().resolve() for root in input_roots)
        self._codecs = codecs or ManifestCodecRegistry()
        self._document_codecs = document_codecs or DocumentCodecRegistry()

    def create_video(self, video_asset: VideoAsset) -> CatalogSnapshot:
        self._validate_video_id(video_asset.video_id)
        source_path = self._validate_artifact_path(video_asset.source, require_exists=True)
        source_size = source_path.stat().st_size
        if (
            video_asset.source.size_bytes is not None
            and video_asset.source.size_bytes != source_size
        ):
            raise CatalogError(
                CatalogErrorCode.INVALID_REFERENCE,
                "source video size no longer matches its registration",
            )
        source_digest = (
            video_asset.source.sha256.lower()
            if video_asset.source.sha256 is not None
            else self._fingerprint(source_path)
        )
        catalog_path = self._catalog_path(video_asset.video_id)
        with self._locked(video_asset.video_id):
            if catalog_path.exists():
                snapshot = self._read(video_asset.video_id)
                if (
                    snapshot.video_asset.source.uri == video_asset.source.uri
                    and snapshot.video_asset.source.sha256 == video_asset.source.sha256
                ):
                    return snapshot
                raise CatalogError(
                    CatalogErrorCode.VIDEO_ALREADY_REGISTERED,
                    f"video catalog already exists: {video_asset.video_id}",
                )
            now = datetime.now(UTC)
            key = CatalogKey(
                CatalogResourceType.SOURCE,
                artifact_kind=ArtifactKind.SOURCE_VIDEO,
            )
            entry = CatalogEntry(
                entry_id=f"entry_{uuid4().hex}",
                video_id=video_asset.video_id,
                key=key,
                reference=video_asset.source,
                operation_id=f"catalog_create_{video_asset.video_id}",
                producer_name="video-registration",
                producer_version="1",
                dependency_entry_ids=(),
                content_sha256=source_digest,
                size_bytes=source_size,
                derivation_key=source_digest,
                created_at=now,
            )
            snapshot = CatalogSnapshot(
                schema_version=self.SCHEMA_VERSION,
                video_asset=video_asset,
                revision=1,
                entries=(entry,),
                active_selections=(CatalogSelection(key, entry.entry_id),),
                created_at=now,
                updated_at=now,
            )
            self._write(snapshot)
            return snapshot

    def get_snapshot(self, video_id: str) -> CatalogSnapshot:
        return self._read(video_id)

    def register(
        self,
        video_id: str,
        registration: CatalogRegistration,
        *,
        expected_revision: int | None = None,
    ) -> CatalogSnapshot:
        self._validate_video_id(video_id)
        with self._locked(video_id):
            snapshot = self._read(video_id)
            self._check_revision(snapshot, expected_revision)
            entries_by_id = {entry.entry_id: entry for entry in snapshot.entries}
            missing_dependencies = tuple(
                dependency
                for dependency in registration.dependency_entry_ids
                if dependency not in entries_by_id
            )
            if missing_dependencies:
                raise CatalogError(
                    CatalogErrorCode.DEPENDENCY_NOT_FOUND,
                    f"unknown catalog dependencies: {', '.join(missing_dependencies)}",
                )
            path = self._validate_registration_reference(
                video_id,
                registration.key,
                registration.reference,
            )
            digest = self._fingerprint(path)
            artifact = _reference_artifact(registration.reference)
            if artifact.sha256 is not None and artifact.sha256.lower() != digest:
                raise CatalogError(
                    CatalogErrorCode.HASH_MISMATCH,
                    f"registered artifact hash does not match file: {artifact.artifact_id}",
                )
            producer_name, producer_version, parameters_hash = self._producer(registration)
            duplicate = next(
                (
                    entry
                    for entry in snapshot.entries
                    if entry.key == registration.key
                    and entry.content_sha256 == digest
                    and entry.dependency_entry_ids == registration.dependency_entry_ids
                    and entry.derivation_key == registration.derivation_key
                    and entry.producer_name == producer_name
                    and entry.producer_version == producer_version
                ),
                None,
            )
            if duplicate is not None:
                if registration.activate and not self._is_selected(snapshot, duplicate):
                    return self._activate_snapshot(snapshot, duplicate)
                return snapshot
            entry = CatalogEntry(
                entry_id=f"entry_{uuid4().hex}",
                video_id=video_id,
                key=registration.key,
                reference=registration.reference,
                operation_id=registration.operation_id,
                producer_name=producer_name,
                producer_version=producer_version,
                dependency_entry_ids=registration.dependency_entry_ids,
                content_sha256=digest,
                size_bytes=path.stat().st_size,
                parameters_hash=parameters_hash,
                derivation_key=registration.derivation_key,
            )
            selections = list(snapshot.active_selections)
            if registration.activate:
                selections = self._with_selection(selections, entry.key, entry.entry_id)
            updated = CatalogSnapshot(
                snapshot.schema_version,
                snapshot.video_asset,
                snapshot.revision + 1,
                (*snapshot.entries, entry),
                tuple(selections),
                snapshot.created_at,
                datetime.now(UTC),
            )
            self._write(updated)
            return updated

    def activate(
        self,
        video_id: str,
        key: CatalogKey,
        entry_id: str,
        *,
        expected_revision: int | None = None,
    ) -> CatalogSnapshot:
        self._validate_video_id(video_id)
        with self._locked(video_id):
            snapshot = self._read(video_id)
            self._check_revision(snapshot, expected_revision)
            entry = next((item for item in snapshot.entries if item.entry_id == entry_id), None)
            if entry is None or entry.key != key:
                raise CatalogError(
                    CatalogErrorCode.ENTRY_NOT_FOUND,
                    "catalog entry does not exist or has a different key",
                )
            if self._is_selected(snapshot, entry):
                return snapshot
            return self._activate_snapshot(snapshot, entry)

    def resolve(
        self,
        video_id: str,
        key: CatalogKey,
        *,
        require_fresh: bool = True,
    ) -> ResolvedCatalogEntry:
        snapshot = self._read(video_id)
        selection = next((item for item in snapshot.active_selections if item.key == key), None)
        if selection is None:
            raise CatalogError(
                CatalogErrorCode.RESOURCE_NOT_REGISTERED,
                f"no active catalog resource for {key.canonical_name}",
            )
        entry = next(item for item in snapshot.entries if item.entry_id == selection.entry_id)
        resolved = self._resolve_entry(snapshot, entry)
        if require_fresh and resolved.state is not CatalogEntryState.AVAILABLE:
            code = {
                CatalogEntryState.STALE: CatalogErrorCode.STALE_RESOURCE,
                CatalogEntryState.MISSING: CatalogErrorCode.RESOURCE_MISSING,
                CatalogEntryState.CORRUPT: CatalogErrorCode.CORRUPT_RESOURCE,
            }.get(resolved.state, CatalogErrorCode.CORRUPT_RESOURCE)
            raise CatalogError(code, f"catalog resource is {resolved.state.value}")
        return resolved

    def list_entries(
        self,
        video_id: str,
        key: CatalogKey | None = None,
    ) -> tuple[CatalogEntry, ...]:
        entries = self._read(video_id).entries
        return entries if key is None else tuple(entry for entry in entries if entry.key == key)

    def find_reusable(
        self,
        video_id: str,
        key: CatalogKey,
        derivation_key: str,
        dependency_entry_ids: tuple[str, ...],
    ) -> ResolvedCatalogEntry | None:
        if not re.fullmatch(r"[0-9a-f]{64}", derivation_key):
            raise ValueError("derivation_key must be a lowercase SHA-256 digest")
        if len(set(dependency_entry_ids)) != len(dependency_entry_ids):
            raise ValueError("catalog dependencies must be unique")
        snapshot = self._read(video_id)
        known_entry_ids = {entry.entry_id for entry in snapshot.entries}
        if any(dependency not in known_entry_ids for dependency in dependency_entry_ids):
            raise CatalogError(
                CatalogErrorCode.DEPENDENCY_NOT_FOUND,
                "reusable resource query contains an unknown dependency",
            )
        for entry in reversed(snapshot.entries):
            if (
                entry.key != key
                or entry.derivation_key != derivation_key
                or entry.dependency_entry_ids != dependency_entry_ids
            ):
                continue
            resolved = self._resolve_entry(snapshot, entry)
            if resolved.state is not CatalogEntryState.AVAILABLE:
                continue
            try:
                path = self._entry_path(entry, require_exists=True)
                if self._fingerprint(path) != entry.content_sha256:
                    continue
            except (CatalogError, OSError):
                continue
            return resolved
        return None

    def load_manifest(
        self,
        video_id: str,
        key: CatalogKey,
        expected_type: type[T],
        *,
        require_fresh: bool = True,
    ) -> T:
        resolved = self.resolve(video_id, key, require_fresh=require_fresh)
        reference = resolved.entry.reference
        if not isinstance(reference, ManifestRef):
            raise CatalogError(
                CatalogErrorCode.TYPE_MISMATCH,
                "catalog resource is not a manifest",
            )
        try:
            expected_kind = self._codecs.expected_kind(expected_type)
        except TypeError as error:
            raise CatalogError(CatalogErrorCode.TYPE_MISMATCH, str(error)) from error
        if expected_kind is not reference.kind:
            raise CatalogError(
                CatalogErrorCode.TYPE_MISMATCH,
                "requested manifest type does not match catalog reference",
            )
        path = self._entry_path(resolved.entry, require_exists=True)
        if self._fingerprint(path) != resolved.entry.content_sha256:
            raise CatalogError(
                CatalogErrorCode.HASH_MISMATCH,
                f"manifest content changed: {reference.manifest_id}",
            )
        try:
            loaded = self._codecs.load(path, expected_type)
            self._validate_loaded_manifest(loaded, reference, key, video_id)
            self._validate_nested_artifacts(loaded)
        except CatalogError:
            raise
        except (TypeError, ValueError) as error:
            raise CatalogError(CatalogErrorCode.CORRUPT_RESOURCE, str(error)) from error
        return loaded

    def load_document(
        self,
        video_id: str,
        key: CatalogKey,
        expected_type: type[T],
        *,
        require_fresh: bool = True,
    ) -> T:
        resolved = self.resolve(video_id, key, require_fresh=require_fresh)
        reference = resolved.entry.reference
        if not isinstance(reference, CatalogDocumentRef):
            raise CatalogError(
                CatalogErrorCode.TYPE_MISMATCH,
                "catalog resource is not a typed document",
            )
        try:
            expected_kind = self._document_codecs.expected_kind(expected_type)
        except TypeError as error:
            raise CatalogError(CatalogErrorCode.TYPE_MISMATCH, str(error)) from error
        if expected_kind is not reference.kind:
            raise CatalogError(
                CatalogErrorCode.TYPE_MISMATCH,
                "requested document type does not match catalog reference",
            )
        path = self._entry_path(resolved.entry, require_exists=True)
        if self._fingerprint(path) != resolved.entry.content_sha256:
            raise CatalogError(
                CatalogErrorCode.HASH_MISMATCH,
                f"document content changed: {reference.document_id}",
            )
        try:
            loaded = self._document_codecs.load(path, expected_type)
            self._validate_loaded_document(loaded, reference, key, video_id)
            self._validate_nested_artifacts(loaded)
        except CatalogError:
            raise
        except (TypeError, ValueError) as error:
            raise CatalogError(CatalogErrorCode.CORRUPT_RESOURCE, str(error)) from error
        return loaded

    def load_artifact(
        self,
        video_id: str,
        key: CatalogKey,
        *,
        require_fresh: bool = True,
    ) -> ArtifactRef:
        resolved = self.resolve(video_id, key, require_fresh=require_fresh)
        if not isinstance(resolved.entry.reference, ArtifactRef):
            raise CatalogError(
                CatalogErrorCode.TYPE_MISMATCH,
                "catalog resource is structured metadata, not a direct artifact",
            )
        return resolved.entry.reference

    def audit(self, video_id: str, *, deep: bool = False) -> CatalogAuditReport:
        snapshot = self._read(video_id)
        active_ids = {item.entry_id for item in snapshot.active_selections}
        audits: list[CatalogEntryAudit] = []
        issues: list[CatalogAuditIssue] = []
        for entry in snapshot.entries:
            if entry.entry_id not in active_ids:
                audits.append(CatalogEntryAudit(entry.entry_id, CatalogEntryState.SUPERSEDED))
                continue
            resolved = self._resolve_entry(snapshot, entry)
            state = resolved.state
            if state is CatalogEntryState.MISSING:
                issues.append(
                    CatalogAuditIssue(
                        entry.entry_id,
                        CatalogAuditIssueCode.MISSING_FILE,
                        "catalog resource file is missing",
                    )
                )
            elif state is CatalogEntryState.CORRUPT:
                issues.append(
                    CatalogAuditIssue(
                        entry.entry_id,
                        CatalogAuditIssueCode.PATH_NOT_ALLOWED,
                        "catalog resource path is not allowed",
                    )
                )
            elif deep:
                state, deep_issues = self._deep_audit(entry)
                issues.extend(deep_issues)
            if state is CatalogEntryState.AVAILABLE and resolved.stale_dependency_entry_ids:
                state = CatalogEntryState.STALE
            if state is CatalogEntryState.STALE:
                issues.append(
                    CatalogAuditIssue(
                        entry.entry_id,
                        CatalogAuditIssueCode.STALE_DEPENDENCY,
                        "catalog resource depends on superseded entries",
                    )
                )
            audits.append(CatalogEntryAudit(entry.entry_id, state))
        return CatalogAuditReport(video_id, snapshot.revision, deep, tuple(audits), tuple(issues))

    def _deep_audit(
        self,
        entry: CatalogEntry,
    ) -> tuple[CatalogEntryState, tuple[CatalogAuditIssue, ...]]:
        path = self._entry_path(entry, require_exists=True)
        if self._fingerprint(path) != entry.content_sha256:
            return (
                CatalogEntryState.CORRUPT,
                (
                    CatalogAuditIssue(
                        entry.entry_id,
                        CatalogAuditIssueCode.HASH_MISMATCH,
                        "catalog resource content hash changed",
                    ),
                ),
            )
        if isinstance(entry.reference, ArtifactRef):
            return CatalogEntryState.AVAILABLE, ()
        try:
            if isinstance(entry.reference, CatalogDocumentRef):
                document_type = self._document_codecs.type_for_kind(entry.reference.kind)
                loaded = self._document_codecs.load(path, document_type)
                self._validate_loaded_document(
                    loaded,
                    entry.reference,
                    entry.key,
                    entry.video_id,
                )
            else:
                manifest_type = self._codecs.type_for_kind(entry.reference.kind)
                loaded = self._codecs.load(path, manifest_type)
                self._validate_loaded_manifest(
                    loaded,
                    entry.reference,
                    entry.key,
                    entry.video_id,
                )
            self._validate_nested_artifacts(loaded)
        except (CatalogError, TypeError, ValueError) as error:
            issue_code = (
                CatalogAuditIssueCode.INVALID_DOCUMENT
                if isinstance(entry.reference, CatalogDocumentRef)
                else CatalogAuditIssueCode.INVALID_MANIFEST
            )
            return (
                CatalogEntryState.CORRUPT,
                (
                    CatalogAuditIssue(
                        entry.entry_id,
                        issue_code,
                        str(error),
                    ),
                ),
            )
        return CatalogEntryState.AVAILABLE, ()

    def _resolve_entry(
        self,
        snapshot: CatalogSnapshot,
        entry: CatalogEntry,
    ) -> ResolvedCatalogEntry:
        try:
            path = self._entry_path(entry, require_exists=False)
        except CatalogError:
            return ResolvedCatalogEntry(entry, CatalogEntryState.CORRUPT)
        if not path.is_file():
            return ResolvedCatalogEntry(entry, CatalogEntryState.MISSING)
        active_ids = {selection.entry_id for selection in snapshot.active_selections}
        stale = tuple(
            dependency
            for dependency in entry.dependency_entry_ids
            if dependency not in active_ids
        )
        state = CatalogEntryState.STALE if stale else CatalogEntryState.AVAILABLE
        return ResolvedCatalogEntry(entry, state, stale)

    def _activate_snapshot(
        self,
        snapshot: CatalogSnapshot,
        entry: CatalogEntry,
    ) -> CatalogSnapshot:
        selections = self._with_selection(
            list(snapshot.active_selections),
            entry.key,
            entry.entry_id,
        )
        updated = CatalogSnapshot(
            snapshot.schema_version,
            snapshot.video_asset,
            snapshot.revision + 1,
            snapshot.entries,
            tuple(selections),
            snapshot.created_at,
            datetime.now(UTC),
        )
        self._write(updated)
        return updated

    @staticmethod
    def _with_selection(
        selections: list[CatalogSelection],
        key: CatalogKey,
        entry_id: str,
    ) -> list[CatalogSelection]:
        return [
            *(item for item in selections if item.key != key),
            CatalogSelection(key, entry_id),
        ]

    @staticmethod
    def _is_selected(snapshot: CatalogSnapshot, entry: CatalogEntry) -> bool:
        return any(
            selection.key == entry.key and selection.entry_id == entry.entry_id
            for selection in snapshot.active_selections
        )

    @staticmethod
    def _check_revision(snapshot: CatalogSnapshot, expected_revision: int | None) -> None:
        if expected_revision is not None and expected_revision != snapshot.revision:
            raise CatalogError(
                CatalogErrorCode.REVISION_CONFLICT,
                f"expected revision {expected_revision}, found {snapshot.revision}",
            )

    @staticmethod
    def _producer(registration: CatalogRegistration) -> tuple[str, str, str | None]:
        artifact = _reference_artifact(registration.reference)
        provenance = artifact.provenance
        if registration.producer_name is not None:
            assert registration.producer_version is not None
            return (
                registration.producer_name,
                registration.producer_version,
                registration.parameters_hash
                or (provenance.parameters_hash if provenance is not None else None),
            )
        if provenance is None:
            raise CatalogError(
                CatalogErrorCode.INVALID_REFERENCE,
                "catalog registration requires producer metadata or artifact provenance",
            )
        return (
            provenance.producer.name,
            provenance.producer.version,
            registration.parameters_hash or provenance.parameters_hash,
        )

    def _validate_registration_reference(
        self,
        video_id: str,
        key: CatalogKey,
        reference: ArtifactRef | CatalogDocumentRef | ManifestRef,
    ) -> Path:
        try:
            CatalogEntry(
                entry_id="validation",
                video_id=video_id,
                key=key,
                reference=reference,
                operation_id="validation",
                producer_name="validation",
                producer_version="1",
                dependency_entry_ids=(),
                content_sha256="0" * 64,
                size_bytes=0,
            )
        except ValueError as error:
            raise CatalogError(CatalogErrorCode.INVALID_REFERENCE, str(error)) from error
        return self._validate_artifact_path(
            _reference_artifact(reference),
            require_exists=True,
        )

    def _validate_nested_artifacts(self, value: Any) -> None:
        for artifact in _walk_artifacts(value):
            self._validate_artifact_path(artifact, require_exists=True)

    def _validate_loaded_manifest(
        self,
        loaded: Any,
        reference: ManifestRef,
        key: CatalogKey,
        video_id: str,
    ) -> None:
        loaded_ref = getattr(loaded, "ref", None)
        loaded_video_id = getattr(loaded, "video_id", None)
        if not isinstance(loaded_ref, ManifestRef) or loaded_video_id != video_id:
            raise ValueError("loaded manifest identity does not match catalog video")
        identity = (
            loaded_ref.manifest_id,
            loaded_ref.kind,
            loaded_ref.source_video_id,
            loaded_ref.item_count,
            loaded_ref.schema_version,
        )
        expected = (
            reference.manifest_id,
            reference.kind,
            reference.source_video_id,
            reference.item_count,
            reference.schema_version,
        )
        if identity != expected:
            raise ValueError("loaded manifest reference does not match catalog reference")
        if key.resource_type is CatalogResourceType.INDEX:
            if not isinstance(loaded, IndexManifest):
                raise ValueError("index catalog entry did not load an IndexManifest")
            if loaded.modality is not key.modality or loaded.index_kind is not key.index_kind:
                raise ValueError("loaded index metadata does not match catalog key")

    @staticmethod
    def _validate_loaded_document(
        loaded: Any,
        reference: CatalogDocumentRef,
        key: CatalogKey,
        video_id: str,
    ) -> None:
        loaded_ref = getattr(loaded, "ref", None)
        loaded_video_id = getattr(loaded, "video_id", None)
        if not isinstance(loaded_ref, CatalogDocumentRef) or loaded_video_id != video_id:
            raise ValueError("loaded document identity does not match catalog video")
        identity = (
            loaded_ref.document_id,
            loaded_ref.kind,
            loaded_ref.source_video_id,
            loaded_ref.schema_version,
        )
        expected = (
            reference.document_id,
            reference.kind,
            reference.source_video_id,
            reference.schema_version,
        )
        if identity != expected:
            raise ValueError("loaded document reference does not match catalog reference")
        if key.document_kind is not reference.kind:
            raise ValueError("loaded document metadata does not match catalog key")

    def _entry_path(self, entry: CatalogEntry, *, require_exists: bool) -> Path:
        return self._validate_artifact_path(entry.artifact, require_exists=require_exists)

    def _validate_artifact_path(
        self,
        artifact: ArtifactRef,
        *,
        require_exists: bool,
    ) -> Path:
        raw = Path(artifact.uri).expanduser()
        if raw.is_symlink():
            raise CatalogError(
                CatalogErrorCode.PATH_NOT_ALLOWED,
                f"symbolic-link artifact path is not allowed: {artifact.artifact_id}",
            )
        try:
            resolved = raw.resolve(strict=require_exists)
        except FileNotFoundError as error:
            raise CatalogError(
                CatalogErrorCode.RESOURCE_MISSING,
                f"artifact file does not exist: {artifact.artifact_id}",
            ) from error
        roots = (
            self._input_roots
            if artifact.kind is ArtifactKind.SOURCE_VIDEO
            else (self._artifact_root,)
        )
        if not any(resolved.is_relative_to(root) for root in roots):
            raise CatalogError(
                CatalogErrorCode.PATH_NOT_ALLOWED,
                f"artifact path is outside allowed roots: {artifact.artifact_id}",
            )
        if require_exists and not resolved.is_file():
            raise CatalogError(
                CatalogErrorCode.RESOURCE_MISSING,
                f"artifact is not a regular file: {artifact.artifact_id}",
            )
        return resolved

    def _read(self, video_id: str) -> CatalogSnapshot:
        path = self._catalog_path(video_id)
        if not path.is_file():
            raise CatalogError(
                CatalogErrorCode.VIDEO_NOT_FOUND,
                f"video catalog does not exist: {video_id}",
            )
        try:
            snapshot = TypeAdapter(CatalogSnapshot).validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError, TypeError) as error:
            raise CatalogError(
                CatalogErrorCode.CORRUPT_CATALOG,
                f"cannot load video catalog: {error}",
            ) from error
        if snapshot.schema_version != self.SCHEMA_VERSION:
            raise CatalogError(
                CatalogErrorCode.UNSUPPORTED_SCHEMA,
                f"unsupported catalog schema: {snapshot.schema_version}",
            )
        if snapshot.video_id != video_id:
            raise CatalogError(
                CatalogErrorCode.CORRUPT_CATALOG,
                "catalog video identity does not match its path",
            )
        return snapshot

    def _write(self, snapshot: CatalogSnapshot) -> None:
        import json

        path = self._catalog_path(snapshot.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            json_value(snapshot),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
            temp.write(payload)
            temp_path = Path(temp.name)
        os.replace(temp_path, path)

    @contextmanager
    def _locked(self, video_id: str) -> Iterator[None]:
        directory = self._catalog_path(video_id).parent
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".catalog.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _catalog_path(self, video_id: str) -> Path:
        self._validate_video_id(video_id)
        return self._catalog_root / video_id / "catalog.json"

    @staticmethod
    def _validate_video_id(video_id: str) -> None:
        if not _VIDEO_ID.fullmatch(video_id):
            raise CatalogError(
                CatalogErrorCode.PATH_NOT_ALLOWED,
                "video_id cannot be used as a catalog path component",
            )

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


def _reference_artifact(
    reference: ArtifactRef | CatalogDocumentRef | ManifestRef,
) -> ArtifactRef:
    return (
        reference.artifact
        if isinstance(reference, CatalogDocumentRef | ManifestRef)
        else reference
    )


def _walk_artifacts(value: Any) -> tuple[ArtifactRef, ...]:
    found: list[ArtifactRef] = []

    def visit(item: Any) -> None:
        if isinstance(item, ArtifactRef):
            found.append(item)
            return
        if isinstance(item, CatalogDocumentRef | ManifestRef):
            visit(item.artifact)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, tuple | list):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple({artifact.artifact_id: artifact for artifact in found}.values())

from __future__ import annotations

from typing import Protocol, TypeVar

from grounded_video_agent.domain import ArtifactRef, VideoAsset
from grounded_video_agent.workspace.catalog.contracts import (
    CatalogAuditReport,
    CatalogEntry,
    CatalogKey,
    CatalogRegistration,
    CatalogSnapshot,
    ResolvedCatalogEntry,
)

T = TypeVar("T")


class ArtifactCatalog(Protocol):
    def create_video(self, video_asset: VideoAsset) -> CatalogSnapshot: ...

    def get_snapshot(self, video_id: str) -> CatalogSnapshot: ...

    def register(
        self,
        video_id: str,
        registration: CatalogRegistration,
        *,
        expected_revision: int | None = None,
    ) -> CatalogSnapshot: ...

    def activate(
        self,
        video_id: str,
        key: CatalogKey,
        entry_id: str,
        *,
        expected_revision: int | None = None,
    ) -> CatalogSnapshot: ...

    def resolve(
        self,
        video_id: str,
        key: CatalogKey,
        *,
        require_fresh: bool = True,
    ) -> ResolvedCatalogEntry: ...

    def list_entries(
        self,
        video_id: str,
        key: CatalogKey | None = None,
    ) -> tuple[CatalogEntry, ...]: ...

    def find_reusable(
        self,
        video_id: str,
        key: CatalogKey,
        derivation_key: str,
        dependency_entry_ids: tuple[str, ...],
    ) -> ResolvedCatalogEntry | None: ...

    def load_manifest(
        self,
        video_id: str,
        key: CatalogKey,
        expected_type: type[T],
        *,
        require_fresh: bool = True,
    ) -> T: ...

    def load_document(
        self,
        video_id: str,
        key: CatalogKey,
        expected_type: type[T],
        *,
        require_fresh: bool = True,
    ) -> T: ...

    def load_artifact(
        self,
        video_id: str,
        key: CatalogKey,
        *,
        require_fresh: bool = True,
    ) -> ArtifactRef: ...

    def audit(self, video_id: str, *, deep: bool = False) -> CatalogAuditReport: ...

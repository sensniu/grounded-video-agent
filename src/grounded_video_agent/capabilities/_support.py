"""Internal helpers shared by deterministic local capabilities."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    ManifestKind,
    ManifestRef,
    ProducerInfo,
    Provenance,
)


def json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [json_value(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def parameters_hash(value: Any) -> str:
    payload = json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_provenance(
    producer_name: str,
    producer_version: str,
    parameters: Any,
    *,
    video_id: str,
    source_artifact_ids: tuple[str, ...] = (),
) -> Provenance:
    return Provenance(
        producer=ProducerInfo(producer_name, producer_version),
        parameters_hash=parameters_hash(parameters),
        source_video_id=video_id,
        source_artifact_ids=source_artifact_ids,
        created_at=datetime.now(UTC),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_value(value), ensure_ascii=False, indent=2, sort_keys=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp.write(payload)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def file_artifact(
    path: Path,
    *,
    artifact_id: str,
    kind: ArtifactKind,
    provenance: Provenance,
) -> ArtifactRef:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=str(path.resolve()),
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        provenance=provenance,
    )


def manifest_ref(
    path: Path,
    *,
    manifest_id: str,
    kind: ManifestKind,
    video_id: str,
    item_count: int,
    provenance: Provenance,
) -> ManifestRef:
    artifact = ArtifactRef(
        artifact_id=f"{manifest_id}_artifact",
        kind=ArtifactKind.MANIFEST,
        uri=str(path.resolve()),
        provenance=provenance,
    )
    return ManifestRef(
        manifest_id=manifest_id,
        kind=kind,
        artifact=artifact,
        source_video_id=video_id,
        item_count=item_count,
    )

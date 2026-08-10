from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from grounded_video_agent.infrastructure.visual_model import (
    VLLMBackendConfig,
    VLLMVisualModelBackend,
)
from grounded_video_agent.services.visual_model_api.app import create_app


@dataclass(frozen=True, slots=True)
class VisualModelServiceSettings:
    allowed_roots: tuple[Path, ...]
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_model_id: str | None = None
    vllm_api_key: str | None = None
    context_length: int | None = 8_192
    max_frames_per_target: int = 4
    max_image_edge: int = 1_536
    max_request_frames: int = 64
    max_request_targets: int = 16

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> VisualModelServiceSettings:
        values = os.environ if environ is None else environ
        raw_roots = values.get(
            "GVA_VLM_ALLOWED_ROOTS",
            values.get("GVA_ARTIFACT_ROOT", "artifacts"),
        )
        roots = tuple(
            Path(item.strip()).expanduser().resolve()
            for item in raw_roots.split(",")
            if item.strip()
        )
        if not roots:
            raise ValueError("GVA_VLM_ALLOWED_ROOTS must contain at least one path")
        return cls(
            allowed_roots=roots,
            vllm_base_url=_text(values, "GVA_VLLM_BASE_URL", "http://127.0.0.1:8000"),
            vllm_model_id=_optional_text(values, "GVA_VLLM_MODEL_ID"),
            vllm_api_key=_optional_text(values, "GVA_VLLM_API_KEY"),
            context_length=_optional_positive_int(values, "GVA_VLLM_CONTEXT_LENGTH", 8_192),
            max_frames_per_target=_positive_int(
                values,
                "GVA_VLLM_MAX_FRAMES_PER_TARGET",
                4,
            ),
            max_image_edge=_positive_int(values, "GVA_VLLM_MAX_IMAGE_EDGE", 1_536),
            max_request_frames=_positive_int(values, "GVA_VLM_MAX_REQUEST_FRAMES", 64),
            max_request_targets=_positive_int(values, "GVA_VLM_MAX_REQUEST_TARGETS", 16),
        )


def create_app_from_env() -> FastAPI:
    settings = VisualModelServiceSettings.from_environ()
    backend = VLLMVisualModelBackend(
        VLLMBackendConfig(
            base_url=settings.vllm_base_url,
            allowed_roots=settings.allowed_roots,
            model_id=settings.vllm_model_id,
            api_key=settings.vllm_api_key,
            context_length=settings.context_length,
            max_frames_per_target=settings.max_frames_per_target,
            max_image_edge=settings.max_image_edge,
        )
    )
    return create_app(
        backend,
        allowed_roots=settings.allowed_roots,
        max_frames=settings.max_request_frames,
        max_targets=settings.max_request_targets,
    )


def _text(values: Mapping[str, str], name: str, default: str) -> str:
    value = values.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_text(values: Mapping[str, str], name: str) -> str | None:
    value = values.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_positive_int(
    values: Mapping[str, str],
    name: str,
    default: int,
) -> int | None:
    raw = values.get(name, str(default)).strip()
    if not raw:
        return None
    return _positive_int({name: raw}, name, default)

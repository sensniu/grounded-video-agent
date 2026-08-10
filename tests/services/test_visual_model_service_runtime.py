from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast

import pytest
from fastapi.routing import APIRoute

from grounded_video_agent.infrastructure.visual_model import AsyncVisualModelBackend
from grounded_video_agent.services.visual_model_api import VisualModelServiceSettings
from grounded_video_agent.services.visual_model_api.app import create_app


def test_visual_service_settings_map_environment(tmp_path: Path) -> None:
    first = tmp_path / "artifacts-a"
    second = tmp_path / "artifacts-b"
    settings = VisualModelServiceSettings.from_environ(
        {
            "GVA_VLM_ALLOWED_ROOTS": f"{first},{second}",
            "GVA_VLLM_BASE_URL": "http://vllm.test:8000",
            "GVA_VLLM_MODEL_ID": "qwen3-vl-4b",
            "GVA_VLLM_API_KEY": "secret",
            "GVA_VLLM_CONTEXT_LENGTH": "8192",
            "GVA_VLLM_MAX_FRAMES_PER_TARGET": "4",
            "GVA_VLLM_MAX_IMAGE_EDGE": "1280",
        }
    )

    assert settings.allowed_roots == (first.resolve(), second.resolve())
    assert settings.vllm_model_id == "qwen3-vl-4b"
    assert settings.vllm_api_key == "secret"
    assert settings.context_length == 8_192
    assert settings.max_frames_per_target == 4
    assert settings.max_image_edge == 1_280


def test_visual_service_settings_reject_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        VisualModelServiceSettings.from_environ(
            {
                "GVA_VLM_ALLOWED_ROOTS": str(tmp_path),
                "GVA_VLLM_MAX_FRAMES_PER_TARGET": "0",
            }
        )


def test_visual_service_offloads_blocking_backend_calls(
    tmp_path: Path,
) -> None:
    backend = cast(AsyncVisualModelBackend, object())
    app = create_app(backend, allowed_roots=(tmp_path,))
    routes = {
        route.path: route.endpoint for route in app.routes if isinstance(route, APIRoute)
    }

    assert inspect.iscoroutinefunction(routes["/health"])
    assert inspect.iscoroutinefunction(routes["/v1/analyze"])

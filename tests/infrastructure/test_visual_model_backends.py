from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from PIL import Image

from grounded_video_agent.infrastructure.visual_model import (
    FastAPIVisualModelClient,
    LlamaCppBackendConfig,
    LlamaCppVisualModelBackend,
    VisualModelBackendError,
    VisualModelFrame,
    VisualModelRequest,
    VisualModelTarget,
    VLLMBackendConfig,
    VLLMVisualModelBackend,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M"
VLLM_MODEL_ID = "qwen3-vl-4b"


def _model_payload() -> dict[str, object]:
    return {
        "models": [
            {
                "name": MODEL_ID,
                "capabilities": ["completion", "multimodal"],
            }
        ],
        "data": [
            {
                "id": MODEL_ID,
                "meta": {"n_ctx": 8192, "ftype": "Q4_K - Medium"},
            }
        ],
    }


def _config(tmp_path: Path, **overrides: object) -> LlamaCppBackendConfig:
    values: dict[str, object] = {
        "base_url": "http://llama.test:8080",
        "allowed_roots": (tmp_path,),
        "model_id": MODEL_ID,
        "transient_retries": 0,
    }
    values.update(overrides)
    return LlamaCppBackendConfig(**values)  # type: ignore[arg-type]


def _request(image_path: Path, *, two_targets: bool = False) -> VisualModelRequest:
    frames = (
        VisualModelFrame("frame-1", str(image_path), 1_000),
        VisualModelFrame("frame-2", str(image_path), 2_000),
    )
    targets = [VisualModelTarget("target-1", 0, 1_500, ("frame-1",))]
    if two_targets:
        targets.append(VisualModelTarget("target-2", 1_500, 2_500, ("frame-2",)))
    return VisualModelRequest("operation", "generic", frames, tuple(targets))


def _handler_with_completions(
    completions: list[str],
    captured: list[dict[str, object]],
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_model_payload())
        payload = json.loads(request.content)
        captured.append(payload)
        content = completions.pop(0)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    return httpx.MockTransport(handler)


def test_llama_cpp_backend_maps_multimodal_request(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    captured: list[dict[str, object]] = []
    transport = _handler_with_completions(
        ['```json\n{"description":"a red car","tags":["car"],"confidence":0.8}\n```'],
        captured,
    )
    backend = LlamaCppVisualModelBackend(_config(tmp_path), transport=transport)

    result = backend.analyze(_request(image))

    assert result.model.provider == "llama.cpp"
    assert result.model.context_length == 8192
    assert result.model.quantization == "Q4_K - Medium"
    assert result.observations[0].text == "a red car"
    assert result.observations[0].target_id == "target-1"
    assert result.observations[0].frame_ids == ("frame-1",)
    assert result.model_calls == 1
    assert captured[0]["model"] == MODEL_ID
    messages = cast(list[dict[str, Any]], captured[0]["messages"])
    image_url = messages[1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    response_format = cast(dict[str, Any], captured[0]["response_format"])
    assert response_format["type"] == "json_schema"


def test_llama_cpp_backend_returns_partial_target_warnings(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    backend = LlamaCppVisualModelBackend(
        _config(tmp_path),
        transport=_handler_with_completions(
            [
                '{"description":"visible person","tags":["person"]}',
                "not-json",
            ],
            [],
        ),
    )

    result = backend.analyze(_request(image, two_targets=True))

    assert len(result.observations) == 1
    assert result.model_calls == 2
    assert "target-2" in result.warnings[0].lower()


def test_llama_cpp_backend_rejects_untrusted_frame_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"outside")
    backend = LlamaCppVisualModelBackend(
        _config(tmp_path),
        transport=_handler_with_completions([], []),
    )

    with pytest.raises(VisualModelBackendError, match="not allowed"):
        backend.analyze(_request(outside))


def test_llama_cpp_backend_retries_transient_status(tmp_path: Path) -> None:
    image = tmp_path / "frame.webp"
    image.write_bytes(b"webp")
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_model_payload())
        post_count += 1
        if post_count == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"description":"recovered","tags":[]}'
                        }
                    }
                ]
            },
        )

    backend = LlamaCppVisualModelBackend(
        _config(tmp_path, transient_retries=1),
        transport=httpx.MockTransport(handler),
    )

    result = backend.analyze(_request(image))

    assert result.observations[0].text == "recovered"
    assert post_count == 2


def test_llama_cpp_config_rejects_unsafe_or_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowed_roots"):
        LlamaCppBackendConfig("http://localhost:8080", ())
    with pytest.raises(ValueError, match="max_frames_per_target"):
        LlamaCppBackendConfig(
            "http://localhost:8080",
            (tmp_path,),
            max_frames_per_target=0,
        )


@pytest.mark.asyncio
async def test_vllm_backend_maps_and_limits_multimodal_frames(tmp_path: Path) -> None:
    frames: list[VisualModelFrame] = []
    for index in range(6):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (2_000, 1_000), color=(index * 20, 0, 0)).save(path)
        frames.append(VisualModelFrame(f"frame-{index}", str(path), index * 1_000))
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": VLLM_MODEL_ID}]})
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"description":"a sequence of frames",'
                                '"tags":["sequence"],"confidence":0.75}'
                            )
                        }
                    }
                ]
            },
        )

    backend = VLLMVisualModelBackend(
        VLLMBackendConfig(
            "http://vllm.test:8000",
            (tmp_path,),
            model_id=VLLM_MODEL_ID,
            context_length=8_192,
            max_frames_per_target=4,
            max_image_edge=1_024,
            transient_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.analyze(
        VisualModelRequest(
            "vllm-operation",
            "question_conditioned",
            tuple(frames),
            (VisualModelTarget("target", 0, 6_000, tuple(item.frame_id for item in frames)),),
            question="What changes?",
        )
    )

    assert result.model.provider == "vllm"
    assert result.model.context_length == 8_192
    assert result.observations[0].text == "a sequence of frames"
    assert len(result.observations[0].frame_ids) == 4
    assert captured[0]["model"] == VLLM_MODEL_ID
    assert captured[0]["response_format"]["type"] == "json_schema"
    content = cast(list[dict[str, Any]], captured[0]["messages"])[1]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]
    assert len(image_items) == 4
    assert image_items[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_vllm_backend_rejects_untrusted_frame_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-vllm.png"
    Image.new("RGB", (10, 10)).save(outside)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"data": [{"id": VLLM_MODEL_ID}]})

    backend = VLLMVisualModelBackend(
        VLLMBackendConfig(
            "http://vllm.test:8000",
            (tmp_path,),
            model_id=VLLM_MODEL_ID,
            transient_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(VisualModelBackendError, match="not allowed"):
        await backend.analyze(_request(outside))


@pytest.mark.asyncio
async def test_vllm_backend_health_check_is_not_satisfied_by_cached_model_info(
    tmp_path: Path,
) -> None:
    health_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal health_calls
        if request.url.path == "/health":
            health_calls += 1
            return httpx.Response(200 if health_calls == 1 else 503)
        return httpx.Response(200, json={"data": [{"id": VLLM_MODEL_ID}]})

    backend = VLLMVisualModelBackend(
        VLLMBackendConfig(
            "http://vllm.test:8000",
            (tmp_path,),
            model_id=VLLM_MODEL_ID,
            transient_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )

    assert (await backend.get_model_info()).model_name == VLLM_MODEL_ID
    with pytest.raises(VisualModelBackendError, match="health check failed"):
        await backend.get_model_info()


def test_fastapi_visual_model_client_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        FastAPIVisualModelClient("visual-model:8081")


def test_fastapi_visual_model_client_defaults_to_four_minute_timeout() -> None:
    captured_timeout: list[dict[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_timeout.append(cast(dict[str, float | None], request.extensions["timeout"]))
        return httpx.Response(
            200,
            json={
                "model_name": "test-model",
                "model_version": "test-version",
                "provider": "test",
            },
        )

    client = FastAPIVisualModelClient(
        "http://visual-model.test:8081",
        transport=httpx.MockTransport(handler),
    )

    client.get_model_info()

    assert captured_timeout[0]["read"] == 240.0

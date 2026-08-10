from __future__ import annotations

from dataclasses import asdict
from urllib.parse import urlparse

import httpx

from grounded_video_agent.infrastructure.visual_model.backend import (
    VisualModelBackendError,
    VisualModelFrame,
    VisualModelInfo,
    VisualModelObservation,
    VisualModelRequest,
    VisualModelResponse,
    VisualModelTarget,
)


class FastAPIVisualModelClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 240.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        normalized_base_url = base_url.rstrip("/")
        parsed = urlparse(normalized_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an absolute HTTP(S) URL without credentials")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = normalized_base_url
        self._timeout = timeout_seconds
        self._transport = transport

    def get_model_info(self) -> VisualModelInfo:
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.get(f"{self._base_url}/health")
                response.raise_for_status()
            payload = response.json()
            return VisualModelInfo(
                model_name=str(payload["model_name"]),
                model_version=str(payload["model_version"]),
                provider=(str(payload["provider"]) if payload.get("provider") else "fastapi"),
                quantization=(
                    str(payload["quantization"]) if payload.get("quantization") else None
                ),
                context_length=(
                    int(payload["context_length"])
                    if payload.get("context_length") is not None
                    else None
                ),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise VisualModelBackendError(f"FastAPI health check failed: {error}") from error

    def analyze(self, request: VisualModelRequest) -> VisualModelResponse:
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(
                    f"{self._base_url}/v1/analyze",
                    json=asdict(request),
                )
                response.raise_for_status()
            payload = response.json()
            model = payload["model"]
            observations = tuple(
                VisualModelObservation(
                    target_id=str(item["target_id"]),
                    text=str(item["text"]),
                    frame_ids=tuple(str(value) for value in item["frame_ids"]),
                    tags=tuple(str(value) for value in item.get("tags", ())),
                    confidence=(
                        float(item["confidence"])
                        if item.get("confidence") is not None
                        else None
                    ),
                )
                for item in payload["observations"]
            )
            return VisualModelResponse(
                model=VisualModelInfo(
                    str(model["model_name"]),
                    str(model["model_version"]),
                    provider=str(model.get("provider", "fastapi")),
                    quantization=(
                        str(model["quantization"]) if model.get("quantization") else None
                    ),
                    context_length=(
                        int(model["context_length"])
                        if model.get("context_length") is not None
                        else None
                    ),
                ),
                observations=observations,
                warnings=tuple(str(item) for item in payload.get("warnings", ())),
                model_calls=int(payload.get("model_calls", 1)),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise VisualModelBackendError(f"FastAPI analysis failed: {error}") from error


__all__ = [
    "FastAPIVisualModelClient",
    "VisualModelFrame",
    "VisualModelTarget",
]

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from grounded_video_agent.infrastructure.visual_model.backend import (
    VisualModelBackendError,
    VisualModelFrame,
    VisualModelInfo,
    VisualModelObservation,
    VisualModelRequest,
    VisualModelResponse,
    VisualModelTarget,
)
from grounded_video_agent.infrastructure.visual_model.vllm_contracts import VLLMBackendConfig

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class VLLMVisualModelBackend:
    """Translate the project visual contract to vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        config: VLLMBackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._model_info: VisualModelInfo | None = None
        self._resolved_model_id: str | None = None

    async def get_model_info(self) -> VisualModelInfo:
        try:
            async with self._client() as client:
                health = await client.get("/health")
                health.raise_for_status()
                payload = await self._request_json(client, "GET", "/v1/models")
            info, model_id = self._parse_model_info(payload)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise VisualModelBackendError(f"vLLM health check failed: {error}") from error
        self._model_info = info
        self._resolved_model_id = model_id
        return info

    async def analyze(self, request: VisualModelRequest) -> VisualModelResponse:
        info = await self.get_model_info()
        model_id = self._resolved_model_id
        if model_id is None:
            raise VisualModelBackendError("vLLM model identity was not resolved")
        frames = {frame.frame_id: frame for frame in request.frames}
        if len(frames) != len(request.frames):
            raise VisualModelBackendError("visual request contains duplicate frame ids")
        observations: list[VisualModelObservation] = []
        warnings: list[str] = []
        model_calls = 0
        async with self._client() as client:
            for target in request.targets:
                model_calls += 1
                try:
                    observations.append(
                        await self._analyze_target(client, model_id, request, target, frames)
                    )
                except (VisualModelBackendError, httpx.HTTPError) as error:
                    warnings.append(f"Target {target.target_id} failed: {error}")
        if not observations:
            detail = "; ".join(warnings) or "vLLM returned no observations"
            raise VisualModelBackendError(detail)
        return VisualModelResponse(
            model=info,
            observations=tuple(observations),
            warnings=tuple(warnings),
            model_calls=model_calls,
        )

    async def _analyze_target(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        request: VisualModelRequest,
        target: VisualModelTarget,
        frames: dict[str, VisualModelFrame],
    ) -> VisualModelObservation:
        if not target.frame_ids:
            raise VisualModelBackendError("target contains no frames")
        try:
            target_frames = tuple(frames[frame_id] for frame_id in target.frame_ids)
        except KeyError as error:
            raise VisualModelBackendError(f"unknown frame id: {error.args[0]}") from error
        selected_frames = _uniform_sample(target_frames, self._config.max_frames_per_target)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._messages(request, target, selected_frames),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if self._config.use_json_schema:
            payload["response_format"] = self._response_format()
        response = await self._request_json(
            client,
            "POST",
            "/v1/chat/completions",
            json_payload=payload,
        )
        parsed = self._parse_completion(response)
        return VisualModelObservation(
            target_id=target.target_id,
            text=parsed["description"],
            frame_ids=tuple(frame.frame_id for frame in selected_frames),
            tags=parsed["tags"],
            confidence=parsed["confidence"],
        )

    def _messages(
        self,
        request: VisualModelRequest,
        target: VisualModelTarget,
        frames: tuple[VisualModelFrame, ...],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for index, frame in enumerate(frames, start=1):
            content.extend(
                (
                    {
                        "type": "text",
                        "text": f"Frame {index}, source timestamp {frame.timestamp_ms} ms.",
                    },
                    {"type": "image_url", "image_url": {"url": self._data_url(frame)}},
                )
            )
        content.append({"type": "text", "text": self._analysis_prompt(request, target)})
        return [
            {
                "role": "system",
                "content": (
                    "You inspect ordered video frames. Report only directly visible facts. "
                    "Do not infer identities, motives, causes, or unseen events. Return JSON only."
                ),
            },
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _analysis_prompt(request: VisualModelRequest, target: VisualModelTarget) -> str:
        timeline = f"Analyze the ordered frames from {target.start_ms} ms to {target.end_ms} ms."
        if request.mode == "question_conditioned":
            question = request.question or ""
            instruction = (
                f"Extract only visible evidence relevant to this question: {question}. "
                "Do not answer beyond the visible evidence."
            )
        else:
            instruction = (
                "Describe visible people, objects, actions, scene changes, and readable text."
            )
        return (
            f"{timeline} {instruction} Return an object with description, tags, and optional "
            "confidence between 0 and 1."
        )

    def _data_url(self, frame: VisualModelFrame) -> str:
        path = Path(frame.uri).expanduser().resolve()
        if not path.is_file() or not any(
            path.is_relative_to(root) for root in self._config.allowed_roots
        ):
            raise VisualModelBackendError(f"frame path is not allowed: {frame.frame_id}")
        size = path.stat().st_size
        if size <= 0 or size > self._config.max_source_image_bytes:
            raise VisualModelBackendError(
                "source frame size must be between 1 and "
                f"{self._config.max_source_image_bytes} bytes"
            )
        try:
            with Image.open(path) as source:
                width, height = source.size
                if width <= 0 or height <= 0:
                    raise VisualModelBackendError(f"frame has invalid dimensions: {frame.frame_id}")
                if width * height > self._config.max_source_image_pixels:
                    raise VisualModelBackendError(f"frame has too many pixels: {frame.frame_id}")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail(
                    (self._config.max_image_edge, self._config.max_image_edge),
                    Image.Resampling.LANCZOS,
                )
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=88, optimize=True)
        except (OSError, UnidentifiedImageError) as error:
            raise VisualModelBackendError(
                f"frame is not a readable image: {frame.frame_id}"
            ) from error
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempts = self._config.transient_retries + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = await client.request(method, path, json=json_payload)
                if response.status_code in _TRANSIENT_STATUS_CODES:
                    last_error = httpx.HTTPStatusError(
                        f"transient vLLM status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise VisualModelBackendError("vLLM returned a non-object response")
                return payload
            except (httpx.TransportError, json.JSONDecodeError) as error:
                last_error = error
        if last_error is None:
            raise VisualModelBackendError("vLLM request failed")
        raise VisualModelBackendError(str(last_error)) from last_error

    def _parse_model_info(self, payload: dict[str, Any]) -> tuple[VisualModelInfo, str]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("/v1/models returned no model data")
        configured = self._config.model_id
        selected = next(
            (
                item
                for item in data
                if isinstance(item, dict)
                and (configured is None or item.get("id") == configured)
            ),
            None,
        )
        if not isinstance(selected, dict) or not isinstance(selected.get("id"), str):
            raise ValueError(f"configured model is not served: {configured}")
        model_id = selected["id"]
        return (
            VisualModelInfo(
                model_name=model_id,
                model_version="served",
                provider="vllm",
                context_length=self._config.context_length,
            ),
            model_id,
        )

    @staticmethod
    def _parse_completion(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VisualModelBackendError("vLLM completion has no message content") from error
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise VisualModelBackendError("vLLM completion content is empty")
        try:
            parsed = json.loads(content.strip())
        except json.JSONDecodeError as error:
            raise VisualModelBackendError("vLLM completion is not valid JSON") from error
        if not isinstance(parsed, dict):
            raise VisualModelBackendError("vLLM completion JSON must be an object")
        description = parsed.get("description")
        tags = parsed.get("tags", ())
        if not isinstance(description, str) or not description.strip():
            raise VisualModelBackendError("completion description is missing")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise VisualModelBackendError("completion tags must be a string array")
        normalized_tags = tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        confidence = parsed.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            confidence = None
        elif not 0 <= confidence <= 1:
            confidence = None
        return {
            "description": description.strip(),
            "tags": normalized_tags,
            "confidence": float(confidence) if confidence is not None else None,
        }

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "visual_observation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "minLength": 1},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "confidence": {
                            "anyOf": [
                                {"type": "number", "minimum": 0, "maximum": 1},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["description", "tags", "confidence"],
                    "additionalProperties": False,
                },
            },
        }

    def _client(self) -> httpx.AsyncClient:
        headers = {"Accept": "application/json"}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        timeout = httpx.Timeout(
            self._config.inference_timeout_seconds,
            connect=self._config.connect_timeout_seconds,
        )
        return httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=headers,
            timeout=timeout,
            transport=self._transport,
        )


def _uniform_sample(
    frames: tuple[VisualModelFrame, ...],
    limit: int,
) -> tuple[VisualModelFrame, ...]:
    if len(frames) <= limit:
        return frames
    if limit == 1:
        return (frames[len(frames) // 2],)
    indices = tuple(round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit))
    return tuple(frames[index] for index in dict.fromkeys(indices))

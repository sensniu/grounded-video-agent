from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

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
from grounded_video_agent.infrastructure.visual_model.llama_cpp_contracts import (
    LlamaCppBackendConfig,
)

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_TRANSIENT_STATUS_CODES = {502, 503, 504}


class LlamaCppVisualModelBackend:
    def __init__(
        self,
        config: LlamaCppBackendConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._model_info: VisualModelInfo | None = None
        self._resolved_model_id: str | None = None

    def get_model_info(self) -> VisualModelInfo:
        if self._model_info is not None:
            return self._model_info
        try:
            with self._client() as client:
                health = client.get("/health")
                health.raise_for_status()
                payload = self._request_json(client, "GET", "/v1/models")
            info, model_id = self._parse_model_info(payload)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise VisualModelBackendError(f"llama.cpp health check failed: {error}") from error
        self._model_info = info
        self._resolved_model_id = model_id
        return info

    def analyze(self, request: VisualModelRequest) -> VisualModelResponse:
        info = self.get_model_info()
        model_id = self._resolved_model_id
        if model_id is None:
            raise VisualModelBackendError("llama.cpp model identity was not resolved")
        frames = {frame.frame_id: frame for frame in request.frames}
        if len(frames) != len(request.frames):
            raise VisualModelBackendError("visual request contains duplicate frame ids")
        observations: list[VisualModelObservation] = []
        warnings: list[str] = []
        model_calls = 0
        with self._client() as client:
            for target in request.targets:
                model_calls += 1
                try:
                    observation = self._analyze_target(
                        client,
                        model_id,
                        request,
                        target,
                        frames,
                    )
                except (VisualModelBackendError, httpx.HTTPError) as error:
                    warnings.append(f"Target {target.target_id} failed: {error}")
                    continue
                observations.append(observation)
        if not observations:
            detail = "; ".join(warnings) or "llama.cpp returned no observations"
            raise VisualModelBackendError(detail)
        return VisualModelResponse(
            model=info,
            observations=tuple(observations),
            warnings=tuple(warnings),
            model_calls=model_calls,
        )

    def _analyze_target(
        self,
        client: httpx.Client,
        model_id: str,
        request: VisualModelRequest,
        target: VisualModelTarget,
        frames: dict[str, VisualModelFrame],
    ) -> VisualModelObservation:
        if not target.frame_ids:
            raise VisualModelBackendError("target contains no frames")
        if len(target.frame_ids) > self._config.max_frames_per_target:
            raise VisualModelBackendError(
                f"target exceeds the {self._config.max_frames_per_target}-frame limit"
            )
        try:
            target_frames = tuple(frames[frame_id] for frame_id in target.frame_ids)
        except KeyError as error:
            raise VisualModelBackendError(f"unknown frame id: {error.args[0]}") from error
        messages = self._messages(request, target, target_frames)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if self._config.use_json_schema:
            payload["response_format"] = self._response_format()
        response_payload = self._request_json(
            client,
            "POST",
            "/v1/chat/completions",
            json_payload=payload,
        )
        parsed = self._parse_completion(response_payload)
        return VisualModelObservation(
            target_id=target.target_id,
            text=parsed["description"],
            frame_ids=target.frame_ids,
            tags=parsed["tags"],
            confidence=parsed["confidence"],
        )

    def _messages(
        self,
        request: VisualModelRequest,
        target: VisualModelTarget,
        frames: tuple[VisualModelFrame, ...],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You inspect ordered video frames. Report only directly visible facts. "
                    "Do not infer identities, motives, causes, or unseen events. Return JSON only."
                ),
            }
        ]
        for index, frame in enumerate(frames, start=1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Frame {index}, source timestamp {frame.timestamp_ms} ms.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._data_url(frame)},
                        },
                    ],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self._analysis_prompt(request, target),
            }
        )
        return messages

    @staticmethod
    def _analysis_prompt(request: VisualModelRequest, target: VisualModelTarget) -> str:
        timeline = f"Analyze the ordered frames from {target.start_ms} ms to {target.end_ms} ms."
        if request.mode == "question_conditioned":
            question = request.question or ""
            instruction = (
                f"Extract only visible evidence relevant to this question: {question}. "
                "Do not answer the question beyond the visible evidence."
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
        mime_type = _MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            raise VisualModelBackendError(f"unsupported frame type: {path.suffix}")
        size = path.stat().st_size
        if size <= 0 or size > self._config.max_image_bytes:
            raise VisualModelBackendError(
                f"frame size must be between 1 and {self._config.max_image_bytes} bytes"
            )
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _request_json(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempts = self._config.transient_retries + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = client.request(method, path, json=json_payload)
                if response.status_code in _TRANSIENT_STATUS_CODES:
                    last_error = httpx.HTTPStatusError(
                        f"transient llama.cpp status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise VisualModelBackendError("llama.cpp returned a non-object response")
                return payload
            except (httpx.TransportError, json.JSONDecodeError) as error:
                last_error = error
        if last_error is None:
            raise VisualModelBackendError("llama.cpp request failed")
        raise VisualModelBackendError(str(last_error)) from last_error

    def _parse_model_info(self, payload: dict[str, Any]) -> tuple[VisualModelInfo, str]:
        data = payload.get("data")
        models = payload.get("models")
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
        if isinstance(models, list):
            summary = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict)
                    and item.get("name", item.get("model")) == model_id
                ),
                None,
            )
            if isinstance(summary, dict):
                capabilities = summary.get("capabilities", ())
                if isinstance(capabilities, list) and "multimodal" not in capabilities:
                    raise ValueError("served model does not advertise multimodal capability")
        raw_meta = selected.get("meta")
        meta = cast(dict[str, Any], raw_meta) if isinstance(raw_meta, dict) else {}
        quantization = str(meta["ftype"]) if meta.get("ftype") else None
        context_length = int(meta["n_ctx"]) if meta.get("n_ctx") is not None else None
        return (
            VisualModelInfo(
                model_name=model_id,
                model_version=quantization or "unknown",
                provider="llama.cpp",
                quantization=quantization,
                context_length=context_length,
            ),
            model_id,
        )

    @staticmethod
    def _parse_completion(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VisualModelBackendError("llama.cpp completion has no message content") from error
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise VisualModelBackendError("llama.cpp completion content is empty")
        parsed = LlamaCppVisualModelBackend._parse_json_text(content)
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
    def _parse_json_text(content: str) -> dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            stripped = stripped[first_newline + 1 :] if first_newline >= 0 else stripped
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start < 0 or end <= start:
                raise VisualModelBackendError("completion is not valid JSON") from None
            try:
                parsed = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as error:
                raise VisualModelBackendError("completion is not valid JSON") from error
        if not isinstance(parsed, dict):
            raise VisualModelBackendError("completion JSON must be an object")
        return parsed

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "minLength": 1},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                },
                "required": ["description", "tags"],
                "additionalProperties": False,
            },
        }

    def _client(self) -> httpx.Client:
        headers = {"Accept": "application/json"}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        timeout = httpx.Timeout(
            self._config.inference_timeout_seconds,
            connect=self._config.connect_timeout_seconds,
        )
        return httpx.Client(
            base_url=self._config.base_url,
            headers=headers,
            timeout=timeout,
            transport=self._transport,
        )

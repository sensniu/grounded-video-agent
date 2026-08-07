from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded_video_agent.infrastructure.visual_model import (
    VisualModelBackend,
    VisualModelFrame,
    VisualModelRequest,
    VisualModelTarget,
)


class _FramePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)


class _TargetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    frame_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_range(self) -> _TargetPayload:
        if self.start_ms >= self.end_ms:
            raise ValueError("target start_ms must be less than end_ms")
        if not self.frame_ids or len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("target frame_ids must be non-empty and unique")
        return self


class _AnalyzePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    frames: tuple[_FramePayload, ...]
    targets: tuple[_TargetPayload, ...]
    question: str | None = None

    @model_validator(mode="after")
    def validate_references(self) -> _AnalyzePayload:
        frame_ids = tuple(frame.frame_id for frame in self.frames)
        target_ids = tuple(target.target_id for target in self.targets)
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame ids must be unique")
        if not self.targets or len(set(target_ids)) != len(target_ids):
            raise ValueError("target ids must be non-empty and unique")
        known_frames = set(frame_ids)
        if any(not set(target.frame_ids).issubset(known_frames) for target in self.targets):
            raise ValueError("targets must reference submitted frames")
        return self


def create_app(
    backend: VisualModelBackend,
    *,
    allowed_roots: tuple[str | Path, ...],
) -> FastAPI:
    """Create a single-worker app; model inference is intentionally serialized."""
    roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
    if not roots:
        raise ValueError("at least one allowed frame root is required")
    app = FastAPI(title="Grounded Video Agent Visual Model API", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, object]:
        info = backend.get_model_info()
        return {
            "status": "ok",
            "model_name": info.model_name,
            "model_version": info.model_version,
            "provider": info.provider,
            "quantization": info.quantization,
            "context_length": info.context_length,
        }

    @app.post("/v1/analyze")
    async def analyze(payload: _AnalyzePayload) -> dict[str, object]:
        frames = tuple(
            VisualModelFrame(item.frame_id, item.uri, item.timestamp_ms)
            for item in payload.frames
        )
        _validate_frame_paths(frames, roots)
        request = VisualModelRequest(
            operation_id=payload.operation_id,
            mode=payload.mode,
            question=payload.question,
            frames=frames,
            targets=tuple(
                VisualModelTarget(
                    item.target_id,
                    item.start_ms,
                    item.end_ms,
                    item.frame_ids,
                )
                for item in payload.targets
            ),
        )
        try:
            result = backend.analyze(request)
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "model": {
                "model_name": result.model.model_name,
                "model_version": result.model.model_version,
                "provider": result.model.provider,
                "quantization": result.model.quantization,
                "context_length": result.model.context_length,
            },
            "observations": [
                {
                    "target_id": item.target_id,
                    "text": item.text,
                    "frame_ids": item.frame_ids,
                    "tags": item.tags,
                    "confidence": item.confidence,
                }
                for item in result.observations
            ],
            "warnings": result.warnings,
            "model_calls": result.model_calls,
        }

    return app


def _validate_frame_paths(
    frames: tuple[VisualModelFrame, ...],
    allowed_roots: tuple[Path, ...],
) -> None:
    for frame in frames:
        path = Path(frame.uri).expanduser().resolve()
        if not path.is_file() or not any(path.is_relative_to(root) for root in allowed_roots):
            message = f"Frame path is not allowed: {frame.frame_id}"
            raise HTTPException(status_code=400, detail=message)

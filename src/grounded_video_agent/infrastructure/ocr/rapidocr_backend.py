from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

from grounded_video_agent.infrastructure.ocr.backend import (
    OCRDetection,
    OCRFrameInput,
    OCRFrameResult,
    OCRModelInfo,
)


class RapidOCRBackend:
    def __init__(self, *, engine: Any | None = None, params: dict[str, Any] | None = None) -> None:
        if engine is None:
            from rapidocr import RapidOCR

            engine = RapidOCR(params=params)
        self._engine: Any = engine

    def get_model_info(self) -> OCRModelInfo:
        return OCRModelInfo("PP-OCRv6", version("rapidocr"), "RapidOCR/ONNX Runtime")

    def recognize(self, frames: tuple[OCRFrameInput, ...]) -> tuple[OCRFrameResult, ...]:
        results: list[OCRFrameResult] = []
        for frame in frames:
            path = Path(frame.uri).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"OCR frame does not exist: {frame.frame_id}")
            output: Any = self._engine(path)
            image = output.img
            if image is None or len(image.shape) < 2:
                raise RuntimeError(f"RapidOCR returned no source image: {frame.frame_id}")
            height, width = int(image.shape[0]), int(image.shape[1])
            boxes = output.boxes if output.boxes is not None else ()
            texts = output.txts if output.txts is not None else ()
            scores = output.scores if output.scores is not None else ()
            if not (len(boxes) == len(texts) == len(scores)):
                raise RuntimeError("RapidOCR returned inconsistent result lengths")
            detections = tuple(
                OCRDetection(
                    text=str(text),
                    polygon=tuple((float(point[0]), float(point[1])) for point in box),
                    confidence=float(score),
                )
                for box, text, score in zip(boxes, texts, scores, strict=True)
                if str(text).strip()
            )
            results.append(OCRFrameResult(frame.frame_id, width, height, detections))
        return tuple(results)

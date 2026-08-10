from __future__ import annotations

from grounded_video_agent.agent import build_local_video_agent
from grounded_video_agent.infrastructure.llm import (
    DeepSeekBackendConfig,
    DeepSeekLLMBackend,
)
from grounded_video_agent.infrastructure.ocr import RapidOCRBackend
from grounded_video_agent.infrastructure.visual_model import (
    FastAPIVisualModelClient,
    LlamaCppBackendConfig,
    LlamaCppVisualModelBackend,
    VisualModelBackend,
)

from .adapter import AgentInvoker
from .config import CLIRuntimeSettings, OCRProvider, VLMProvider


def build_cli_agent(settings: CLIRuntimeSettings) -> AgentInvoker:
    llm_backend = DeepSeekLLMBackend(
        DeepSeekBackendConfig(
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
        )
    )
    ocr_backend = (
        RapidOCRBackend() if settings.ocr_provider is OCRProvider.RAPIDOCR else None
    )
    visual_backend: VisualModelBackend | None = None
    if settings.vlm_provider is VLMProvider.LLAMA_CPP:
        visual_backend = LlamaCppVisualModelBackend(
            LlamaCppBackendConfig(
                base_url=settings.llama_cpp_base_url,
                allowed_roots=(settings.artifact_root,),
                model_id=settings.llama_cpp_model_id,
            )
        )
    elif settings.vlm_provider is VLMProvider.FASTAPI:
        visual_backend = FastAPIVisualModelClient(settings.fastapi_base_url)
    return build_local_video_agent(
        llm_backend,
        input_root=settings.input_root,
        artifact_root=settings.artifact_root,
        visual_backend=visual_backend,
        ocr_backend=ocr_backend,
    )

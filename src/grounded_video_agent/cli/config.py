from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from grounded_video_agent.infrastructure.llm import DeepSeekBackendConfig
from grounded_video_agent.infrastructure.visual_model import (
    FastAPIVisualModelClient,
    LlamaCppBackendConfig,
)

from .errors import CLIConfigurationError


class OCRProvider(StrEnum):
    OFF = "off"
    RAPIDOCR = "rapidocr"


class VLMProvider(StrEnum):
    OFF = "off"
    LLAMA_CPP = "llama-cpp"
    FASTAPI = "fastapi"


@dataclass(frozen=True, slots=True)
class CLIRuntimeSettings:
    input_root: Path
    artifact_root: Path
    deepseek_model: str
    deepseek_base_url: str
    ocr_provider: OCRProvider
    vlm_provider: VLMProvider
    llama_cpp_base_url: str
    fastapi_base_url: str
    llama_cpp_model_id: str | None = None

    def __post_init__(self) -> None:
        DeepSeekBackendConfig(
            model=self.deepseek_model,
            base_url=self.deepseek_base_url,
        )
        if self.vlm_provider is VLMProvider.LLAMA_CPP:
            LlamaCppBackendConfig(
                base_url=self.llama_cpp_base_url,
                allowed_roots=(self.artifact_root,),
                model_id=self.llama_cpp_model_id,
            )
        elif self.vlm_provider is VLMProvider.FASTAPI:
            FastAPIVisualModelClient(self.fastapi_base_url)

    @property
    def selected_vlm_base_url(self) -> str | None:
        if self.vlm_provider is VLMProvider.LLAMA_CPP:
            return self.llama_cpp_base_url
        if self.vlm_provider is VLMProvider.FASTAPI:
            return self.fastapi_base_url
        return None

    @classmethod
    def from_namespace(
        cls,
        namespace: Namespace,
        environ: Mapping[str, str],
    ) -> CLIRuntimeSettings:
        defaults = DeepSeekBackendConfig()
        try:
            return cls(
                input_root=_path_setting(
                    namespace,
                    "input_root",
                    environ,
                    "GVA_INPUT_ROOT",
                    "analyzed_video",
                ),
                artifact_root=_path_setting(
                    namespace,
                    "artifact_root",
                    environ,
                    "GVA_ARTIFACT_ROOT",
                    "artifacts",
                ),
                deepseek_model=_text_setting(
                    namespace,
                    "llm_model",
                    environ,
                    "GVA_DEEPSEEK_MODEL",
                    defaults.model,
                ),
                deepseek_base_url=_text_setting(
                    namespace,
                    "llm_base_url",
                    environ,
                    "GVA_DEEPSEEK_BASE_URL",
                    defaults.base_url,
                ),
                ocr_provider=OCRProvider(
                    _text_setting(
                        namespace,
                        "ocr",
                        environ,
                        "GVA_OCR_BACKEND",
                        OCRProvider.OFF.value,
                    )
                ),
                vlm_provider=VLMProvider(
                    _text_setting(
                        namespace,
                        "vlm",
                        environ,
                        "GVA_VLM_BACKEND",
                        VLMProvider.FASTAPI.value,
                    )
                ),
                llama_cpp_base_url=_text_setting(
                    namespace,
                    "vlm_url",
                    environ,
                    "GVA_LLAMA_CPP_BASE_URL",
                    "http://127.0.0.1:8080",
                ),
                fastapi_base_url=_text_setting(
                    namespace,
                    "vlm_url",
                    environ,
                    "GVA_FASTAPI_VLM_BASE_URL",
                    "http://127.0.0.1:8081",
                ),
                llama_cpp_model_id=_optional_text_setting(
                    namespace,
                    "vlm_model",
                    environ,
                    "GVA_LLAMA_CPP_MODEL_ID",
                ),
            )
        except ValueError as error:
            raise CLIConfigurationError(str(error)) from error


def _path_setting(
    namespace: Namespace,
    attribute: str,
    environ: Mapping[str, str],
    environment_name: str,
    default: str,
) -> Path:
    value = _text_setting(namespace, attribute, environ, environment_name, default)
    return Path(value).expanduser().resolve()


def _text_setting(
    namespace: Namespace,
    attribute: str,
    environ: Mapping[str, str],
    environment_name: str,
    default: str,
) -> str:
    cli_value = getattr(namespace, attribute, None)
    value = cli_value if cli_value is not None else environ.get(environment_name, default)
    if not isinstance(value, str) or not value.strip():
        raise CLIConfigurationError(f"{environment_name} must not be empty")
    return value.strip()


def _optional_text_setting(
    namespace: Namespace,
    attribute: str,
    environ: Mapping[str, str],
    environment_name: str,
) -> str | None:
    cli_value = getattr(namespace, attribute, None)
    value = cli_value if cli_value is not None else environ.get(environment_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()

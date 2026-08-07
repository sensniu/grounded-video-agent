from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from grounded_video_agent.infrastructure.visual_model import (
    LlamaCppBackendConfig,
    LlamaCppVisualModelBackend,
)

from .config import CLIRuntimeSettings, OCRProvider, VLMProvider


class CheckStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status is not CheckStatus.ERROR for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}


def run_doctor(
    settings: CLIRuntimeSettings,
    environ: Mapping[str, str],
    *,
    check_network: bool = True,
) -> DoctorReport:
    checks = [
        _python_check(),
        _executable_check("ffmpeg"),
        _executable_check("ffprobe"),
        _input_root_check(settings.input_root),
        _artifact_root_check(settings.artifact_root),
        _api_key_check(environ),
        _ocr_check(settings.ocr_provider),
    ]
    checks.append(_vlm_check(settings, check_network=check_network))
    checks.append(_gpu_check())
    return DoctorReport(tuple(checks))


def _python_check() -> DoctorCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return DoctorCheck("python", CheckStatus.OK, version)


def _executable_check(name: str) -> DoctorCheck:
    path = shutil.which(name)
    if path is None:
        return DoctorCheck(name, CheckStatus.ERROR, f"{name} was not found on PATH")
    return DoctorCheck(name, CheckStatus.OK, path)


def _input_root_check(path: Path) -> DoctorCheck:
    if not path.exists():
        return DoctorCheck("input_root", CheckStatus.ERROR, f"directory does not exist: {path}")
    if not path.is_dir():
        return DoctorCheck("input_root", CheckStatus.ERROR, f"not a directory: {path}")
    if not os.access(path, os.R_OK):
        return DoctorCheck("input_root", CheckStatus.ERROR, f"directory is not readable: {path}")
    return DoctorCheck("input_root", CheckStatus.OK, str(path))


def _artifact_root_check(path: Path) -> DoctorCheck:
    if path.exists():
        if not path.is_dir():
            return DoctorCheck("artifact_root", CheckStatus.ERROR, f"not a directory: {path}")
        if not os.access(path, os.W_OK):
            return DoctorCheck(
                "artifact_root", CheckStatus.ERROR, f"directory is not writable: {path}"
            )
        return DoctorCheck("artifact_root", CheckStatus.OK, str(path))
    parent = _nearest_existing_parent(path)
    if not os.access(parent, os.W_OK):
        return DoctorCheck(
            "artifact_root",
            CheckStatus.ERROR,
            f"cannot create {path}; parent is not writable: {parent}",
        )
    return DoctorCheck(
        "artifact_root",
        CheckStatus.OK,
        f"will be created under writable parent: {parent}",
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _api_key_check(environ: Mapping[str, str]) -> DoctorCheck:
    value = environ.get("DEEPSEEK_API_KEY")
    if value is None or not value.strip():
        return DoctorCheck(
            "deepseek_api_key",
            CheckStatus.ERROR,
            "DEEPSEEK_API_KEY is not configured",
        )
    return DoctorCheck("deepseek_api_key", CheckStatus.OK, "configured")


def _ocr_check(provider: OCRProvider) -> DoctorCheck:
    if provider is OCRProvider.OFF:
        return DoctorCheck("ocr", CheckStatus.SKIPPED, "disabled")
    missing = [
        package
        for package in ("rapidocr", "onnxruntime")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        return DoctorCheck(
            "ocr",
            CheckStatus.ERROR,
            f"missing package(s): {', '.join(missing)}",
        )
    return DoctorCheck("ocr", CheckStatus.OK, "RapidOCR and ONNX Runtime are available")


def _vlm_check(settings: CLIRuntimeSettings, *, check_network: bool) -> DoctorCheck:
    if settings.vlm_provider is VLMProvider.OFF:
        return DoctorCheck("vlm", CheckStatus.SKIPPED, "disabled")
    if not check_network:
        return DoctorCheck(
            "vlm", CheckStatus.SKIPPED, f"network check disabled: {settings.llama_cpp_base_url}"
        )
    try:
        backend = LlamaCppVisualModelBackend(
            LlamaCppBackendConfig(
                base_url=settings.llama_cpp_base_url,
                allowed_roots=(settings.artifact_root,),
                model_id=settings.llama_cpp_model_id,
            )
        )
        model = backend.get_model_info()
    except Exception as error:
        return DoctorCheck("vlm", CheckStatus.ERROR, str(error))
    return DoctorCheck(
        "vlm",
        CheckStatus.OK,
        f"{model.provider or 'llama.cpp'}/{model.model_name} at "
        f"{settings.llama_cpp_base_url}",
    )


def _gpu_check() -> DoctorCheck:
    if importlib.util.find_spec("torch") is None:
        return DoctorCheck("gpu", CheckStatus.WARNING, "PyTorch is not installed")
    try:
        import torch

        if torch.cuda.is_available():
            return DoctorCheck("gpu", CheckStatus.OK, torch.cuda.get_device_name(0))
        return DoctorCheck("gpu", CheckStatus.WARNING, "CUDA is not available; CPU mode only")
    except Exception as error:
        return DoctorCheck("gpu", CheckStatus.WARNING, f"could not inspect CUDA: {error}")

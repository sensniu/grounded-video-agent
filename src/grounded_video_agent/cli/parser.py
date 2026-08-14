from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gva",
        description="Evidence-grounded local video analysis Agent",
    )
    parser.add_argument("--version", action="version", version=_version_text())
    parser.add_argument("--debug", action="store_true", help="show a traceback on failure")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="answer a question about one local video")
    _add_runtime_arguments(analyze)
    analyze.add_argument("video", help="plain filename under the configured input root")
    analyze.add_argument("-q", "--question", required=True, help="question about the video")
    analyze.add_argument("--language", default="zh-CN", help="response language")
    analyze.add_argument("--request-id", help="optional stable request identifier")
    analyze.add_argument(
        "--evidence-clip",
        action="store_true",
        help="export a clip only after its evidence passes verification",
    )
    analyze.add_argument("--force-refresh", action="store_true", help="rebuild preprocessing")
    analyze.add_argument("--max-iterations", type=_positive_int)
    analyze.add_argument("--max-tool-calls", type=_positive_int)
    analyze.add_argument("--max-llm-calls", type=_positive_int)
    analyze.add_argument("--max-total-tokens", type=_positive_int)
    analyze.add_argument(
        "--progress",
        choices=("auto", "off", "compact", "verbose"),
        default="auto",
        help="progress display mode (default: auto)",
    )
    _add_output_arguments(analyze)

    doctor = subparsers.add_parser("doctor", help="check the local runtime without paid LLM calls")
    _add_runtime_arguments(doctor)
    doctor.add_argument(
        "--no-network",
        action="store_true",
        help="skip the selected visual backend health check",
    )
    _add_output_arguments(doctor, allow_file=False)
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", help="video input directory (env: GVA_INPUT_ROOT)")
    parser.add_argument("--artifact-root", help="artifact directory (env: GVA_ARTIFACT_ROOT)")
    parser.add_argument("--llm-model", help="DeepSeek model (env: GVA_DEEPSEEK_MODEL)")
    parser.add_argument("--llm-base-url", help="DeepSeek API URL (env: GVA_DEEPSEEK_BASE_URL)")
    parser.add_argument("--ocr", choices=("off", "rapidocr"), help="OCR backend")
    parser.add_argument(
        "--vlm",
        choices=("off", "llama-cpp", "fastapi"),
        help="visual model backend (default: fastapi)",
    )
    parser.add_argument("--vlm-url", help="selected visual backend URL")
    parser.add_argument("--vlm-model", help="optional served llama.cpp model id")


def _add_output_arguments(parser: argparse.ArgumentParser, *, allow_file: bool = True) -> None:
    parser.add_argument("--format", choices=("text", "json"), default="text")
    if allow_file:
        parser.add_argument("--output", help="write the result to a new file")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="allow --output to replace an existing file",
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _version_text() -> str:
    try:
        package_version = version("grounded-video-agent")
    except PackageNotFoundError:
        package_version = "development"
    return f"%(prog)s {package_version}"

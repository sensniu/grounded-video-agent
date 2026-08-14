from __future__ import annotations

import os
import sys
import traceback
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from dotenv import load_dotenv

from grounded_video_agent.agent import AgentStatus

from .adapter import AgentInvoker, AnalyzeOptions, invoke_agent
from .bootstrap import build_cli_agent
from .config import CLIRuntimeSettings
from .doctor import DoctorReport, run_doctor
from .errors import CLIConfigurationError, CLIError, ExitCode
from .parser import build_parser
from .progress import CLIProgressRenderer
from .rendering import render_agent_result, render_doctor_report

AgentFactory = Callable[[CLIRuntimeSettings], AgentInvoker]
DoctorRunner = Callable[[CLIRuntimeSettings, Mapping[str, str], bool], DoctorReport]


def main() -> int:
    load_dotenv()
    return run()


def run(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    agent_factory: AgentFactory = build_cli_agent,
    doctor_runner: DoctorRunner | None = None,
) -> int:
    namespace = build_parser().parse_args(argv)
    resolved_environ = os.environ if environ is None else environ
    try:
        settings = CLIRuntimeSettings.from_namespace(namespace, resolved_environ)
        if namespace.command == "analyze":
            return _run_analyze(namespace, settings, agent_factory)
        if namespace.command == "doctor":
            runner = doctor_runner or _doctor_runner
            report = runner(settings, resolved_environ, not namespace.no_network)
            print(render_doctor_report(report, namespace.format))
            return int(ExitCode.OK if report.ok else ExitCode.INVALID_CONFIGURATION)
        raise CLIConfigurationError(f"unknown command: {namespace.command}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
    except CLIError as error:
        _print_error(error, debug=namespace.debug)
        return int(error.exit_code)
    except Exception as error:
        _print_error(error, debug=namespace.debug)
        return int(ExitCode.RUNTIME_ERROR)


def _run_analyze(
    namespace: Namespace,
    settings: CLIRuntimeSettings,
    agent_factory: AgentFactory,
) -> int:
    _validate_filename(namespace.video)
    _validate_text(namespace.question, "question")
    _validate_text(namespace.language, "language")
    if namespace.request_id is not None:
        _validate_text(namespace.request_id, "request-id")
    output_path = None
    if namespace.output:
        output_path = _validated_output_path(
            Path(namespace.output),
            overwrite=namespace.overwrite,
            source_path=settings.input_root / namespace.video,
        )
    agent = agent_factory(settings)
    progress_renderer = CLIProgressRenderer(namespace.progress)
    with progress_renderer:
        result = invoke_agent(
            agent,
            AnalyzeOptions(
                filename=namespace.video,
                question=namespace.question,
                response_language=namespace.language,
                evidence_clip_requested=namespace.evidence_clip,
                force_refresh=namespace.force_refresh,
                request_id=namespace.request_id,
                max_iterations=namespace.max_iterations,
                max_tool_calls=namespace.max_tool_calls,
                max_llm_calls=namespace.max_llm_calls,
                max_total_tokens=namespace.max_total_tokens,
            ),
            progress=progress_renderer.emit if progress_renderer.enabled else None,
        )
    rendered = render_agent_result(result, namespace.format)
    if output_path is not None:
        _write_output(output_path, rendered)
    else:
        print(rendered)
    return int(ExitCode.RUNTIME_ERROR if result.status is AgentStatus.FAILED else ExitCode.OK)


def _doctor_runner(
    settings: CLIRuntimeSettings,
    environ: Mapping[str, str],
    check_network: bool,
) -> DoctorReport:
    return run_doctor(settings, environ, check_network=check_network)


def _validate_filename(filename: str) -> None:
    if not filename.strip() or Path(filename).name != filename:
        raise CLIConfigurationError(
            "video must be a plain filename under the configured input root"
        )


def _validate_text(value: str, name: str) -> None:
    if not value.strip():
        raise CLIConfigurationError(f"{name} must not be empty")


def _validated_output_path(path: Path, *, overwrite: bool, source_path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise CLIConfigurationError(f"output must not be a symbolic link: {expanded}")
    resolved = expanded.resolve()
    if resolved == source_path.resolve():
        raise CLIConfigurationError("output must not overwrite the source video")
    if resolved.exists() and not overwrite:
        raise CLIConfigurationError(
            f"output already exists: {resolved}; pass --overwrite to replace it"
        )
    if resolved.exists() and not resolved.is_file():
        raise CLIConfigurationError(f"output is not a regular file: {resolved}")
    return resolved


def _write_output(path: Path, content: str) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(f"{content}\n", encoding="utf-8")
    print(f"Result written to {resolved}", file=sys.stderr)


def _print_error(error: BaseException, *, debug: bool) -> None:
    print(f"error: {error}", file=sys.stderr)
    if debug:
        traceback.print_exc()

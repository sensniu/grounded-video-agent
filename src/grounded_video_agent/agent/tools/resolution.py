"""Resolve LLM-visible stable ids to catalog-backed source timeline targets."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.agent.tools.runtime import ToolRuntimeContext, merge_ranges, stable_id
from grounded_video_agent.domain import ChunkManifest, ShotManifest, TimeRange
from grounded_video_agent.pipelines.preprocessing.keys import CHUNKS_KEY, SHOTS_KEY


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    target_id: str
    time_range: TimeRange
    chunk_id: str | None = None


def resolve_targets(
    runtime: ToolRuntimeContext,
    *,
    candidate_ids: tuple[str, ...] = (),
    context_window_ids: tuple[str, ...] = (),
    chunk_ids: tuple[str, ...] = (),
    ranges: tuple[TimeRange, ...] = (),
) -> tuple[ResolvedTarget, ...]:
    chunks = runtime.catalog.load_manifest(runtime.video_id, CHUNKS_KEY, ChunkManifest)
    by_id = {chunk.chunk_id: chunk for chunk in chunks.chunks}
    resolved: list[ResolvedTarget] = []
    for candidate_id in candidate_ids:
        candidate = runtime.candidates.get(candidate_id)
        chunk = by_id.get(candidate.chunk_id)
        if chunk is None:
            raise KeyError(f"candidate references unknown chunk_id: {candidate.chunk_id}")
        resolved.append(ResolvedTarget(candidate_id, chunk.observation_range, chunk.chunk_id))
    for context_id in context_window_ids:
        context = runtime.contexts.get(context_id)
        for index, item in enumerate(context.ranges):
            resolved.append(ResolvedTarget(f"{context_id}_{index:03d}", item))
    for chunk_id in chunk_ids:
        try:
            chunk = by_id[chunk_id]
        except KeyError as error:
            raise KeyError(f"unknown chunk_id: {chunk_id}") from error
        resolved.append(ResolvedTarget(chunk_id, chunk.observation_range, chunk_id))
    for item in ranges:
        resolved.append(
            ResolvedTarget(stable_id("range", (runtime.video_id, item)), item)
        )
    return _unique_targets(tuple(resolved))


def load_shots(runtime: ToolRuntimeContext) -> ShotManifest:
    return runtime.catalog.load_manifest(runtime.video_id, SHOTS_KEY, ShotManifest)


def sampling_ranges(targets: tuple[ResolvedTarget, ...]) -> tuple[TimeRange, ...]:
    return merge_ranges(tuple(target.time_range for target in targets))


def _unique_targets(targets: tuple[ResolvedTarget, ...]) -> tuple[ResolvedTarget, ...]:
    unique: dict[tuple[int, int], ResolvedTarget] = {}
    for item in targets:
        key = (item.time_range.start_ms, item.time_range.end_ms)
        unique.setdefault(key, item)
    return tuple(sorted(unique.values(), key=lambda item: item.time_range))

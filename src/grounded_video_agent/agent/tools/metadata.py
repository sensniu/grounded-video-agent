from __future__ import annotations

from typing import cast

from grounded_video_agent.agent.tools._support import SCHEMA_VERSION, catalog_failure, start_tool
from grounded_video_agent.agent.tools.contracts import (
    GetVideoMetadataInput,
    StreamSummary,
    ToolProgress,
    ToolResult,
    ToolStatus,
    VideoMetadataOutput,
)
from grounded_video_agent.agent.tools.runtime import ToolRuntimeContext
from grounded_video_agent.pipelines.preprocessing.keys import (
    CHUNKS_KEY,
    DENSE_INDEX_KEY,
    MEDIA_INSPECTION_KEY,
    SPARSE_INDEX_KEY,
    TRANSCRIPT_KEY,
)
from grounded_video_agent.workspace.catalog import CatalogError, MediaInspectionDocument


class GetVideoMetadataTool:
    name = "get_video_metadata"
    enabled = True
    description = (
        "Read basic media facts, validation state, stream availability, and preprocessing "
        "readiness for the current video."
    )
    input_type = GetVideoMetadataInput

    def execute(
        self,
        request: GetVideoMetadataInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[VideoMetadataOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[VideoMetadataOutput], early)
        assert call_id is not None
        try:
            document = runtime.catalog.load_document(
                runtime.video_id,
                MEDIA_INSPECTION_KEY,
                MediaInspectionDocument,
            )
        except CatalogError as error:
            return cast(ToolResult[VideoMetadataOutput], catalog_failure(call_id, error))

        probe = document.media_probe
        primary_video = probe.primary_video_stream
        streams = self._streams(document) if request.include_streams else ()
        availability = {
            "transcript": self._available(runtime, TRANSCRIPT_KEY),
            "timeline": self._available(runtime, CHUNKS_KEY),
            "sparse": self._available(runtime, SPARSE_INDEX_KEY),
            "dense": self._available(runtime, DENSE_INDEX_KEY),
        }
        limitations = tuple(issue.message for issue in document.validation.issues)
        if not availability["transcript"]:
            limitations += ("No transcript is available; transcript search is disabled.",)
        if not availability["timeline"]:
            limitations += ("No logical chunk timeline is available.",)
        output = VideoMetadataOutput(
            video_id=runtime.video_id,
            display_name=document.video_asset.display_name,
            duration_ms=probe.container.duration_ms,
            format_names=probe.container.format_names,
            width=primary_video.width if primary_video is not None else None,
            height=primary_video.height if primary_video is not None else None,
            frame_rate=(
                primary_video.average_frame_rate.frames_per_second
                if primary_video is not None and primary_video.average_frame_rate is not None
                else None
            ),
            validation_status=document.validation.status.value,
            processable=document.validation.is_processable,
            next_action=document.next_action.value,
            has_audio=document.basic_flags.has_audio,
            has_embedded_subtitles=document.basic_flags.has_embedded_subtitles,
            transcript_ready=availability["transcript"],
            timeline_ready=availability["timeline"],
            sparse_search_ready=availability["sparse"],
            dense_search_ready=availability["dense"],
            streams=streams,
            limitations=limitations,
        )
        return ToolResult(
            SCHEMA_VERSION,
            call_id,
            ToolStatus.SUCCESS,
            output,
            progress=ToolProgress(no_information_gain=False),
        )

    @staticmethod
    def _available(runtime: ToolRuntimeContext, key: object) -> bool:
        try:
            runtime.catalog.resolve(runtime.video_id, key)  # type: ignore[arg-type]
        except CatalogError:
            return False
        return True

    @staticmethod
    def _streams(document: MediaInspectionDocument) -> tuple[StreamSummary, ...]:
        probe = document.media_probe
        video = tuple(
            StreamSummary(
                item.stream_index,
                "video",
                item.codec_name,
                width=item.width,
                height=item.height,
                is_default=item.is_default,
            )
            for item in probe.video_streams
        )
        audio = tuple(
            StreamSummary(
                item.stream_index,
                "audio",
                item.codec_name,
                language=item.language,
                sample_rate_hz=item.sample_rate_hz,
                channels=item.channels,
                is_default=item.is_default,
            )
            for item in probe.audio_streams
        )
        subtitles = tuple(
            StreamSummary(
                item.stream_index,
                "subtitle",
                item.codec_name,
                language=item.language,
                is_default=item.is_default,
            )
            for item in probe.subtitle_streams
        )
        return (*video, *audio, *subtitles)

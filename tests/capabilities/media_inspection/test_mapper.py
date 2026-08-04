from grounded_video_agent.capabilities.media_inspection.mapper import map_ffprobe_payload

from .sample_payload import sample_ffprobe_payload


def test_maps_ffprobe_payload_to_normalized_media_probe() -> None:
    probe = map_ffprobe_payload("video-1", sample_ffprobe_payload())

    assert probe.container.format_names == ("mov", "mp4", "m4a", "3gp", "3g2", "mj2")
    assert probe.container.duration_ms == 12_345
    assert probe.primary_video_stream is not None
    assert probe.primary_video_stream.codec_name == "h264"
    assert probe.primary_video_stream.frame_rate is not None
    assert probe.primary_video_stream.frame_rate.numerator == 30_000
    assert probe.primary_audio_stream is not None
    assert probe.primary_audio_stream.sample_rate_hz == 48_000
    assert probe.primary_subtitle_stream is not None
    assert probe.primary_subtitle_stream.language == "eng"


def test_maps_embedded_subtitle_without_treating_it_as_generated_text() -> None:
    probe = map_ffprobe_payload("video-1", sample_ffprobe_payload())

    assert len(probe.subtitle_streams) == 1
    assert probe.subtitle_streams[0].codec_name == "mov_text"
    assert probe.subtitle_streams[0].title == "English"

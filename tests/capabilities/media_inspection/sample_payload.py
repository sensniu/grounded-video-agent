from typing import Any


def sample_ffprobe_payload(*, include_audio: bool = True) -> dict[str, Any]:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_name": "h264",
            "codec_long_name": "H.264",
            "profile": "Main",
            "codec_type": "video",
            "width": 1280,
            "height": 720,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "30000/1001",
            "time_base": "1/90000",
            "start_time": "0.000000",
            "duration": "12.345",
            "bit_rate": "1000000",
            "nb_frames": "370",
            "sample_aspect_ratio": "1:1",
            "display_aspect_ratio": "16:9",
            "disposition": {"default": 1, "attached_pic": 0},
            "tags": {"language": "und"},
        },
        {
            "index": 2,
            "codec_name": "mov_text",
            "codec_type": "subtitle",
            "time_base": "1/1000",
            "start_time": "0.000000",
            "duration": "12.345",
            "disposition": {"default": 1, "forced": 0},
            "tags": {"language": "eng", "title": "English"},
        },
    ]
    if include_audio:
        streams.insert(
            1,
            {
                "index": 1,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_fmt": "fltp",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "time_base": "1/48000",
                "start_time": "0.000000",
                "duration": "12.345",
                "bit_rate": "128000",
                "disposition": {"default": 1},
                "tags": {"language": "eng"},
            },
        )
    return {
        "streams": streams,
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "start_time": "0.000000",
            "duration": "12.345",
            "size": "1234567",
            "bit_rate": "1128000",
            "probe_score": 100,
            "tags": {"major_brand": "isom"},
        },
    }

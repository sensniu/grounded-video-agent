from pathlib import Path

import pytest

from grounded_video_agent.input import (
    RegistrationErrorCode,
    RegistrationStatus,
    VideoRegistrar,
)


def test_registers_plain_filename_as_stable_video_asset(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not-decoded-by-registration")
    registrar = VideoRegistrar(tmp_path, hash_chunk_size=4)

    first = registrar.register("sample.mp4")
    second = registrar.register("sample.mp4")

    assert first.status is RegistrationStatus.SUCCEEDED
    assert first.video_asset is not None
    assert first.file_info is not None
    assert first.video_asset.video_id == second.video_asset.video_id  # type: ignore[union-attr]
    assert first.video_asset.source.sha256 is not None
    assert first.video_asset.source.size_bytes == len(b"not-decoded-by-registration")
    assert first.file_info.relative_uri == f"{tmp_path.name}/sample.mp4"
    assert first.registration_id != second.registration_id


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        "../sample.mp4",
        "folder/sample.mp4",
        "folder\\sample.mp4",
        "/tmp/x",
        "bad\x00name.mp4",
    ],
)
def test_rejects_non_plain_filenames(tmp_path: Path, filename: str) -> None:
    result = VideoRegistrar(tmp_path).register(filename)

    assert result.status is RegistrationStatus.FAILED
    assert result.error is not None
    assert result.error.code is RegistrationErrorCode.INVALID_FILENAME


def test_reports_missing_file_without_trying_to_probe(tmp_path: Path) -> None:
    result = VideoRegistrar(tmp_path).register("missing.mp4")

    assert result.status is RegistrationStatus.FAILED
    assert result.error is not None
    assert result.error.code is RegistrationErrorCode.FILE_NOT_FOUND


def test_rejects_directories(tmp_path: Path) -> None:
    (tmp_path / "directory.mp4").mkdir()

    result = VideoRegistrar(tmp_path).register("directory.mp4")

    assert result.error is not None
    assert result.error.code is RegistrationErrorCode.NOT_A_FILE


def test_rejects_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "target.mp4"
    target.write_bytes(b"video")
    (tmp_path / "link.mp4").symlink_to(target)

    result = VideoRegistrar(tmp_path).register("link.mp4")

    assert result.error is not None
    assert result.error.code is RegistrationErrorCode.SYMLINK_NOT_ALLOWED

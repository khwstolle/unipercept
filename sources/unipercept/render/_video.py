"""
Tools for rendering videos
"""

import functools
import os
import shutil
import subprocess
import sys
import tempfile
import typing as T
from contextlib import contextmanager

import PIL.Image as pil_image

from unipercept.file_io import Path
from unipercept.log import logger
from unipercept.types import Pathable

__all__ = ["video_writer"]

_IgnoreExceptionsType: T.TypeAlias = type[Exception] | T.Callable[[Exception], bool]


@contextmanager
def video_writer(
    out_video: Pathable,
    out_frames: Pathable | None = None,
    *,
    fps: int = 30,
    overwrite: bool = False,
    ignore_exceptions: bool | _IgnoreExceptionsType = False,
):
    """
    Used for writing a sequence of PIL images to a (temporary) directory, and then
    encoding them into a video file using ``ffmpeg`` commands.
    """

    def _parse_output_path(out: Pathable) -> str:
        out = Path(out)
        if out.is_dir():
            shutil.rmtree(out)
        if out.is_file():
            if not overwrite:
                msg = f"File {out!r} already exists, and overwrite is set to False."
                raise FileExistsError(msg)
            out.unlink()
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    def _get_ffmpeg_path() -> str:
        return "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    def _get_ffmpeg_cmd(fps: int, dir_frames: str, out: str) -> tuple[str, ...]:
        frame_glob = Path(dir_frames) / "*.png"
        return (
            _get_ffmpeg_path(),
            f"-framerate {fps}",
            "-pattern_type glob",
            f"-i {str(frame_glob)!r}",
            "-c:v libx264",
            "-pix_fmt yuv420p",
            f"{str(out)!r}",
        )

    def _save_image(im: pil_image.Image, *, dir_frames: str):
        next_frame = len(os.listdir(dir_frames))
        im.save(Path(dir_frames) / f"frame_{next_frame:010d}.png")

    def _should_ignore(e: Exception) -> bool:
        if isinstance(ignore_exceptions, bool):
            return ignore_exceptions
        if isinstance(ignore_exceptions, type):
            return isinstance(e, ignore_exceptions)
        if callable(ignore_exceptions):
            return ignore_exceptions(e)
        msg = (
            "ignore_exceptions must be a bool, a (collection of) exception type(s), or "
            "a callable that takes an exception and returns a bool. "
            f"Got {ignore_exceptions!r} instead."
        )
        raise TypeError(msg)

    out_video = Path(out_video)
    assert out_video.suffix == ".mp4", f"Expected .mp4 extension, got {out_video!r}."

    out_frames = Path(out_frames) if out_frames else None

    write_video = isinstance(ignore_exceptions, bool) and ignore_exceptions or False

    with tempfile.TemporaryDirectory() as tmp_frames:
        try:
            yield functools.partial(_save_image, dir_frames=tmp_frames)
            write_video = True
        except Exception as e:  # noqa: PIE786
            if not _should_ignore(e):
                write_video = False
            if not write_video:
                raise
            logger.warning("Ignoring exception: %s", e)
        finally:
            if write_video:
                cmd = " ".join(
                    _get_ffmpeg_cmd(fps, tmp_frames, out=_parse_output_path(out_video))
                )
                logger.debug("Writing video: %s", cmd)
                res = subprocess.run(cmd, shell=True, capture_output=True, check=False)
                if res.returncode != 0:
                    msg = f"Failed to write video: {res}"
                    logger.error(msg)
            else:
                logger.debug("Video writing was skipped.")

        # Copy all frames to the output directory if one was provided
        if out_frames:
            out_frames = Path(out_frames)
            out_frames.mkdir(parents=True, exist_ok=True)
            for frame in os.listdir(tmp_frames):
                shutil.copy(Path(tmp_frames) / frame, out_frames)
            logger.debug("Frames were copied to %s.", out_frames)
        else:
            logger.debug("Individual frames were discarded, no path provided.")

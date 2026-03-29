"""
utils/error_handling.py – Failsafe helpers for Director-Local.

Covers:
  * OOM protection with automatic 720p downscale (Section 6.1)
  * "No Plot" fallback via CLI prompt (Section 6.2)
  * Corrupted subtitle recovery via Whisper (Section 6.3)
"""

import os
import subprocess
import tempfile

import psutil

from utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# OOM Protection (Section 6.1)
# ---------------------------------------------------------------------------

_SAFETY_MARGIN_GB = 2.0


def check_memory_headroom_gb() -> float:
    """Return available system RAM in GiB."""
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def maybe_downscale_to_720p(video_path: str, tmp_dir: str) -> str:
    """
    If available RAM is below the safety margin, transcode *video_path* to
    720p and return the path to the downscaled file; otherwise return the
    original path unchanged.
    """
    available = check_memory_headroom_gb()
    if available >= _SAFETY_MARGIN_GB:
        return video_path

    logger.warning(
        "Low memory detected (%.1f GiB available). "
        "Downscaling working copy to 720p to prevent OOM crash.",
        available,
    )

    out_path = os.path.join(tmp_dir, "working_720p.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "scale=-2:720",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg downscale failed:\n%s", result.stderr)
        raise RuntimeError("Failed to downscale video to 720p for OOM protection.")
    logger.info("Downscaled working copy saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# "No Plot" Fallback (Section 6.2)
# ---------------------------------------------------------------------------

def prompt_manual_plot() -> str:
    """
    Prompt the user to paste a plot synopsis directly into the terminal.
    Reads until the user enters a lone period on its own line.
    """
    print(
        "\n[WARNING] Automatic plot scraping failed.\n"
        "Please paste a brief plot synopsis below.\n"
        "Type a single period '.' on its own line when finished:\n"
    )
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)

    synopsis = "\n".join(lines).strip()
    if not synopsis:
        raise RuntimeError(
            "No plot synopsis provided. Cannot proceed without plot data."
        )
    return synopsis


# ---------------------------------------------------------------------------
# Corrupted Subtitle Recovery (Section 6.3)
# ---------------------------------------------------------------------------

def recover_subtitles_with_whisper(video_path: str, tmp_dir: str, model_size: str = "base") -> str:
    """
    Generate a fresh SRT transcript from *video_path* using OpenAI Whisper
    when the provided subtitle file is corrupted or unreadable.

    Returns the path to the newly generated .srt file.
    """
    import whisper  # imported lazily to avoid import-time cost when not needed

    logger.warning(
        "Subtitle file is corrupted. Falling back to Whisper transcription "
        "(model=%s). This may take several minutes…",
        model_size,
    )

    model = whisper.load_model(model_size)
    result = model.transcribe(video_path)

    srt_path = os.path.join(tmp_dir, "recovered_subtitles.srt")
    _write_whisper_srt(result["segments"], srt_path)
    logger.info("Whisper recovery complete. SRT saved to %s", srt_path)
    return srt_path


def _write_whisper_srt(segments: list, out_path: str) -> None:
    """Convert Whisper segments to a standard SRT file."""

    def _fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, seg in enumerate(segments, start=1):
            fh.write(f"{i}\n")
            fh.write(f"{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n")
            fh.write(f"{seg['text'].strip()}\n\n")

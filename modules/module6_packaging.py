"""
Module 6: Packaging & Output

Responsibilities
----------------
6.1  Execute the final FFmpeg master render to produce
     ``final_summary_ready.mp4``.
6.2  Feed the compiled transcript into a local Ollama LLM and write
     ``metadata.txt`` containing a viral title, description, chapter
     timestamps, and hashtags.
6.3  Purge the ``/tmp`` working directory on success.
"""

import json
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from utils.logger import get_logger, log_timing

logger = get_logger()

Chunk = Tuple[float, float, str]

# ---------------------------------------------------------------------------
# 6.1  Final Master Render
# ---------------------------------------------------------------------------

OUTPUT_FILENAME = "final_summary_ready.mp4"


@log_timing("Final Master Render")
def final_render(
    processed_video: str,
    vocals_eq_path: str,
    bg_music_path: Optional[str],
    captions_ass: Optional[str],
    tmp_dir: str,
    output_path: str = OUTPUT_FILENAME,
    test_mode: bool = False,
    test_duration: float = 30.0,
) -> str:
    """
    Compile all processed layers into the final output file.

    Layers applied (in order):
      - Subtitles (.ass overlay)
      - EQ-processed vocal audio (primary)
      - Ducked background music (if provided)

    *test_mode* limits the output to the first *test_duration* seconds.

    Returns the path to the rendered output file.
    """
    inputs: List[str] = ["-i", processed_video, "-i", vocals_eq_path]
    filter_parts: List[str] = []
    audio_inputs = ["[1:a]"]

    # Mix in background music if available
    if bg_music_path and os.path.isfile(bg_music_path):
        inputs += ["-i", bg_music_path]
        # Duck the background to -18 dB relative to vocals
        filter_parts.append(
            "[2:a]volume=0.15[bgm];"
            "[1:a][bgm]amix=inputs=2:duration=first[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = "1:a"

    # Subtitle overlay
    vf_parts: List[str] = []
    if captions_ass and os.path.isfile(captions_ass):
        # Escape path for FFmpeg subtitles filter
        safe_ass = captions_ass.replace("\\", "/").replace(":", "\\:")
        vf_parts.append(f"ass='{safe_ass}'")

    # Test mode: limit duration
    time_args: List[str] = []
    if test_mode:
        time_args = ["-t", str(test_duration)]
        logger.info("TEST MODE: rendering first %.0f seconds only.", test_duration)

    vf_str = ",".join(vf_parts) if vf_parts else "null"
    filter_complex = ";".join(filter_parts)

    cmd: List[str] = ["ffmpeg", "-y"] + inputs + time_args

    if filter_complex:
        cmd += ["-filter_complex", filter_complex]

    cmd += [
        "-vf", vf_str,
        "-map", "0:v",
        "-map", audio_map,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info("Starting final render → %s", output_path)
    subprocess.run(cmd, check=True)
    logger.info("Final render complete: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 6.2  SEO Metadata Generation
# ---------------------------------------------------------------------------

@log_timing("SEO Metadata Generation")
def generate_metadata(
    selected_chunks: List[Chunk],
    chapters: List[Tuple[float, str]],
    ollama_model: str = "llama3",
    output_path: str = "metadata.txt",
) -> str:
    """
    Compile the transcript of *selected_chunks* and prompt the local Ollama
    LLM to produce a ``metadata.txt`` file containing:
      - 1× viral hook title
      - 2-sentence algorithm-optimised description
      - YouTube-formatted chapter timestamps
      - 5-10 targeted hashtags

    Returns the path to the written metadata file.
    """
    transcript = "\n\n".join(c[2] for c in selected_chunks)

    chapter_ts_lines = _format_chapter_timestamps(chapters, selected_chunks)

    prompt = (
        "You are a viral social-media video editor. "
        "Based on the following transcript from a movie mini-documentary, "
        "output ONLY the following (no extra commentary):\n\n"
        "TITLE: <one punchy viral hook title under 70 characters>\n\n"
        "DESCRIPTION: <two sentences optimised for social media algorithms>\n\n"
        "CHAPTERS:\n"
        + "\n".join(chapter_ts_lines)
        + "\n\nHASHTAGS: <5 to 10 comma-separated hashtags>\n\n"
        f"TRANSCRIPT:\n{transcript[:3000]}"
    )

    metadata_content = _call_ollama(ollama_model, prompt)

    if not metadata_content:
        logger.warning("Ollama metadata generation failed; writing placeholder.")
        metadata_content = _placeholder_metadata(selected_chunks, chapters, chapter_ts_lines)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(metadata_content)

    logger.info("Metadata written to %s", output_path)
    return output_path


def _call_ollama(model: str, prompt: str) -> str:
    """Call the local Ollama API and return the response text."""
    try:
        import ollama
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Ollama call failed: %s", exc)
        return ""


def _format_chapter_timestamps(
    chapters: List[Tuple[float, str]],
    selected_chunks: List[Chunk],
) -> List[str]:
    """Format chapter timestamps in YouTube style (MM:SS Title)."""
    lines = ["00:00 Intro"]
    for ts, title in chapters:
        # ts is absolute movie time; convert to relative position in the
        # compiled video by subtracting the start of the first selected chunk
        offset = selected_chunks[0][0] if selected_chunks else 0.0
        relative = max(0.0, ts - offset)
        m = int(relative // 60)
        s = int(relative % 60)
        lines.append(f"{m:02d}:{s:02d} {title}")
    return lines


def _placeholder_metadata(
    selected_chunks: List[Chunk],
    chapters: List[Tuple[float, str]],
    chapter_ts_lines: List[str],
) -> str:
    """Return a minimal placeholder metadata file."""
    return (
        "TITLE: Movie Mini-Documentary\n\n"
        "DESCRIPTION: An AI-curated 20-minute highlight reel of the film's "
        "most pivotal moments. Watch until the end for a surprise.\n\n"
        "CHAPTERS:\n"
        + "\n".join(chapter_ts_lines)
        + "\n\nHASHTAGS: #movies, #documentary, #film, #cinema, #mustwatch\n"
    )


# ---------------------------------------------------------------------------
# 6.3  Temp File Cleanup
# ---------------------------------------------------------------------------

def cleanup_tmp(tmp_dir: str) -> None:
    """
    Purge the temporary working directory after a successful render.
    """
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Temporary directory '%s' purged.", tmp_dir)
    else:
        logger.debug("Tmp dir '%s' does not exist; nothing to clean.", tmp_dir)

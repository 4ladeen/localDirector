#!/usr/bin/env python3
"""
director.py – Director-Local Automated Movie Summarisation & Packaging Engine
==============================================================================

Usage
-----
    python director.py \\
        --video  "movie.mp4" \\
        --subs   "movie.srt" \\
        --plot_url "https://en.wikipedia.org/wiki/The_Movie" \\
        --target_length 20

Optional flags
--------------
    --output       Output file path (default: final_summary_ready.mp4)
    --ollama_model  Local Ollama model name (default: llama3)
    --whisper_model Whisper model size (default: base)
    --bg_music      Path to background music file (optional)
    --hf_token      Hugging Face token for pyannote (or set HF_TOKEN env var)
    --test_mode     Render only the first 30 seconds (rapid A/B testing)
    --no_parallax   Skip the parallax VFX step
    --no_shake      Skip the camera shake VFX step
    --no_zoom       Skip the anti-boring zoom step

Environment variables
---------------------
    HF_TOKEN            Hugging Face access token (for pyannote.audio)
    DLIB_LANDMARKS_MODEL  Path to dlib shape_predictor_68_face_landmarks.dat
"""

import argparse
import os
import sys
import tempfile

from utils.logger import get_logger
from utils.error_handling import maybe_downscale_to_720p

logger = get_logger()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="director.py",
        description="Director-Local: automated 20-min vertical movie mini-doc engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    p.add_argument("--video", required=True, help="Path to source .mp4 or .mkv file.")
    p.add_argument("--subs", required=True, help="Path to source .srt subtitle file.")

    # Plot acquisition
    p.add_argument(
        "--plot_url",
        default="",
        help="Wikipedia or IMDb URL for plot scraping. "
             "Leave blank to be prompted for a manual synopsis.",
    )

    # Render settings
    p.add_argument(
        "--target_length",
        type=float,
        default=20.0,
        help="Target output duration in minutes.",
    )
    p.add_argument(
        "--output",
        default="final_summary_ready.mp4",
        help="Path for the final rendered MP4.",
    )

    # AI model selection
    p.add_argument("--ollama_model", default="llama3", help="Local Ollama LLM model name.")
    p.add_argument(
        "--whisper_model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (smaller = faster, less accurate).",
    )
    p.add_argument(
        "--sentence_transformer",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformer model for semantic scoring.",
    )

    # Optional inputs
    p.add_argument("--bg_music", default="", help="Path to background music WAV/MP3.")
    p.add_argument(
        "--hf_token",
        default="",
        help="Hugging Face token for pyannote diarization (or set HF_TOKEN env var).",
    )

    # VFX toggles
    p.add_argument("--no_parallax", action="store_true", help="Skip parallax VFX.")
    p.add_argument("--no_shake", action="store_true", help="Skip camera shake VFX.")
    p.add_argument("--no_zoom", action="store_true", help="Skip anti-boring zoom.")

    # Operational modes
    p.add_argument(
        "--test_mode",
        action="store_true",
        help="Render only the first 30 seconds (dry run for rapid testing).",
    )

    return p


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full Director-Local pipeline end-to-end."""

    logger.info("=" * 60)
    logger.info("Director-Local pipeline starting.")
    logger.info("  Video     : %s", args.video)
    logger.info("  Subs      : %s", args.subs)
    logger.info("  Plot URL  : %s", args.plot_url or "(manual input)")
    logger.info("  Target    : %.0f min", args.target_length)
    logger.info("  Test mode : %s", args.test_mode)
    logger.info("=" * 60)

    # Create a managed temporary directory
    tmp_dir = tempfile.mkdtemp(prefix="director_local_")
    logger.info("Temporary working directory: %s", tmp_dir)

    try:
        _run(args, tmp_dir)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        # Only clean up on success; leave tmp on failure for debugging
        pass


def _run(args: argparse.Namespace, tmp_dir: str) -> None:
    target_seconds = args.target_length * 60.0

    # ------------------------------------------------------------------
    # Module 1: Ingestion & Preparation
    # ------------------------------------------------------------------
    from modules.module1_ingestion import validate_inputs, scrape_plot, chunk_subtitles
    from utils.error_handling import recover_subtitles_with_whisper

    validate_inputs(args.video, args.subs)

    if args.plot_url:
        plot_synopsis = scrape_plot(args.plot_url)
    else:
        from utils.error_handling import prompt_manual_plot
        plot_synopsis = prompt_manual_plot()

    try:
        chunks = chunk_subtitles(args.subs)
    except ValueError as exc:
        logger.warning("Subtitle parsing failed (%s). Attempting Whisper recovery…", exc)
        recovered_srt = recover_subtitles_with_whisper(
            args.video, tmp_dir, model_size=args.whisper_model
        )
        chunks = chunk_subtitles(recovered_srt)

    # ------------------------------------------------------------------
    # Module 2: AI Curation & Narrative Structuring
    # ------------------------------------------------------------------
    from modules.module2_curation import (
        load_encoder, score_chunks, select_clips,
        find_cold_open, identify_broll, generate_chapters,
    )

    encoder = load_encoder(args.sentence_transformer)
    scored_chunks = score_chunks(chunks, plot_synopsis, encoder)
    selected, discarded = select_clips(scored_chunks, target_seconds)
    cold_open = find_cold_open(scored_chunks, chunks, plot_synopsis, encoder)
    broll_map = identify_broll(selected, discarded, encoder)
    chapters = generate_chapters(selected, ollama_model=args.ollama_model)

    logger.info("Selected %d clips; cold open at %.1fs.", len(selected), cold_open[0] if cold_open else 0)

    # ------------------------------------------------------------------
    # Slice and concatenate selected clips
    # ------------------------------------------------------------------
    assembled_video = _assemble_clips(args.video, selected, cold_open, tmp_dir)

    # ------------------------------------------------------------------
    # Module 3: Audio Engineering
    # ------------------------------------------------------------------
    from modules.module3_audio import separate_stems, apply_eq_compression, diarize_speakers, detect_impacts

    # Extract audio from assembled video for processing
    # OOM check before heavy AI models – downscale the video if needed
    assembled_video = maybe_downscale_to_720p(assembled_video, tmp_dir)

    raw_audio = os.path.join(tmp_dir, "raw_audio.wav")
    _extract_audio(assembled_video, raw_audio)

    vocals_path = separate_stems(raw_audio, tmp_dir)
    vocals_eq = apply_eq_compression(vocals_path, tmp_dir)
    diarization = diarize_speakers(vocals_eq, hf_token=args.hf_token)
    impact_times = detect_impacts(vocals_eq)

    # ------------------------------------------------------------------
    # Module 4: Video Processing & Dynamic Framing
    # ------------------------------------------------------------------
    from modules.module4_video import (
        remove_silence, analyse_crop_coordinates, apply_dynamic_crop, apply_antiboringzoom,
        _probe_video,
    )

    paced_video = remove_silence(assembled_video, tmp_dir)
    src_w, src_h, _ = _probe_video(paced_video)
    crop_kf = analyse_crop_coordinates(paced_video)
    cropped_video = apply_dynamic_crop(paced_video, crop_kf, tmp_dir, src_w, src_h)

    if not args.no_zoom:
        cropped_video = apply_antiboringzoom(cropped_video, tmp_dir)

    # ------------------------------------------------------------------
    # Module 5: VFX & Rendering
    # ------------------------------------------------------------------
    from modules.module5_vfx import (
        apply_parallax, apply_camera_shake, generate_captions, apply_perfect_loop,
    )

    vfx_video = cropped_video

    if not args.no_parallax:
        vfx_video = apply_parallax(vfx_video, tmp_dir)

    if not args.no_shake:
        vfx_video = apply_camera_shake(vfx_video, impact_times, tmp_dir)

    captions_ass = generate_captions(
        vocals_eq, diarization, tmp_dir, model_size=args.whisper_model
    )

    vfx_video = apply_perfect_loop(vfx_video, cold_open, tmp_dir)

    # ------------------------------------------------------------------
    # Module 6: Packaging & Output
    # ------------------------------------------------------------------
    from modules.module6_packaging import final_render, generate_metadata, cleanup_tmp

    bg_music = args.bg_music if args.bg_music and os.path.isfile(args.bg_music) else None

    final_render(
        processed_video=vfx_video,
        vocals_eq_path=vocals_eq,
        bg_music_path=bg_music,
        captions_ass=captions_ass,
        tmp_dir=tmp_dir,
        output_path=args.output,
        test_mode=args.test_mode,
    )

    generate_metadata(
        selected_chunks=selected,
        chapters=chapters,
        ollama_model=args.ollama_model,
    )

    cleanup_tmp(tmp_dir)

    logger.info("Pipeline complete! Output: %s", args.output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str, out_wav: str) -> None:
    """Extract audio track from *video_path* to a mono 16 kHz WAV."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out_wav,
    ]
    logger.info("Extracting audio from %s…", video_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _assemble_clips(
    source_video: str,
    selected,
    cold_open,
    tmp_dir: str,
) -> str:
    """
    Slice the selected chunks from *source_video*, prepend the cold open
    (with a monochrome filter), and concatenate into a single file.
    """
    import subprocess

    clip_paths = []
    cold_open_path = None

    # Cold open clip (monochrome + first position)
    if cold_open:
        cold_start, cold_end, _ = cold_open
        co_path = os.path.join(tmp_dir, "cold_open.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(cold_start),
            "-t", str(cold_end - cold_start),
            "-i", source_video,
            "-vf", "hue=s=0",   # monochrome
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            co_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cold_open_path = co_path
        clip_paths.append(co_path)

    # Main selected clips
    for i, (start, end, _) in enumerate(selected):
        clip_path = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(end - start),
            "-i", source_video,
            "-c", "copy",
            clip_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_paths.append(clip_path)

    if not clip_paths:
        raise RuntimeError("No clips were selected for assembly.")

    # Write concat list
    concat_list = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_list, "w") as fh:
        for p in clip_paths:
            fh.write(f"file '{os.path.abspath(p)}'\n")

    assembled = os.path.join(tmp_dir, "assembled.mp4")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        assembled,
    ]
    logger.info("Assembling %d clips…", len(clip_paths))
    subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Assembly complete: %s", assembled)
    return assembled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(args)

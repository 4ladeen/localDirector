"""
Module 5: Advanced Visual Effects & Rendering Layer

Responsibilities
----------------
5.1  2.5D Parallax: rembg foreground mask + opposing zoompan layers.
5.2  Audio-Reactive VFX: inject X/Y jitter crop at librosa impact timestamps.
5.3  Styled, diarized captions: Whisper → .ass file with per-speaker colours.
5.4  Dynamic Overlays: waveform visualiser, progress bar, chapter cards.
5.5  "Perfect Loop": duplicate first 1.5 s of cold open and append with xfade.
"""

import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from modules.module3_audio import SPEAKER_COLORS, DEFAULT_SPEAKER_COLOR
from utils.logger import get_logger, log_timing

logger = get_logger()

Chunk = Tuple[float, float, str]

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# ---------------------------------------------------------------------------
# 5.1  2.5D Parallax Depth Effect
# ---------------------------------------------------------------------------

@log_timing("Parallax VFX")
def apply_parallax(
    video_path: str,
    tmp_dir: str,
    bg_scale: float = 0.95,
    fg_scale: float = 1.05,
) -> str:
    """
    For designated static shots, use rembg to separate the foreground actor
    from the background, then apply opposing zoompan filters so the
    background recedes while the actor advances.

    Returns the path to the composited output file.
    """
    try:
        from rembg import remove
        from PIL import Image
        import numpy as np
        import cv2
    except ImportError as exc:
        logger.warning("rembg/PIL not available (%s); skipping parallax.", exc)
        return video_path

    # Extract a representative mid-frame to build the mask
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open %s for parallax; skipping.", video_path)
        return video_path

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning("Could not read mid-frame for parallax; skipping.")
        return video_path

    # Generate foreground mask via rembg
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fg_image = remove(pil_frame)
    alpha = np.array(fg_image)[:, :, 3]   # alpha channel = foreground mask

    mask_path = os.path.join(tmp_dir, "fg_mask.png")
    Image.fromarray(alpha).save(mask_path)

    # Build FFmpeg complex filter for parallax
    # Background: scale down (zoom out)
    # Foreground: scale up (zoom in) using alpha mask overlay
    bg_filter = (
        f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"zoompan=z='{bg_scale}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={TARGET_WIDTH}x{TARGET_HEIGHT}[bg]"
    )
    fg_filter = (
        f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"zoompan=z='{fg_scale}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={TARGET_WIDTH}x{TARGET_HEIGHT}[fg_scaled];"
        f"[1:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}[mask];"
        f"[fg_scaled][mask]alphamerge[fg];"
        f"[bg][fg]overlay=0:0[out]"
    )
    complex_filter = f"{bg_filter};{fg_filter}"

    out_path = os.path.join(tmp_dir, "parallax.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", mask_path,
        "-filter_complex", complex_filter,
        "-map", "[out]",
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "copy",
        out_path,
    ]
    logger.info("Applying parallax VFX…")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Parallax output saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 5.2  Audio-Reactive Camera Shake VFX
# ---------------------------------------------------------------------------

SHAKE_DURATION = 0.2   # seconds
SHAKE_PIXELS = 8       # maximum pixel jitter


@log_timing("Camera Shake VFX")
def apply_camera_shake(
    video_path: str,
    impact_times: List[float],
    tmp_dir: str,
) -> str:
    """
    At each timestamp in *impact_times*, inject a 0.2-second X/Y jitter
    crop to simulate a camera shake.
    """
    if not impact_times:
        logger.info("No impact timestamps; skipping camera shake.")
        return video_path

    # Build a crop filter expression that jitters during impact windows
    # crop=w:h:x:y  where x/y oscillate during the shake window
    shake_exprs = []
    for t in impact_times:
        t_end = t + SHAKE_DURATION
        shake_exprs.append(
            f"if(between(t,{t:.3f},{t_end:.3f}),"
            f"({SHAKE_PIXELS}*sin(t*200)),0)"
        )

    x_expr = "+".join(shake_exprs) if shake_exprs else "0"
    y_expr = "+".join(
        f"if(between(t,{t:.3f},{t + SHAKE_DURATION:.3f}),"
        f"({SHAKE_PIXELS}*cos(t*200)),0)"
        for t in impact_times
    ) if impact_times else "0"

    # Ensure crop stays inside frame by clamping
    crop_filter = (
        f"crop={TARGET_WIDTH - SHAKE_PIXELS * 2}:"
        f"{TARGET_HEIGHT - SHAKE_PIXELS * 2}:"
        f"({SHAKE_PIXELS}+({x_expr})):"
        f"({SHAKE_PIXELS}+({y_expr})),"
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    )

    out_path = os.path.join(tmp_dir, "shaken.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", crop_filter,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast",
        out_path,
    ]
    logger.info("Applying camera shake at %d timestamps…", len(impact_times))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Shake output saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 5.3  Styled, Diarized Captions
# ---------------------------------------------------------------------------

@log_timing("Caption Generation")
def generate_captions(
    vocals_path: str,
    diarization: List[Tuple[float, float, str]],
    tmp_dir: str,
    model_size: str = "base",
) -> str:
    """
    Transcribe *vocals_path* with Whisper and generate a diarized .ass
    subtitle file with per-speaker hex colours.

    Returns the path to the generated .ass file.
    """
    import whisper

    logger.info("Transcribing with Whisper (model=%s)…", model_size)
    model = whisper.load_model(model_size)
    result = model.transcribe(vocals_path, word_timestamps=True)

    ass_path = os.path.join(tmp_dir, "captions.ass")
    _write_ass(result["segments"], diarization, ass_path)
    logger.info("Diarized captions saved to %s", ass_path)
    return ass_path


def _speaker_at(t: float, diarization: List[Tuple[float, float, str]]) -> str:
    """Return the speaker label active at time *t*."""
    for start, end, speaker in diarization:
        if start <= t < end:
            return speaker
    return "SPEAKER_00"


def _write_ass(
    segments: list,
    diarization: List[Tuple[float, float, str]],
    out_path: str,
) -> None:
    """Write a styled ASS subtitle file from Whisper segments + diarization."""

    def _ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {TARGET_WIDTH}\n"
        f"PlayResY: {TARGET_HEIGHT}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,56,&H00FFFFFF&,&H000000FF&,&H00000000&,"
        "&H80000000&,-1,0,0,0,100,100,0,0,3,3,0,2,30,30,60,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        if not text:
            continue
        speaker = _speaker_at(start, diarization)
        colour = SPEAKER_COLORS.get(speaker, DEFAULT_SPEAKER_COLOR)
        styled_text = f"{{\\c{colour}}}{text}"
        lines.append(
            f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{styled_text}\n"
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# 5.4  Dynamic Overlays (waveform, progress bar, chapter cards)
# ---------------------------------------------------------------------------

def build_overlay_filter(
    vocals_path: str,
    total_duration: float,
    chapters: List[Tuple[float, str]],
    speaker_at_fn,
) -> str:
    """
    Return an FFmpeg complex-filter string that overlays:
      - A ``showwaves`` waveform at the bottom of the frame.
      - An animated progress bar via ``drawbox``.
      - Chapter card titles via ``drawtext`` at their milestone timestamps.

    The filter is designed to be appended to an existing filter graph.
    *speaker_at_fn* is a callable(t) → speaker_label used to colour the waveform.
    """
    # Waveform visualiser (pinned to bottom, 1080×120)
    waveform = (
        f"[audio]showwaves=s={TARGET_WIDTH}x120:mode=line:rate=25:"
        f"colors=0xFFFF00[wave];"
        f"[v][wave]overlay=0:{TARGET_HEIGHT - 120}[v_wave]"
    )

    # Progress bar: animated drawbox
    progress = (
        f"[v_wave]drawbox="
        f"x=0:y={TARGET_HEIGHT - 10}:"
        f"w='({TARGET_WIDTH}*t/{total_duration:.2f})':"
        f"h=10:color=white@0.9:t=fill[v_prog]"
    )

    # Chapter cards
    chapter_filters = []
    for i, (ts, title) in enumerate(chapters):
        display_until = ts + 3.0
        card = (
            f"drawtext=text='{title}':"
            f"fontsize=52:fontcolor=white:"
            f"x='(w-text_w)/2':"
            f"y='(h/2)-50':"
            f"enable='between(t,{ts:.2f},{display_until:.2f})':"
            f"box=1:boxcolor=black@0.6:boxborderw=12"
        )
        chapter_filters.append(card)

    chapter_chain = ",".join(chapter_filters) if chapter_filters else ""
    final_filter = f"{waveform};{progress}"
    if chapter_chain:
        final_filter += f";[v_prog]{chapter_chain}[v_final]"
    else:
        final_filter += ";[v_prog]null[v_final]"

    return final_filter


# ---------------------------------------------------------------------------
# 5.5  Perfect Loop
# ---------------------------------------------------------------------------

@log_timing("Perfect Loop")
def apply_perfect_loop(
    video_path: str,
    cold_open_chunk: Optional[Chunk],
    tmp_dir: str,
    loop_duration: float = 1.5,
    xfade_duration: float = 0.5,
) -> str:
    """
    Duplicate the first *loop_duration* seconds of *cold_open_chunk* and
    append it to the end of *video_path* with an xfade transition.

    Returns the path to the looped output file.
    """
    if cold_open_chunk is None:
        logger.info("No cold open defined; skipping perfect loop.")
        return video_path

    cold_start, cold_end, _ = cold_open_chunk
    clip_dur = min(loop_duration, cold_end - cold_start)

    # Extract the loop clip from the cold open
    loop_clip = os.path.join(tmp_dir, "loop_clip.mp4")
    cmd_extract = [
        "ffmpeg", "-y",
        "-ss", str(cold_start),
        "-t", str(clip_dur),
        "-i", video_path,
        "-c", "copy",
        loop_clip,
    ]
    subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Probe total duration of main video
    cmd_probe = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path,
    ]
    dur_data = json.loads(subprocess.check_output(cmd_probe).decode())
    main_dur = float(dur_data["format"]["duration"])

    # Concatenate main + loop clip with xfade
    offset = max(0.0, main_dur - xfade_duration)
    filelist = os.path.join(tmp_dir, "loop_concat.txt")
    with open(filelist, "w") as fh:
        fh.write(f"file '{os.path.abspath(video_path)}'\n")
        fh.write(f"file '{os.path.abspath(loop_clip)}'\n")

    # Use xfade transition
    xfade_filter = (
        f"[0:v][1:v]xfade=transition=fade:duration={xfade_duration}:"
        f"offset={offset:.3f}[vout];"
        f"[0:a][1:a]acrossfade=d={xfade_duration}[aout]"
    )

    out_path = os.path.join(tmp_dir, "perfect_loop.mp4")
    cmd_concat = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", loop_clip,
        "-filter_complex", xfade_filter,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        out_path,
    ]
    logger.info("Appending perfect loop (%.1fs xfade)…", xfade_duration)
    subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Perfect loop output saved to %s", out_path)
    return out_path

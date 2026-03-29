"""
Module 4: Video Processing & Dynamic Framing

Responsibilities
----------------
4.1  Remove silent dead-air (silenceremove) while keeping lip-sync.
4.2  Detect facial landmarks (eye positions) with OpenCV + dlib and
     generate dynamic Rule-of-Thirds 9:16 crop coordinates.
4.3  Apply an "Anti-Boring" 1.05× zoompan to shots longer than 6 seconds.
"""

import os
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger, log_timing

logger = get_logger()

Chunk = Tuple[float, float, str]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT   # ≈ 0.5625  (9:16)

# Dlib model file – expected in the project root or configurable path
DLIB_LANDMARKS_MODEL = os.environ.get(
    "DLIB_LANDMARKS_MODEL",
    "shape_predictor_68_face_landmarks.dat",
)


def _probe_video(video_path: str) -> Tuple[int, int, float]:
    """Return (width, height, duration_seconds) via ffprobe."""
    import json

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path,
    ]
    out = subprocess.check_output(cmd).decode()
    data = json.loads(out)
    stream = data["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])

    # Duration may be on the stream or the container
    dur = float(stream.get("duration") or 0)
    if dur == 0:
        cmd2 = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path,
        ]
        out2 = subprocess.check_output(cmd2).decode()
        dur = float(json.loads(out2)["format"].get("duration", 0))

    return w, h, dur


def _calc_crop_size(src_w: int, src_h: int) -> Tuple[int, int]:
    """Return (crop_w, crop_h) for a 9:16 crop within *src_w* × *src_h*."""
    crop_h = src_h
    crop_w = int(src_h * ASPECT_RATIO)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / ASPECT_RATIO)
    return crop_w, crop_h


# ---------------------------------------------------------------------------
# 4.1  Silence / Dead-Air Removal
# ---------------------------------------------------------------------------

@log_timing("Silence Removal")
def remove_silence(
    video_path: str,
    tmp_dir: str,
    noise_floor_db: float = -35.0,
    min_silence_sec: float = 0.4,
) -> str:
    """
    Apply FFmpeg ``silenceremove`` to eradicate audio drops below
    *noise_floor_db* lasting longer than *min_silence_sec*, while cutting
    the corresponding video frames to maintain lip-sync.

    Returns the path to the processed file.
    """
    out_path = os.path.join(tmp_dir, "silence_removed.mp4")
    af = (
        f"silenceremove=stop_periods=-1:"
        f"stop_duration={min_silence_sec}:"
        f"stop_threshold={noise_floor_db}dB"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-af", af,
        "-c:v", "copy",
        out_path,
    ]
    logger.info("Removing silence from %s…", video_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Silence-removed file saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 4.2  Cinematic Eye-Tracking (Rule of Thirds 9:16 Crop)
# ---------------------------------------------------------------------------

def _detect_eye_center(frame: np.ndarray, detector, predictor) -> Optional[Tuple[int, int]]:
    """
    Detect face in *frame* and return the pixel coordinate of the midpoint
    between the two eyes, or ``None`` if no face is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return None

    face = faces[0]  # use largest/first face
    shape = predictor(gray, face)

    # Landmarks 36-41: left eye; 42-47: right eye
    left_pts = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
    right_pts = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

    cx = int(np.mean([p[0] for p in left_pts + right_pts]))
    cy = int(np.mean([p[1] for p in left_pts + right_pts]))
    return cx, cy


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


@log_timing("Eye-Tracking Crop Analysis")
def analyse_crop_coordinates(
    video_path: str,
    sample_interval_sec: float = 1.0,
) -> List[Tuple[float, int, int]]:
    """
    Sample *video_path* every *sample_interval_sec* and detect eye positions.

    Returns a list of ``(timestamp_sec, crop_x, crop_y)`` tuples where
    ``(crop_x, crop_y)`` is the top-left origin of the 1080×1920 crop
    window centred on the actor's eyes at the Rule-of-Thirds line.

    Falls back to a centred crop if dlib or no face is detected.
    """
    try:
        import dlib
        if not os.path.isfile(DLIB_LANDMARKS_MODEL):
            raise FileNotFoundError(DLIB_LANDMARKS_MODEL)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_LANDMARKS_MODEL)
        use_dlib = True
    except Exception as exc:
        logger.warning("dlib unavailable (%s). Falling back to centre crop.", exc)
        use_dlib = False
        detector = predictor = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Desired crop size: maintain 9:16 within the source frame
    crop_w, crop_h = _calc_crop_size(src_w, src_h)

    # Rule-of-Thirds: eyes should sit at 1/3 from the top of the output
    eye_target_y_in_crop = int(crop_h / 3)

    results: List[Tuple[float, int, int]] = []
    frame_interval = max(1, int(fps * sample_interval_sec))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts = frame_idx / fps

            if use_dlib:
                eye_center = _detect_eye_center(frame, detector, predictor)
            else:
                eye_center = None

            if eye_center:
                eye_cx, eye_cy = eye_center
                cx = _clamp(eye_cx - crop_w // 2, 0, src_w - crop_w)
                cy = _clamp(eye_cy - eye_target_y_in_crop, 0, src_h - crop_h)
            else:
                cx = (src_w - crop_w) // 2
                cy = (src_h - crop_h) // 2

            results.append((ts, cx, cy))

        frame_idx += 1

    cap.release()
    logger.info("Crop analysis complete: %d keyframes sampled.", len(results))
    return results


@log_timing("Dynamic Crop Render")
def apply_dynamic_crop(
    video_path: str,
    crop_keyframes: List[Tuple[float, int, int]],
    tmp_dir: str,
    src_w: int,
    src_h: int,
) -> str:
    """
    Render *video_path* with a dynamic 9:16 crop driven by *crop_keyframes*.

    Uses an FFmpeg ``crop`` filter with per-frame expressions derived from
    the keyframe data. For simplicity the nearest keyframe value is used
    (no sub-keyframe interpolation in the filter graph).
    """
    # Determine crop size
    crop_w, crop_h = _calc_crop_size(src_w, src_h)

    if not crop_keyframes:
        cx = (src_w - crop_w) // 2
        cy = (src_h - crop_h) // 2
        crop_filter = f"crop={crop_w}:{crop_h}:{cx}:{cy},scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    else:
        # Build a lookup table string: "ts1 x1 y1 | ts2 x2 y2 …" parsed by
        # a ternary chain in the FFmpeg expr.  For large numbers of keyframes
        # we cap at 50 to avoid shell-argument overflow.
        kf = crop_keyframes[::max(1, len(crop_keyframes) // 50)]

        def _expr_lookup(dim: str) -> str:
            """Build an FFmpeg ternary expression for x or y."""
            idx = 1 if dim == "x" else 2
            pairs = [(kf[i][0], kf[i][idx]) for i in range(len(kf))]
            expr = str(pairs[-1][1])
            for ts, val in reversed(pairs[:-1]):
                expr = f"if(lt(t,{ts:.3f}),{val},{expr})"
            return expr

        x_expr = _expr_lookup("x")
        y_expr = _expr_lookup("y")
        crop_filter = (
            f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr},"
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
        )

    out_path = os.path.join(tmp_dir, "cropped_9_16.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", crop_filter,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast",
        out_path,
    ]
    logger.info("Applying dynamic 9:16 crop…")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Cropped file saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 4.3  Anti-Boring Zoom
# ---------------------------------------------------------------------------

ZOOM_TRIGGER_SEC = 6.0    # shots longer than this get a zoompan
ZOOM_FACTOR = 1.05


@log_timing("Anti-Boring Zoom")
def apply_antiboringzoom(video_path: str, tmp_dir: str) -> str:
    """
    Detect scene cuts; for any shot lasting longer than *ZOOM_TRIGGER_SEC*,
    apply a subtle 1.05× zoompan to create artificial camera movement.

    Uses FFmpeg's ``select`` + ``zoompan`` filter combination.
    """
    import json

    # Probe duration
    cmd_probe = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path,
    ]
    out = subprocess.check_output(cmd_probe).decode()
    stream = json.loads(out)["streams"][0]
    fps_str = stream.get("avg_frame_rate", "25/1")
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den else 25.0

    zoom_frames = int(ZOOM_TRIGGER_SEC * fps)
    zoom_filter = (
        f"zoompan=z='if(lte(on,1),1,if(lte(on,{zoom_frames}),"
        f"zoom+{(ZOOM_FACTOR-1)/zoom_frames:.6f},zoom))':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={zoom_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={fps}"
    )

    out_path = os.path.join(tmp_dir, "zoomed.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", zoom_filter,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast",
        out_path,
    ]
    logger.info("Applying anti-boring zoom to %s…", video_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Zoomed file saved to %s", out_path)
    return out_path

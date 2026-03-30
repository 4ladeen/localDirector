"""
Module 3: Audio Engineering & Analysis

Responsibilities
----------------
3.1  Run Demucs to isolate the Vocal track from each selected clip.
3.2  Apply FFmpeg acompressor + highpass filters for studio-quality EQ.
3.3  Run pyannote.audio speaker diarization on the isolated vocal track.
3.4  Use librosa to detect audio transient peaks (impact timestamps) that
     will trigger the "Camera Shake" VFX in Module 5.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from utils.logger import get_logger, log_timing

logger = get_logger()

# Type alias
Chunk = Tuple[float, float, str]

# Speaker colour map used across Module 3 & 5
SPEAKER_COLORS = {
    "SPEAKER_00": "&H00FFFF&",   # Yellow (ASS hex: BBGGRR)
    "SPEAKER_01": "&H00FF00&",   # Green
    "SPEAKER_02": "&H0000FF&",   # Red
    "SPEAKER_03": "&HFF00FF&",   # Magenta
    "SPEAKER_04": "&HFF8000&",   # Blue
}
DEFAULT_SPEAKER_COLOR = "&HFFFFFF&"   # White fallback

# ---------------------------------------------------------------------------
# 3.1  Stem Separation (Demucs)
# ---------------------------------------------------------------------------

def _is_cuda_related_demucs_failure(stderr: str) -> bool:
    """
    Heuristically detect Demucs failures caused by CUDA/GPU availability issues.
    """
    text = (stderr or "").lower()
    markers = (
        "cuda initialization",
        "nvidia driver",
        "found no nvidia driver",
        "cuda driver",
        "torch.cuda",
        "cuda error",
    )
    return any(marker in text for marker in markers)


@log_timing("Demucs Stem Separation")
def separate_stems(
    audio_path: str,
    tmp_dir: str,
    model: str = "htdemucs",
) -> str:
    """
    Separate *audio_path* into vocal + accompaniment stems using Demucs.

    Returns the path to the isolated ``vocals.wav`` file.
    """
    out_dir = os.path.join(tmp_dir, "demucs_out")
    os.makedirs(out_dir, exist_ok=True)

    base_cmd = [
        sys.executable, "-m", "demucs",
        "--out", out_dir,
        "--name", model,
        "--two-stems", "vocals",
    ]

    def _build_demucs_cmd(device: Optional[str] = None) -> List[str]:
        cmd = list(base_cmd)
        if device:
            cmd.extend(["--device", device])
        cmd.append(audio_path)
        return cmd

    cmd = _build_demucs_cmd()
    logger.info("Running Demucs on %s…", audio_path)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if (
        result.returncode != 0
        and _is_cuda_related_demucs_failure(result.stderr)
    ):
        logger.warning(
            "Demucs failed with a CUDA-related error; retrying on CPU."
        )
        cpu_cmd = _build_demucs_cmd("cpu")
        result = subprocess.run(cpu_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Demucs failed:\n%s", result.stderr)
        raise RuntimeError("Demucs stem separation failed.")

    # Demucs writes to: <out_dir>/<model>/<stem_name>/vocals.wav
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(out_dir, model, base_name, "vocals.wav")

    if not os.path.isfile(vocals_path):
        raise FileNotFoundError(
            f"Expected Demucs output not found at: {vocals_path}"
        )

    logger.info("Vocal stem saved to %s", vocals_path)
    return vocals_path


# ---------------------------------------------------------------------------
# 3.2  Studio EQ / Compression
# ---------------------------------------------------------------------------

@log_timing("Studio EQ Compression")
def apply_eq_compression(vocals_path: str, tmp_dir: str) -> str:
    """
    Apply a podcast-style EQ chain to *vocals_path*:
      - highpass filter at 80 Hz to remove low-end rumble
      - acompressor for dynamic range control
      - loudnorm for broadcast-level normalisation

    Returns the path to the processed vocal WAV.
    """
    out_path = os.path.join(tmp_dir, "vocals_eq.wav")
    af_chain = (
        "highpass=f=80,"
        "acompressor=threshold=-18dB:ratio=4:attack=5:release=50:makeup=6dB,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-af", af_chain,
        out_path,
    ]
    logger.info("Applying studio EQ to %s…", vocals_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("EQ output saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3.3  Speaker Diarization
# ---------------------------------------------------------------------------

@log_timing("Speaker Diarization")
def diarize_speakers(
    vocals_path: str,
    hf_token: str = "",
) -> List[Tuple[float, float, str]]:
    """
    Run pyannote.audio diarization on *vocals_path*.

    Returns a list of ``(start_sec, end_sec, speaker_label)`` tuples.

    *hf_token* is the Hugging Face access token required by pyannote models.
    If empty, the function looks for the ``HF_TOKEN`` environment variable.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.warning("pyannote.audio not installed; skipping diarization.")
        return []

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning(
            "No Hugging Face token found (set HF_TOKEN env var). "
            "Skipping diarization."
        )
        return []

    logger.info("Loading pyannote diarization pipeline…")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    logger.info("Diarizing %s…", vocals_path)
    diarization = pipeline(vocals_path)

    segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    logger.info("Diarization found %d speaker turns.", len(segments))
    return segments


# ---------------------------------------------------------------------------
# 3.4  Impact Detection (librosa)
# ---------------------------------------------------------------------------

IMPACT_THRESHOLD_DB = -20.0   # dB threshold above which a transient counts
IMPACT_WAIT_SEC = 0.5          # minimum gap between consecutive impacts


@log_timing("Impact Detection")
def detect_impacts(vocals_path: str) -> List[float]:
    """
    Use librosa onset detection to locate transient volume peaks in
    *vocals_path* that exceed *IMPACT_THRESHOLD_DB*.

    Returns a list of timestamps (in seconds) that will be used by
    Module 5 to trigger the Camera Shake VFX.
    """
    logger.info("Loading audio for impact detection: %s", vocals_path)
    y, sr = librosa.load(vocals_path, sr=None, mono=True)

    # Compute RMS energy in short frames
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Filter by amplitude threshold
    impacts: List[float] = []
    last_t = -IMPACT_WAIT_SEC
    for t, frame in zip(onset_times, onset_frames):
        if frame < len(rms_db) and rms_db[frame] >= IMPACT_THRESHOLD_DB:
            if t - last_t >= IMPACT_WAIT_SEC:
                impacts.append(float(t))
                last_t = t

    logger.info("Detected %d impact timestamps.", len(impacts))
    return impacts

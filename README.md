# localDirector – Director-Local
**Automated Movie Summarisation & Packaging Engine**

> Ingest a full-length movie → AI-curate the best 20 minutes → render a
> stylised 9:16 vertical mini-documentary with captions, VFX, and metadata.
> 100 % local. 100 % offline. $0 operating cost.

---

## Features

| Module | What it does |
|--------|-------------|
| **1 – Ingestion** | Accepts `.mp4`/`.mkv` + `.srt`; scrapes plot from Wikipedia/IMDb |
| **2 – AI Curation** | Sentence-Transformer semantic scoring, clip selection, cold-open hook, B-roll mapping, Ollama chapter titles |
| **3 – Audio** | Demucs vocal isolation, FFmpeg studio EQ, pyannote speaker diarization, librosa impact detection |
| **4 – Video** | Silence removal, dlib eye-tracking Rule-of-Thirds 9:16 crop, anti-boring zoompan |
| **5 – VFX** | rembg 2.5D parallax, audio-reactive camera shake, diarized ASS captions, waveform/progress-bar/chapter overlays, perfect-loop xfade |
| **6 – Packaging** | Final FFmpeg master render, Ollama SEO metadata, temp-file cleanup |

---

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on `PATH`
- [Ollama](https://ollama.ai/) running locally with your chosen model pulled  
  (e.g. `ollama pull llama3`)
- *(Optional)* dlib shape predictor model for eye-tracking crop:  
  download `shape_predictor_68_face_landmarks.dat` from the
  [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
  and place it in the project root (or set `DLIB_LANDMARKS_MODEL` env var).
- *(Optional)* Hugging Face token for pyannote speaker diarization:  
  set the `HF_TOKEN` environment variable.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/4ladeen/localDirector.git
cd localDirector

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

> **Hardware tip:** On systems with limited VRAM, smaller AI models are
> used automatically. You can also force smaller variants via CLI flags
> (e.g. `--whisper_model tiny`).

---

## Usage

### Basic

```bash
python director.py \
  --video  "movie.mp4" \
  --subs   "movie.srt" \
  --plot_url "https://en.wikipedia.org/wiki/The_Dark_Knight" \
  --target_length 20
```

### Test / Dry-Run (first 30 seconds only)

```bash
python director.py \
  --video movie.mp4 --subs movie.srt \
  --plot_url "https://en.wikipedia.org/wiki/Inception" \
  --test_mode
```

### All Options

```
python director.py --help

options:
  --video            Path to source .mp4 or .mkv file          [required]
  --subs             Path to source .srt subtitle file          [required]
  --plot_url         Wikipedia or IMDb URL for plot scraping
  --target_length    Target output duration in minutes          [default: 20]
  --output           Output file path                           [default: final_summary_ready.mp4]
  --ollama_model     Local Ollama LLM model name                [default: llama3]
  --whisper_model    Whisper model size: tiny|base|small|medium|large  [default: base]
  --sentence_transformer  Sentence-Transformer model            [default: all-MiniLM-L6-v2]
  --bg_music         Path to background music WAV/MP3
  --hf_token         Hugging Face token for pyannote diarization
  --no_parallax      Skip parallax VFX
  --no_shake         Skip camera shake VFX
  --no_zoom          Skip anti-boring zoom
  --test_mode        Render first 30 s only (rapid A/B testing)
```

---

## Outputs

| File | Description |
|------|-------------|
| `final_summary_ready.mp4` | The 20-minute vertical mini-documentary |
| `metadata.txt` | Viral title, description, YouTube chapters, hashtags |
| `process.log` | Per-module timing log for bottleneck analysis |

---

## Project Structure

```
localDirector/
├── director.py              # Main entry point & pipeline orchestrator
├── requirements.txt
├── modules/
│   ├── module1_ingestion.py   # Media ingestion, plot scraping, subtitle chunking
│   ├── module2_curation.py    # Semantic scoring, clip selection, chapters
│   ├── module3_audio.py       # Demucs, EQ, diarization, impact detection
│   ├── module4_video.py       # Silence removal, eye-tracking crop, zoom
│   ├── module5_vfx.py         # Parallax, shake, captions, overlays, loop
│   └── module6_packaging.py   # Final render, metadata, cleanup
└── utils/
    ├── logger.py              # Centralised logging + timing decorator
    └── error_handling.py      # OOM protection, "No Plot" fallback, sub recovery
```

---

## Error Handling & Failsafes

- **OOM protection:** If RAM < 2 GiB free, the working video is automatically
  downscaled to 720p before heavy AI steps.
- **No Plot fallback:** If the Wikipedia/IMDb scrape fails, you are prompted
  to paste a synopsis directly into the terminal.
- **Corrupted subtitles:** If pysrt cannot parse the `.srt`, Whisper
  transcribes the full movie to generate a fresh subtitle file.

---

## Implementation Roadmap

- [x] Phase 1 – The Brain (Modules 1 & 2)
- [x] Phase 2 – The Scalpel (Module 4)
- [x] Phase 3 – The Voice (Module 3)
- [x] Phase 4 – The Polish (Module 5)
- [x] Phase 5 – The Package (Module 6)

---

## License

MIT
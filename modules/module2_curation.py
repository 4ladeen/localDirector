"""
Module 2: AI Curation & Narrative Structuring

Responsibilities
----------------
2.1  Load a local Sentence-Transformer model and embed the plot synopsis.
2.2  Score each subtitle chunk by cosine similarity to the plot embedding;
     select chunks that cumulatively reach the target runtime.
2.3  Identify the "Cold Open" hook: the highest-scoring short burst of
     dialogue (5-10 s) across the entire movie.
2.4  Map discarded footage to active dialogue as contextual B-roll.
2.5  Generate 3-word chapter titles via a local Ollama LLM.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger, log_timing

logger = get_logger()

# Type alias reused from module 1
Chunk = Tuple[float, float, str]

# ---------------------------------------------------------------------------
# 2.1  Semantic Scoring
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_encoder(model_name: str = DEFAULT_MODEL):
    """Load and return a SentenceTransformer encoder."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading Sentence-Transformer model '%s'…", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Model loaded.")
    return model


@log_timing("Semantic Scoring")
def score_chunks(
    chunks: List[Chunk],
    plot_synopsis: str,
    model,
) -> List[Tuple[float, Chunk]]:
    """
    Embed the plot synopsis and every chunk; return a list of
    (cosine_similarity_score, chunk) tuples sorted descending by score.
    """
    from sentence_transformers import util

    texts = [c[2] for c in chunks]
    logger.info("Encoding %d chunks…", len(texts))

    plot_emb = model.encode(plot_synopsis, convert_to_tensor=True)
    chunk_embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    scores = util.cos_sim(plot_emb, chunk_embs)[0].tolist()
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    logger.info("Top chunk score: %.4f", scored[0][0] if scored else 0.0)
    return scored


# ---------------------------------------------------------------------------
# 2.2  Clip Selection
# ---------------------------------------------------------------------------

@log_timing("Clip Selection")
def select_clips(
    scored_chunks: List[Tuple[float, Chunk]],
    target_seconds: float = 20 * 60,
) -> Tuple[List[Chunk], List[Chunk]]:
    """
    Greedily select the highest-scoring chunks until *target_seconds* of
    runtime is accumulated.

    Returns ``(selected, discarded)`` both as lists of Chunk tuples,
    with *selected* sorted chronologically by start time.
    """
    selected: List[Chunk] = []
    discarded: List[Chunk] = []
    total = 0.0

    for score, chunk in scored_chunks:
        start, end, text = chunk
        duration = end - start
        if total < target_seconds:
            selected.append(chunk)
            total += duration
            logger.debug("Selected chunk %.1fs-%.1fs (dur=%.1fs, score=%.4f)", start, end, duration, score)
        else:
            discarded.append(chunk)

    selected.sort(key=lambda c: c[0])
    logger.info(
        "Selected %d chunks (%.1f min); %d discarded.",
        len(selected), total / 60, len(discarded),
    )
    return selected, discarded


# ---------------------------------------------------------------------------
# 2.3  Cold Open Hook Extraction
# ---------------------------------------------------------------------------

COLD_OPEN_MIN_SEC = 5
COLD_OPEN_MAX_SEC = 10


@log_timing("Cold Open Extraction")
def find_cold_open(
    scored_chunks: List[Tuple[float, Chunk]],
    all_chunks: List[Chunk],
    plot_synopsis: str,
    model,
) -> Optional[Chunk]:
    """
    Identify the single highest-scoring 5-to-10 second burst of dialogue
    across the entire movie.

    Strategy: for every subtitle chunk, slide a window of COLD_OPEN_MAX_SEC
    and pick the sub-span with the highest cosine similarity to the plot.
    Returns a Chunk (start, end, text) sized to COLD_OPEN_MAX_SEC.
    """
    from sentence_transformers import util

    plot_emb = model.encode(plot_synopsis, convert_to_tensor=True)

    best_score = -1.0
    best_chunk: Optional[Chunk] = None

    for _, (chunk_start, chunk_end, chunk_text) in scored_chunks:
        duration = chunk_end - chunk_start
        if duration < COLD_OPEN_MIN_SEC:
            continue

        # Slide window
        step = COLD_OPEN_MIN_SEC
        t = chunk_start
        while t + COLD_OPEN_MIN_SEC <= chunk_end:
            win_end = min(t + COLD_OPEN_MAX_SEC, chunk_end)
            # Approximate the text for this window proportionally
            frac_start = (t - chunk_start) / duration
            frac_end = (win_end - chunk_start) / duration
            words = chunk_text.split()
            w0 = int(frac_start * len(words))
            w1 = max(w0 + 1, int(frac_end * len(words)))
            snippet = " ".join(words[w0:w1])

            if snippet:
                emb = model.encode(snippet, convert_to_tensor=True)
                score = float(util.cos_sim(plot_emb, emb)[0][0])
                if score > best_score:
                    best_score = score
                    best_chunk = (t, win_end, snippet)

            t += step

    if best_chunk:
        logger.info(
            "Cold open hook: %.2fs – %.2fs (score=%.4f)",
            best_chunk[0], best_chunk[1], best_score,
        )
    else:
        logger.warning("Could not identify a cold open hook.")

    return best_chunk


# ---------------------------------------------------------------------------
# 2.4  B-Roll Identification
# ---------------------------------------------------------------------------

@log_timing("B-Roll Identification")
def identify_broll(
    selected: List[Chunk],
    discarded: List[Chunk],
    model,
) -> Dict[Chunk, List[Chunk]]:
    """
    For each *selected* chunk, find the semantically closest discarded clips
    to use as B-roll cutaways.

    Returns a mapping ``{selected_chunk: [broll_chunk, …]}``.
    """
    from sentence_transformers import util

    if not discarded:
        logger.info("No discarded clips available for B-roll.")
        return {}

    selected_texts = [c[2] for c in selected]
    discarded_texts = [c[2] for c in discarded]

    sel_embs = model.encode(selected_texts, convert_to_tensor=True)
    dis_embs = model.encode(discarded_texts, convert_to_tensor=True)

    # cosine similarity matrix: (n_selected, n_discarded)
    sim_matrix = util.cos_sim(sel_embs, dis_embs)

    broll_map: Dict[Chunk, List[Chunk]] = {}
    for i, sel_chunk in enumerate(selected):
        row = sim_matrix[i].tolist()
        # Pick top-3 discarded clips
        ranked = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:3]
        broll_map[sel_chunk] = [discarded[j] for j, _ in ranked]

    logger.info("B-roll mapping complete for %d selected clips.", len(selected))
    return broll_map


# ---------------------------------------------------------------------------
# 2.5  Chapter Generation
# ---------------------------------------------------------------------------

@log_timing("Chapter Generation")
def generate_chapters(
    selected: List[Chunk],
    ollama_model: str = "llama3",
) -> List[Tuple[float, str]]:
    """
    Ask the local Ollama LLM to produce a short 3-word chapter title for
    each selected chunk.

    Returns a list of ``(start_seconds, title)`` tuples sorted
    chronologically.
    """
    try:
        import ollama
    except ImportError:
        logger.warning("Ollama not installed; using placeholder chapter titles.")
        return [(c[0], f"Chapter {i+1}") for i, c in enumerate(selected)]

    chapters: List[Tuple[float, str]] = []

    for i, (start, end, text) in enumerate(selected):
        prompt = (
            "You are a documentary editor. Given this transcript excerpt, "
            "reply with ONLY a punchy 3-word chapter title. No punctuation.\n\n"
            f"Excerpt:\n{text[:500]}"
        )
        try:
            resp = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            title = resp["message"]["content"].strip().split("\n")[0]
        except Exception as exc:
            logger.warning("Ollama chapter generation failed for chunk %d: %s", i, exc)
            title = f"Chapter {i + 1}"

        chapters.append((start, title))
        logger.debug("Chapter at %.1fs: '%s'", start, title)

    logger.info("Generated %d chapter titles.", len(chapters))
    return chapters

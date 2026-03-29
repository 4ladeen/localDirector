"""
Module 1: Data Ingestion & Preparation

Responsibilities
----------------
1.1  Accept a local .mp4/.mkv video file and a .srt subtitle file.
1.2  Scrape the movie plot synopsis from Wikipedia or IMDb using
     BeautifulSoup (with a "No Plot" CLI fallback if scraping fails).
1.3  Parse the subtitle file with pysrt and group dialogue into overlapping
     3-to-5 minute chronological "chunks".
"""

import re
import time
from typing import List, Tuple

import pysrt
import requests
from bs4 import BeautifulSoup

from utils.error_handling import prompt_manual_plot
from utils.logger import get_logger, log_timing

logger = get_logger()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A chunk is a tuple of (start_seconds, end_seconds, combined_text)
Chunk = Tuple[float, float, str]

# ---------------------------------------------------------------------------
# 1.1  Media Validation
# ---------------------------------------------------------------------------

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv"}


def validate_inputs(video_path: str, subs_path: str) -> None:
    """
    Raise ValueError if the supplied paths are not valid / unsupported.
    Does *not* open the files – that is left to downstream modules.
    """
    import os

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video format '{ext}'. "
            f"Accepted formats: {', '.join(SUPPORTED_VIDEO_EXTS)}"
        )

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.isfile(subs_path):
        raise FileNotFoundError(f"Subtitle file not found: {subs_path}")

    logger.info("Input validation passed: video=%s  subs=%s", video_path, subs_path)


# ---------------------------------------------------------------------------
# 1.2  Plot Acquisition
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DirectorLocal/1.0; "
        "+https://github.com/localDirector)"
    )
}
_SCRAPE_TIMEOUT = 15  # seconds


@log_timing("Plot Acquisition")
def scrape_plot(url: str) -> str:
    """
    Attempt to scrape a plot synopsis from *url* (Wikipedia or IMDb).

    Wikipedia strategy: look for a section whose heading contains "Plot"
    and return the concatenated paragraph text beneath it.

    IMDb strategy: look for the first <div> with ``data-testid="plot"``
    or a ``<span>`` with class ``sc-plot-block``.

    Falls back to :func:`~utils.error_handling.prompt_manual_plot` if
    scraping fails for any reason.
    """
    from urllib.parse import urlparse

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_SCRAPE_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("HTTP request to '%s' failed: %s", url, exc)
        return prompt_manual_plot()

    soup = BeautifulSoup(resp.text, "lxml")
    hostname = urlparse(url).hostname or ""
    # Extract the registered domain (last two labels, e.g. "wikipedia.org")
    hostname_parts = hostname.lower().split(".")
    registered_domain = ".".join(hostname_parts[-2:]) if len(hostname_parts) >= 2 else hostname

    # --- Wikipedia ---
    if registered_domain == "wikipedia.org":
        plot = _scrape_wikipedia_plot(soup)
        if plot:
            logger.info("Successfully scraped Wikipedia plot (%d chars).", len(plot))
            return plot

    # --- IMDb ---
    if registered_domain == "imdb.com":
        plot = _scrape_imdb_plot(soup)
        if plot:
            logger.info("Successfully scraped IMDb plot (%d chars).", len(plot))
            return plot

    # --- Generic fallback: longest <p> block ---
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text()) > 80)
    if text:
        logger.info("Used generic paragraph scrape (%d chars).", len(text))
        return text

    logger.warning("Could not extract plot from '%s'.", url)
    return prompt_manual_plot()


def _scrape_wikipedia_plot(soup: BeautifulSoup) -> str:
    """Extract the Plot section from a Wikipedia article."""
    plot_heading = soup.find(
        lambda tag: tag.name in ("h2", "h3")
        and re.search(r"\bplot\b", tag.get_text(), re.IGNORECASE)
    )
    if not plot_heading:
        return ""

    paragraphs = []
    for sibling in plot_heading.find_next_siblings():
        if sibling.name in ("h2", "h3"):
            break
        if sibling.name == "p":
            paragraphs.append(sibling.get_text(" ", strip=True))

    return " ".join(paragraphs)


def _scrape_imdb_plot(soup: BeautifulSoup) -> str:
    """Extract the plot synopsis from an IMDb page."""
    # IMDb storyline / synopsis block
    block = soup.find("div", {"data-testid": "storyline-plot-summary"})
    if block:
        return block.get_text(" ", strip=True)

    # Older IMDb layout
    block = soup.find("div", class_="summary_text")
    if block:
        return block.get_text(" ", strip=True)

    return ""


# ---------------------------------------------------------------------------
# 1.3  Subtitle Chunking
# ---------------------------------------------------------------------------

CHUNK_DURATION_MIN = 3 * 60   # 3 minutes in seconds
CHUNK_DURATION_MAX = 5 * 60   # 5 minutes in seconds
CHUNK_OVERLAP_SEC = 30        # 30-second overlap between consecutive chunks


@log_timing("Subtitle Chunking")
def chunk_subtitles(subs_path: str) -> List[Chunk]:
    """
    Parse *subs_path* with pysrt and group subtitle entries into overlapping
    3-to-5 minute chunks.

    Returns a list of (start_sec, end_sec, text) tuples.

    If parsing fails, raises ``ValueError`` so the caller can trigger the
    Whisper recovery path in Module 6.3.
    """
    try:
        subs = pysrt.open(subs_path, encoding="utf-8", error_handling=pysrt.ERROR_LOG)
    except Exception as exc:
        raise ValueError(f"pysrt failed to open '{subs_path}': {exc}") from exc

    if not subs:
        raise ValueError(f"Subtitle file '{subs_path}' contains no entries.")

    chunks: List[Chunk] = []
    chunk_start = _sub_to_seconds(subs[0].start)
    chunk_texts: List[str] = []
    chunk_end = chunk_start

    for sub in subs:
        sub_start = _sub_to_seconds(sub.start)
        sub_end = _sub_to_seconds(sub.end)
        sub_text = sub.text.replace("\n", " ").strip()

        chunk_texts.append(sub_text)
        chunk_end = sub_end

        duration = chunk_end - chunk_start
        if duration >= CHUNK_DURATION_MAX:
            chunks.append((chunk_start, chunk_end, " ".join(chunk_texts)))
            # Overlapping window: step back by overlap amount
            overlap_start = max(chunk_start, chunk_end - CHUNK_OVERLAP_SEC)
            # Collect subs that fall within the overlap window
            overlap_texts = [
                s.text.replace("\n", " ").strip()
                for s in subs
                if _sub_to_seconds(s.start) >= overlap_start
                and _sub_to_seconds(s.end) <= chunk_end
            ]
            chunk_start = overlap_start
            chunk_texts = overlap_texts
            chunk_end = chunk_start

    # Flush remaining subs as the final chunk (may be shorter than 3 minutes)
    if chunk_texts:
        chunks.append((chunk_start, chunk_end, " ".join(chunk_texts)))

    logger.info("Created %d subtitle chunks from '%s'.", len(chunks), subs_path)
    return chunks


def _sub_to_seconds(t: pysrt.SubRipTime) -> float:
    """Convert a pysrt SubRipTime to fractional seconds."""
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0

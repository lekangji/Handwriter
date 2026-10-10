"""
Document layout engine (PRD section 6).

Responsible for turning arbitrary typed text into a paginated set of
handwriting strokes positioned in physical page coordinates (mm), given a
PageTemplate. Text -> handwriting generation is delegated to hw_model; this
module only decides *where* each generated line goes.

Page coordinate convention used throughout this file and gcode.py:
  origin = top-left corner of the page, x increases right, y increases DOWN
  (ordinary document/reading order), units = millimeters.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np

from ai.inference.hw_model import HandwritingModel, GenerationError, offsets_to_coords, sanitize_line, align_baseline

# Empirical calibration constants for the pretrained model's native unit
# space (see hw_model.py). These come from measuring generated samples, not
# from anything in the paper — they only affect word-wrap estimates and the
# mm-per-unit scale; both are corrected/verified downstream (overflow
# shrink, and the printed calibration pattern in calibration.py's use).
AVG_MODEL_UNITS_PER_CHAR = 7.0
REFERENCE_MODEL_UNIT_HEIGHT = 25.0  # ~ typical ascender-to-descender span

# Physical lines fill the usable paper width; chunk_chars alone bounds
# the model's attention. A wide page must not inherit a model character cap.


@dataclass
class Stroke:
    points_mm: np.ndarray  # [N,2] absolute page coordinates (mm)


@dataclass
class LaidOutLine:
    strokes: List[Stroke]
    baseline_y_mm: float


@dataclass
class Page:
    width_mm: float
    height_mm: float
    lines: List[LaidOutLine] = field(default_factory=list)


@dataclass
class Document:
    pages: List[Page]


def _wrap_paragraph(paragraph: str, max_width_mm: float, scale_mm_per_unit: float) -> List[str]:
    """Greedy word-wrap using the average-width estimate. Actual generated
    strokes are measured later and shrunk if this estimate undershoots."""
    max_chars = max(5, int(max_width_mm / (scale_mm_per_unit * AVG_MODEL_UNITS_PER_CHAR)))
    words = paragraph.split(' ')
    lines, cur = [], ''
    for word in words:
        candidate = (cur + ' ' + word).strip()
        if len(candidate) <= max_chars or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or ['']


def wrap_text(text: str, page: "PageTemplate", letter_spacing=1.0) -> List[dict]:
    """Returns a flat list of {'text': str, 'new_paragraph': bool} entries,
    including empty entries for blank lines the user typed."""
    scale = page.handwriting_size_mm / REFERENCE_MODEL_UNIT_HEIGHT * letter_spacing
    out = []
    paragraphs = text.split('\n')
    for pi, para in enumerate(paragraphs):
        para = para.rstrip()
        if para.strip() == '':
            out.append({'text': '', 'new_paragraph': True})
            continue
        wrapped = _wrap_paragraph(para, page.text_width_mm, scale)
        for li, line in enumerate(wrapped):
            out.append({'text': line, 'new_paragraph': (li == 0)})
    return out


def generate_document(text: str, model: HandwritingModel, page: "PageTemplate",
                       seed: int = None, prime_strokes=None, prime_text=None,
                       bias: float = 1.0, progress_cb=None, chunk_chars=20,
                       attempts=3, steps_per_char=50, letter_spacing=1.0,
                       word_spacing=1.0, slant=0.0) -> Document:
    """Full text -> paginated Document of mm-space strokes."""
    if page.text_width_mm <= 0 or page.text_height_mm <= 0 or page.handwriting_size_mm <= 0 or page.line_spacing_mm <= 0:
        raise ValueError("Page size, margins, handwriting size, and line spacing must leave room to write.")
    if not 12 <= chunk_chars <= 48:
        raise ValueError("Generation chunks must be between 12 and 48 characters.")
    if not (0.5 <= letter_spacing <= 1.5 and 0.5 <= word_spacing <= 2 and -20 <= slant <= 20):
        raise ValueError("Letter width, chunk gap, or slant is outside its allowed range.")
    rng = np.random.default_rng(seed)
    scale = page.handwriting_size_mm / REFERENCE_MODEL_UNIT_HEIGHT
    lines_per_page = int((page.text_height_mm - page.handwriting_size_mm / 2) / page.line_spacing_mm)
    if page.lined:
        lines_per_page = int(page.text_height_mm / page.line_spacing_mm)
    if lines_per_page < 1:
        raise ValueError("The page needs room for one line and its descenders.")

    entries = wrap_text(text, page, letter_spacing)

    pages = [Page(width_mm=page.width_mm, height_mm=page.height_mm)]
    line_idx_on_page = 0

    for n, entry in enumerate(entries):
        if line_idx_on_page >= lines_per_page:
            pages.append(Page(width_mm=page.width_mm, height_mm=page.height_mm))
            line_idx_on_page = 0

        baseline_y = page.margin_top_mm + (line_idx_on_page + (0.5 if page.lined else 1)) * page.line_spacing_mm
        laid = LaidOutLine(strokes=[], baseline_y_mm=baseline_y)

        line_text = sanitize_line(entry['text'], valid_chars=set(model.alphabet))
        if line_text.strip():
            line_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
            chunks = []
            current = ''
            for word in line_text.split():
                if len(word) > chunk_chars:
                    raise ValueError(f"Word {word!r} exceeds the chunk limit. Increase chunk length or shorten it.")
                candidate = (current + ' ' + word).strip()
                if len(candidate) > chunk_chars and current:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                chunks.append(current)
            draws, cursor = [], 0.0
            while chunks:
                chunk = chunks.pop(0)
                try:
                    offsets = model.sample(chunk, bias=bias, seed=line_seed,
                                           prime_strokes=prime_strokes, prime_text=prime_text,
                                           attempts=attempts, steps_per_char=steps_per_char)
                except GenerationError:
                    words = chunk.split()
                    if len(words) == 1:
                        raise
                    # Retry only the failed chunk word by word, never the whole page.
                    chunks[:0] = words
                    continue
                chunk_coords = align_baseline(offsets_to_coords(offsets))
                chunk_coords[:, 0] -= chunk_coords[:, 0].min()
                chunk_coords[:, 0] += cursor
                chunk_coords[-1, 2] = 1
                draws.append(chunk_coords)
                cursor = float(chunk_coords[:, 0].max()) + AVG_MODEL_UNITS_PER_CHAR * word_spacing
                line_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
            coords = np.concatenate(draws)
            coords[:, 0] = coords[:, 0] * letter_spacing + coords[:, 1] * np.tan(np.deg2rad(slant))
            coords[:, 0] -= coords[:, 0].min()

            # measure actual width and correct the scale if our wrap estimate undershot
            actual_width_units = coords[:, 0].max() - coords[:, 0].min() if len(coords) else 1.0
            line_scale = scale
            max_width_units = page.text_width_mm / scale
            if actual_width_units > max_width_units and actual_width_units > 0:
                line_scale = page.text_width_mm / actual_width_units  # isotropic shrink to fit

            if page.lined:
                height = np.ptp(coords[:, 1])
                if height > 0:
                    line_scale = min(line_scale, (page.line_spacing_mm - 0.7) / height)
                # Center the ink bounds inside the blank band, leaving room for pen width.
                coords[:, 1] -= (coords[:, 1].min() + coords[:, 1].max()) / 2

            strokes_raw = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
            for s in strokes_raw:
                if len(s) == 0:
                    continue
                if len(s) == 1:
                    s = np.repeat(s, 2, axis=0)  # A dot is a zero-length pen-down stroke.
                pts = np.copy(s[:, :2]).astype(np.float64)
                pts[:, 0] = page.margin_left_mm + pts[:, 0] * line_scale
                pts[:, 1] = baseline_y - pts[:, 1] * line_scale  # model y-up -> page y-down
                laid.strokes.append(Stroke(points_mm=pts))

        pages[-1].lines.append(laid)
        line_idx_on_page += 1

        if progress_cb:
            progress_cb(n + 1, len(entries))

        if entry['text'] == '' and entry['new_paragraph']:
            # blank line the user typed = paragraph break; add extra vertical
            # gap by burning additional ruled lines proportional to the
            # configured paragraph_spacing_mm (rounded to whole ruled lines,
            # since generated text must stay within the ruled bands).
            extra_lines = max(0, round(page.paragraph_spacing_mm / page.line_spacing_mm) - 1)
            line_idx_on_page += extra_lines

    return Document(pages=pages)

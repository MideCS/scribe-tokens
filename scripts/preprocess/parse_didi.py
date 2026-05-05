from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from constants import DATASET, PARSED_DIR, RAW_DIR, TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH
from schemas.ink import DigitalInk
from schemas.parsed import Parsed
from utils.clear_folder import clear_folder

assert DATASET == "didi", "Dataset must be didi"


def _default_raw_path() -> Path:
    """
    Preferred location is `data/didi/raw/diagrams_wo_text_20200131.ndjson` (via RAW_DIR).
    For convenience, we fall back to `datasets/didi/diagrams_wo_text_20200131.ndjson` if present.
    """
    candidate = RAW_DIR / "diagrams_wo_text_20200131.ndjson"
    if candidate.exists():
        return candidate

    fallback = Path(__file__).resolve().parents[2] / "datasets" / "didi" / "diagrams_wo_text_20200131.ndjson"
    return fallback


def _drawing_to_strokes(drawing: Any) -> list[list[tuple[float, float]]]:
    """
    DiDi `drawing` format (per record) is:
      drawing: list[stroke]
      stroke: [xs, ys, ts]
      xs: list[float], ys: list[float], ts: list[int]

    We drop timestamps and return list of (x, y) points per stroke.
    """
    if not isinstance(drawing, list):
        return []

    strokes: list[list[tuple[float, float]]] = []
    for stroke in drawing:
        if not (isinstance(stroke, list) and len(stroke) >= 2):
            continue
        xs, ys = stroke[0], stroke[1]
        if not (isinstance(xs, list) and isinstance(ys, list)):
            continue
        n = min(len(xs), len(ys))
        if n == 0:
            continue
        pts: list[tuple[float, float]] = []
        for i in range(n):
            try:
                pts.append((float(xs[i]), float(ys[i])))
            except (TypeError, ValueError):
                # Skip malformed points; keep the rest of the stroke
                continue
        if pts:
            strokes.append(pts)
    return strokes


def parse_record(record: dict[str, Any]) -> tuple[Parsed, str] | None:
    """
    Convert one DiDi JSON object to a `Parsed` plus its split label.

    DiDi does not have text labels, so we set `text=""` (safe for IdMapper).
    """
    key = record.get("key")
    split = record.get("split")
    drawing = record.get("drawing")

    if not isinstance(key, str) or not key:
        return None
    if split not in {"train", "val", "test"}:
        # Some datasets use "valid" or "validation"; normalize here if needed.
        if split == "valid" or split == "validation":
            split = "val"
        else:
            return None

    strokes = _drawing_to_strokes(drawing)
    if not strokes:
        return None

    writer = str(record.get("label_id", "unknown"))
    ink = DigitalInk.from_coords(strokes)
    parsed = Parsed(id=key, text="", writer=writer, ink=ink)
    return parsed, split


def save_parsed(parsed: Parsed) -> None:
    parsed_path = PARSED_DIR / f"{parsed.id}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(parsed.model_dump(), f, indent=4)


def _write_split(ids: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for id in ids:
            f.write(f"{id}\n")


def main(raw_path: Path | None = None) -> None:
    raw_path = _default_raw_path() if raw_path is None else raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"DiDi NDJSON not found at `{raw_path}`. "
            f"Place it at `{RAW_DIR / 'diagrams_wo_text_20200131.ndjson'}` (recommended) "
            "or `datasets/didi/diagrams_wo_text_20200131.ndjson` (fallback)."
        )

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {line_no}: invalid JSON")
                continue
            if not isinstance(record, dict):
                continue

            parsed_and_split = parse_record(record)
            if parsed_and_split is None:
                continue
            parsed, split = parsed_and_split
            save_parsed(parsed)

            if split == "train":
                train_ids.append(parsed.id)
            elif split == "val":
                val_ids.append(parsed.id)
            elif split == "test":
                test_ids.append(parsed.id)

    _write_split(sorted(set(train_ids)), TRAIN_SPLIT_PATH)
    _write_split(sorted(set(val_ids)), VAL_SPLIT_PATH)
    _write_split(sorted(set(test_ids)), TEST_SPLIT_PATH)

    print(f"Parsed DiDi samples: train={len(set(train_ids))}, val={len(set(val_ids))}, test={len(set(test_ids))}")


if __name__ == "__main__":
    clear_folder(PARSED_DIR)
    main()

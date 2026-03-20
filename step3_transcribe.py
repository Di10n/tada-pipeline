#!/usr/bin/env python3
"""Step 3: Map existing LibriTTS-R .normalized.txt transcripts to segments."""

import csv
from pathlib import Path

from config import SEGMENTS_DIR, LIBRITTS_R


def read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: Path):
    fieldnames = ["segment_id", "audio_path", "duration_seconds", "transcript_text"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] wrote {len(rows)} rows → {path}")


def transcribe_libritts_r():
    """Map .normalized.txt files to segments."""
    seg_dir = SEGMENTS_DIR / LIBRITTS_R
    in_manifest = seg_dir / "manifest.csv"
    out_manifest = seg_dir / "manifest_with_text.csv"

    if out_manifest.exists():
        print("[skip] LibriTTS-R transcript manifest already exists")
        return

    rows = read_manifest(in_manifest)
    result = []
    missing = 0

    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f"  [{i}/{len(rows)}]")
        audio_path = Path(row["audio_path"])
        # Transcript file: same name but .normalized.txt
        txt_path = audio_path.with_suffix(".normalized.txt")
        if not txt_path.exists():
            # Try alternative naming convention
            txt_path = audio_path.parent / (audio_path.stem + ".normalized.txt")

        if txt_path.exists():
            transcript = txt_path.read_text().strip()
        else:
            missing += 1
            continue

        if not transcript:
            continue

        result.append({
            **row,
            "transcript_text": transcript,
        })

    if missing:
        print(f"[warn] {missing} segments missing transcripts")
    write_manifest(result, out_manifest)


if __name__ == "__main__":
    transcribe_libritts_r()
    print("[done] Transcription complete.")

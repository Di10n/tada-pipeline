#!/usr/bin/env python3
"""Step 2: Build manifest from LibriTTS-R segments.

LibriTTS-R is already segmented — just build manifest, skip segments > 30s or < 1s.
"""

import csv
import time
from pathlib import Path

import torchaudio

from config import (
    RAW_DIR,
    SEGMENTS_DIR,
    LIBRITTS_R,
    SAMPLE_RATE,
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
)


def write_manifest(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "audio_path", "duration_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] wrote {len(rows)} rows → {path}")


def process_libritts_r():
    """Build manifest from existing LibriTTS-R wav files. No re-segmentation needed."""
    raw_dir = RAW_DIR / LIBRITTS_R
    seg_dir = SEGMENTS_DIR / LIBRITTS_R
    seg_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = seg_dir / "manifest.csv"
    if manifest_path.exists():
        print("[skip] LibriTTS-R manifest already exists")
        return

    rows = []
    skipped = 0
    all_wavs = sorted(raw_dir.rglob("*.wav"))
    t0 = time.time()
    # LibriTTS-R structure: {split}/LibriTTS_R/{speaker}/{chapter}/{id}.wav
    for i, wav_path in enumerate(all_wavs):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(all_wavs) - i)
            print(f"  [{i}/{len(all_wavs)}] {wav_path.stem}  ETA {eta/60:.1f}m")
        info = torchaudio.info(str(wav_path))
        dur = info.num_frames / info.sample_rate
        if dur > MAX_DURATION_SEC or dur < MIN_DURATION_SEC:
            skipped += 1
            continue
        segment_id = wav_path.stem
        rows.append({
            "segment_id": segment_id,
            "audio_path": str(wav_path),
            "duration_seconds": round(dur, 3),
        })

    print(f"[libritts_r] {len(rows)} segments kept, {skipped} skipped (duration)")
    write_manifest(rows, manifest_path)


if __name__ == "__main__":
    process_libritts_r()
    print("[done] Manifest build complete.")

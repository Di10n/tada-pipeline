#!/usr/bin/env python3
"""Step 8: Split processed data into train/val by speaker.

Reads the existing manifest.json, groups samples by speaker ID (first number
in the LibriTTS-R filename), assigns ~95% of speakers to train and ~5% to val.

Output:
  /root/work/data/processed/manifest_train.json
  /root/work/data/processed/manifest_val.json

Feature .pt files stay in place — only the manifests change.
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from config import DATA_ROOT

PROCESSED_DIR = DATA_ROOT / "processed"
TRAIN_RATIO = 0.95
SEED = 42


def extract_speaker_id(segment_id: str) -> str:
    """Extract speaker ID from a LibriTTS-R segment ID.

    LibriTTS-R filenames follow: {speaker_id}_{chapter_id}_{utterance_id}
    The speaker ID is the first numeric component.
    """
    return segment_id.split("_")[0]


def enrich_entry(entry: dict) -> dict:
    """Add num_tokens and duration_frames from the .pt file if missing."""
    if "num_tokens" in entry and "duration_frames" in entry:
        return entry
    pt_path = Path(entry["feature_path"])
    if pt_path.exists():
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        entry["num_tokens"] = data["token_ids"].shape[0]
        entry["duration_frames"] = int(data["duration_frames"])
    return entry


def split_by_speaker(manifest: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split manifest into train/val by speaker with no speaker overlap."""
    # Enrich manifest entries with token/duration info from .pt files
    print("[split] Enriching manifest with token counts from .pt files...")
    t0 = time.time()
    for i, entry in enumerate(manifest):
        enrich_entry(entry)
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(manifest) - i)
            print(f"  [{i}/{len(manifest)}]  ETA {eta/60:.1f}m")

    # Group samples by speaker
    speaker_samples = defaultdict(list)
    for entry in manifest:
        spk = extract_speaker_id(entry["segment_id"])
        speaker_samples[spk].append(entry)

    speakers = sorted(speaker_samples.keys())
    print(f"[split] {len(manifest)} samples across {len(speakers)} speakers")

    # Shuffle speakers deterministically and split
    rng = random.Random(SEED)
    rng.shuffle(speakers)

    n_train_speakers = int(len(speakers) * TRAIN_RATIO)
    train_speakers = set(speakers[:n_train_speakers])
    val_speakers = set(speakers[n_train_speakers:])

    train_manifest = []
    val_manifest = []

    for spk in speakers:
        if spk in train_speakers:
            train_manifest.extend(speaker_samples[spk])
        else:
            val_manifest.extend(speaker_samples[spk])

    # Compute token totals
    train_tokens = sum(e.get("num_tokens", 0) for e in train_manifest)
    val_tokens = sum(e.get("num_tokens", 0) for e in val_manifest)

    print(f"[split] Train: {len(train_manifest)} samples, {len(train_speakers)} speakers, {train_tokens:,} tokens")
    print(f"[split] Val:   {len(val_manifest)} samples, {len(val_speakers)} speakers, {val_tokens:,} tokens")

    # Verify no overlap
    overlap = train_speakers & val_speakers
    if overlap:
        raise RuntimeError(f"Speaker overlap detected: {overlap}")
    print("[split] Verified: no speaker overlap between train and val")

    return train_manifest, val_manifest


def main():
    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"[error] Manifest not found: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    train_manifest, val_manifest = split_by_speaker(manifest)

    # Write split manifests
    train_path = PROCESSED_DIR / "manifest_train.json"
    val_path = PROCESSED_DIR / "manifest_val.json"

    with open(train_path, "w") as f:
        json.dump(train_manifest, f, indent=2)
    print(f"[write] {train_path}")

    with open(val_path, "w") as f:
        json.dump(val_manifest, f, indent=2)
    print(f"[write] {val_path}")

    # Summary
    train_dur_frames = sum(e.get("duration_frames", 0) for e in train_manifest)
    val_dur_frames = sum(e.get("duration_frames", 0) for e in val_manifest)
    frame_rate = 50  # TADA encoder frame rate
    print(f"\n[summary]")
    print(f"  Train: ~{train_dur_frames / frame_rate / 3600:.0f} hours")
    print(f"  Val:   ~{val_dur_frames / frame_rate / 3600:.0f} hours")

    print("[done] Train/val split complete.")


if __name__ == "__main__":
    main()

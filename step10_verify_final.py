#!/usr/bin/env python3
"""Step 10: Final verification of all splits after post-pipeline processing.

Checks:
1. All keys exist in .pt files: token_ids, encoder_features, positions,
   f_before, f_after, speaker_embedding
2. Shapes: encoder_features is (L, 512), speaker_embedding is (192,)
3. Speaker embedding quality: same-speaker cosine similarity > 0.8,
   different-speaker similarity < 0.4
4. Train/val split has no speaker overlap
5. Summary statistics per split
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from config import DATA_ROOT

PROCESSED_DIR = DATA_ROOT / "processed"
TEST_DIR = DATA_ROOT / "test"

REQUIRED_KEYS = ["token_ids", "encoder_features", "positions", "f_before", "f_after", "speaker_embedding"]
SAMPLES_PER_SPLIT = 5
SEED = 123


def load_split(name: str) -> list[dict]:
    """Load a manifest by split name."""
    if name == "test":
        path = TEST_DIR / "manifest.json"
    elif name == "train":
        path = PROCESSED_DIR / "manifest_train.json"
    elif name == "val":
        path = PROCESSED_DIR / "manifest_val.json"
    else:
        raise ValueError(f"Unknown split: {name}")

    if not path.exists():
        print(f"[warn] {path} not found")
        return []

    with open(path) as f:
        return json.load(f)


def extract_speaker_id(segment_id: str) -> str:
    return segment_id.split("_")[0]


def check_keys_and_shapes(manifest: list[dict], split_name: str, n: int = 5) -> bool:
    """Check 1 & 2: Verify all keys exist and shapes are correct."""
    print(f"\n=== Keys & Shapes: {split_name} ===")
    rng = random.Random(SEED)
    samples = rng.sample(manifest, min(n, len(manifest)))
    all_ok = True

    for entry in samples:
        sid = entry["segment_id"]
        path = Path(entry["feature_path"])
        if not path.exists():
            print(f"  [FAIL] {sid}: feature file not found")
            all_ok = False
            continue

        data = torch.load(path, map_location="cpu", weights_only=True)

        # Check required keys
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            print(f"  [FAIL] {sid}: missing keys {missing}")
            all_ok = False
            continue

        L = data["token_ids"].shape[0]
        ef = data["encoder_features"]
        se = data["speaker_embedding"]

        errors = []
        if ef.shape != (L, 512):
            errors.append(f"encoder_features {ef.shape} != ({L}, 512)")
        if se.shape != (192,):
            errors.append(f"speaker_embedding {se.shape} != (192,)")

        if errors:
            print(f"  [FAIL] {sid}: {'; '.join(errors)}")
            all_ok = False
        else:
            print(f"  [OK]   {sid}: L={L}, shapes correct")

    return all_ok


def check_speaker_embeddings(manifest: list[dict], split_name: str) -> bool:
    """Check 3: Verify speaker embedding quality via cosine similarity."""
    print(f"\n=== Speaker Embedding Quality: {split_name} ===")

    # Group by speaker, pick speakers with multiple samples
    speaker_samples = defaultdict(list)
    for entry in manifest:
        spk = extract_speaker_id(entry["segment_id"])
        speaker_samples[spk].append(entry)

    multi_sample_speakers = [s for s, samples in speaker_samples.items() if len(samples) >= 2]
    if len(multi_sample_speakers) < 2:
        print("  [skip] Not enough multi-sample speakers for comparison")
        return True

    rng = random.Random(SEED)

    # Same-speaker similarity
    spk = rng.choice(multi_sample_speakers)
    pair = rng.sample(speaker_samples[spk], 2)
    emb1 = torch.load(Path(pair[0]["feature_path"]), map_location="cpu", weights_only=True)["speaker_embedding"]
    emb2 = torch.load(Path(pair[1]["feature_path"]), map_location="cpu", weights_only=True)["speaker_embedding"]
    same_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    # Different-speaker similarity
    two_spks = rng.sample(multi_sample_speakers, 2)
    emb_a = torch.load(
        Path(rng.choice(speaker_samples[two_spks[0]])["feature_path"]),
        map_location="cpu", weights_only=True
    )["speaker_embedding"]
    emb_b = torch.load(
        Path(rng.choice(speaker_samples[two_spks[1]])["feature_path"]),
        map_location="cpu", weights_only=True
    )["speaker_embedding"]
    diff_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

    ok = True
    if same_sim > 0.8:
        print(f"  [OK]   Same-speaker (spk={spk}): cosine_sim={same_sim:.3f} (>0.8)")
    else:
        print(f"  [WARN] Same-speaker (spk={spk}): cosine_sim={same_sim:.3f} (expected >0.8)")
        ok = False

    if diff_sim < 0.4:
        print(f"  [OK]   Diff-speaker ({two_spks[0]} vs {two_spks[1]}): cosine_sim={diff_sim:.3f} (<0.4)")
    else:
        print(f"  [WARN] Diff-speaker ({two_spks[0]} vs {two_spks[1]}): cosine_sim={diff_sim:.3f} (expected <0.4)")
        ok = False

    return ok


def check_speaker_disjoint(train: list[dict], val: list[dict]) -> bool:
    """Check 4: No speaker overlap between train and val."""
    print("\n=== Speaker Disjointness (train vs val) ===")
    train_speakers = {extract_speaker_id(e["segment_id"]) for e in train}
    val_speakers = {extract_speaker_id(e["segment_id"]) for e in val}
    overlap = train_speakers & val_speakers

    if overlap:
        print(f"  [FAIL] {len(overlap)} speakers in both train and val: {list(overlap)[:10]}")
        return False
    else:
        print(f"  [OK]   Train: {len(train_speakers)} speakers, Val: {len(val_speakers)} speakers, overlap: 0")
        return True


def print_summary(manifest: list[dict], split_name: str):
    """Check 5: Print summary statistics."""
    print(f"\n=== Summary: {split_name} ===")

    if not manifest:
        print("  (empty)")
        return

    num_samples = len(manifest)
    speakers = {extract_speaker_id(e["segment_id"]) for e in manifest}

    # Load token counts and durations from .pt files if not in manifest
    total_tokens = 0
    durations_sec = []
    frame_rate = 50
    for entry in manifest:
        pt_path = Path(entry["feature_path"])
        if pt_path.exists():
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            total_tokens += data["token_ids"].shape[0]
            durations_sec.append(int(data["duration_frames"]) / frame_rate)
        else:
            total_tokens += entry.get("num_tokens", 0)
            durations_sec.append(entry.get("duration_frames", 0) / frame_rate)
    durations_sec.sort()

    print(f"  Samples:    {num_samples:,}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Speakers:   {len(speakers):,}")
    print(f"  Duration:   min={durations_sec[0]:.1f}s, "
          f"median={durations_sec[len(durations_sec)//2]:.1f}s, "
          f"max={durations_sec[-1]:.1f}s")
    total_hours = sum(durations_sec) / 3600
    print(f"  Total:      {total_hours:.1f} hours")


def main():
    print("=" * 60)
    print("Final Verification")
    print("=" * 60)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    results = {}

    for split_name, manifest in [("train", train), ("val", val), ("test", test)]:
        if not manifest:
            continue
        results[f"{split_name}_keys"] = check_keys_and_shapes(manifest, split_name, SAMPLES_PER_SPLIT)
        results[f"{split_name}_embed"] = check_speaker_embeddings(manifest, split_name)

    if train and val:
        results["disjoint"] = check_speaker_disjoint(train, val)

    for split_name, manifest in [("train", train), ("val", val), ("test", test)]:
        print_summary(manifest, split_name)

    # Final verdict
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    all_pass = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()

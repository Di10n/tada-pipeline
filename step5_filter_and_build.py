#!/usr/bin/env python3
"""Steps 8-9: Quality filtering and final dataset assembly.

Filters out bad samples and builds the final manifest + optional WebDataset shards.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import torch

from config import (
    FEATURES_DIR,
    PROCESSED_DIR,
    SHARDS_DIR,
    LIBRITTS_R,
    MAX_CONSECUTIVE_FRAMES,
    MAX_POSITION_GAP_FRAMES,
    MIN_DURATION_SEC,
    MAX_DURATION_SEC,
    FRAME_RATE,
)


def check_quality(sample: dict) -> str | None:
    """Return a rejection reason string, or None if the sample passes."""
    positions = sample["positions"]
    token_ids = sample["token_ids"]
    duration_frames = sample["duration_frames"]
    L = positions.shape[0]

    # Empty transcript
    if L == 0:
        return "empty_tokens"

    # Duration bounds
    dur_sec = duration_frames / FRAME_RATE
    if dur_sec < MIN_DURATION_SEC:
        return "too_short"
    if dur_sec > MAX_DURATION_SEC:
        return "too_long"

    # Check for consecutive positions spanning > MAX_CONSECUTIVE_FRAMES
    # This detects hallucinated text (many tokens crammed into few frames)
    if L > 1:
        diffs = positions[1:] - positions[:-1]
        # Count runs of zero diffs (tokens at same frame)
        run_len = 1
        for d in diffs:
            if d.item() == 0:
                run_len += 1
                if run_len > MAX_CONSECUTIVE_FRAMES:
                    return "consecutive_frames"
            else:
                run_len = 1

    # Check for large gaps (missing text / non-speech)
    if L > 1:
        gaps = positions[1:] - positions[:-1]
        if gaps.max().item() > MAX_POSITION_GAP_FRAMES:
            return "large_gap"

    return None


def filter_and_build(dataset_names: list[str]):
    """Filter features and build final manifest."""
    final_features_dir = PROCESSED_DIR / "features"
    final_features_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    stats = Counter()

    for ds_name in dataset_names:
        feat_dir = FEATURES_DIR / ds_name
        if not feat_dir.exists():
            print(f"[skip] No features for {ds_name}")
            continue

        pt_files = sorted(feat_dir.glob("*.pt"))
        print(f"[filter] {ds_name}: {len(pt_files)} feature files")

        for pt_idx, pt_path in enumerate(pt_files):
            if pt_idx % 100 == 0:
                print(f"  [{pt_idx}/{len(pt_files)}]")
            segment_id = pt_path.stem
            if segment_id.startswith("_"):
                continue  # skip metadata files

            try:
                sample = torch.load(pt_path, map_location="cpu", weights_only=True)
            except Exception as e:
                stats["load_error"] += 1
                continue

            reason = check_quality(sample)
            if reason:
                stats[reason] += 1
                continue

            # Copy/link to final features dir (avoid duplicating if same dir)
            dst = final_features_dir / f"{segment_id}.pt"
            if not dst.exists():
                # Use symlink to save space; replace with shutil.copy2 if needed
                try:
                    dst.symlink_to(pt_path.resolve())
                except OSError:
                    shutil.copy2(pt_path, dst)

            manifest.append({
                "segment_id": segment_id,
                "dataset": ds_name,
                "num_tokens": sample["token_ids"].shape[0],
                "duration_frames": int(sample["duration_frames"]),
                "feature_path": str(dst),
            })
            stats["kept"] += 1

    # Write manifest
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print stats
    total = sum(stats.values())
    print(f"\n[filter] Results:")
    print(f"  Total processed: {total}")
    for reason, count in stats.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    print(f"\n[manifest] {len(manifest)} samples → {manifest_path}")
    return manifest


def build_webdataset_shards(manifest: list[dict], samples_per_shard: int = 5000):
    """Optionally pack into WebDataset tar shards for streaming training."""
    try:
        import webdataset as wds
    except ImportError:
        print("[skip] webdataset not installed, skipping shard creation")
        print("  Install with: pip install webdataset")
        return

    import io
    import tarfile

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    sample_count = 0
    tar = None

    for entry in manifest:
        if sample_count % samples_per_shard == 0:
            if tar is not None:
                tar.close()
            shard_path = SHARDS_DIR / f"train-{shard_idx:06d}.tar"
            tar = tarfile.open(shard_path, "w")
            print(f"[shard] Writing {shard_path}")
            shard_idx += 1

        segment_id = entry["segment_id"]
        pt_path = entry["feature_path"]

        # Read the .pt file as bytes
        pt_bytes = Path(pt_path).resolve().read_bytes()

        # Add to tar
        info = tarfile.TarInfo(name=f"{segment_id}.pt")
        info.size = len(pt_bytes)
        tar.addfile(info, io.BytesIO(pt_bytes))

        # Also add metadata as json
        meta = {k: v for k, v in entry.items() if k != "feature_path"}
        meta_bytes = json.dumps(meta).encode()
        info = tarfile.TarInfo(name=f"{segment_id}.json")
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

        sample_count += 1

    if tar is not None:
        tar.close()

    print(f"[shards] {shard_idx} shards written to {SHARDS_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", action="store_true", help="Also build WebDataset shards")
    args = parser.parse_args()

    manifest = filter_and_build([LIBRITTS_R])

    if args.shards:
        build_webdataset_shards(manifest)

    print("[done] Dataset build complete.")

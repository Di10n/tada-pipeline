#!/usr/bin/env python3
"""Step 7: Download LibriTTS-R test-clean and preprocess through the TADA pipeline.

Downloads test-clean from OpenSLR, reads provided normalized transcripts,
then runs the same TADA encoder pipeline (tokenize, align, extract features,
compute f_before/f_after, quality filter) used for training data.

Output:
  /workspace/data/test/
    manifest.json
    features/{segment_id}.pt
"""

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tada"))

from tada.modules.encoder import Encoder

from config import (
    DATA_ROOT,
    RAW_DIR,
    SAMPLE_RATE,
    FRAME_RATE,
    BATCH_SIZE,
    TADA_CODEC_REPO,
    ENCODER_SUBFOLDER,
    MAX_CONSECUTIVE_FRAMES,
    MAX_POSITION_GAP_FRAMES,
    MIN_DURATION_SEC,
    MAX_DURATION_SEC,
)
from step4_extract_features import compute_frame_gaps
from step5_filter_and_build import check_quality

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR = DATA_ROOT / "test"
TEST_FEATURES_DIR = TEST_DIR / "features"

LIBRITTS_R_TEST_URL = "https://www.openslr.org/resources/141/test_clean.tar.gz"


def download_test_clean():
    """Download and extract LibriTTS-R test-clean."""
    out_dir = RAW_DIR / "libritts_r_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if any(out_dir.rglob("*.wav")):
        print("[skip] test-clean already extracted")
        return out_dir

    tar_path = RAW_DIR / "test_clean.tar.gz"
    print(f"[download] {LIBRITTS_R_TEST_URL}")
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", LIBRITTS_R_TEST_URL, "-O", str(tar_path)],
        check=True,
    )

    print(f"[extract] test_clean.tar.gz → {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)

    tar_path.unlink()
    return out_dir


def discover_test_samples(raw_dir: Path) -> list[dict]:
    """Find all wav files in test-clean and pair with normalized transcripts."""
    samples = []
    skipped = 0

    for wav_path in sorted(raw_dir.rglob("*.wav")):
        # Get duration
        info = torchaudio.info(str(wav_path))
        dur = info.num_frames / info.sample_rate
        if dur > MAX_DURATION_SEC or dur < MIN_DURATION_SEC:
            skipped += 1
            continue

        # Find normalized transcript (same stem + .normalized.txt)
        txt_path = wav_path.with_suffix(".normalized.txt")
        if not txt_path.exists():
            print(f"  [warn] No transcript for {wav_path.name}, skipping")
            skipped += 1
            continue

        transcript = txt_path.read_text().strip()
        if not transcript:
            skipped += 1
            continue

        samples.append({
            "segment_id": wav_path.stem,
            "audio_path": str(wav_path),
            "transcript": transcript,
            "duration_seconds": round(dur, 3),
        })

    print(f"[discover] {len(samples)} test samples found, {skipped} skipped")
    return samples


def process_test_set(samples: list[dict], encoder: Encoder, device: str):
    """Run TADA encoder on test samples and save features."""
    TEST_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    processed_log = TEST_FEATURES_DIR / "_processed.json"
    already_done = set()
    if processed_log.exists():
        already_done = set(json.loads(processed_log.read_text()))

    print(f"[extract] test-clean: {len(samples)} samples, {len(already_done)} already done")

    manifest = []
    newly_done = []
    rejected = 0

    for i, sample in enumerate(samples):
        segment_id = sample["segment_id"]

        if segment_id in already_done:
            # Still add to manifest if feature file exists
            pt_path = TEST_FEATURES_DIR / f"{segment_id}.pt"
            if pt_path.exists():
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                manifest.append({
                    "segment_id": segment_id,
                    "dataset": "libritts_r_test_clean",
                    "num_tokens": data["token_ids"].shape[0],
                    "duration_frames": int(data["duration_frames"]),
                    "feature_path": str(pt_path),
                })
            continue

        if i % 100 == 0:
            print(f"  [{i}/{len(samples)}] {segment_id}")

        try:
            waveform, sr = torchaudio.load(sample["audio_path"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.to(device)

            with torch.no_grad():
                enc_out = encoder(
                    waveform,
                    text=[sample["transcript"]],
                    sample_rate=sr,
                    sample=False,
                )

            token_ids = enc_out.text_tokens[0].cpu()
            token_positions = enc_out.token_positions[0].cpu()
            token_values = enc_out.token_values[0].cpu()

            if enc_out.text_tokens_len is not None:
                tlen = enc_out.text_tokens_len[0].item()
                token_ids = token_ids[:tlen]
                token_positions = token_positions[:tlen]
                token_values = token_values[:tlen]

            dur_sec = waveform.shape[1] / sr
            total_frames = int(dur_sec * FRAME_RATE)
            f_before, f_after = compute_frame_gaps(token_positions, total_frames)

            sample_dict = {
                "token_ids": token_ids,
                "encoder_features": token_values,
                "positions": token_positions,
                "f_before": f_before,
                "f_after": f_after,
                "duration_frames": total_frames,
            }

            # Quality filter
            reason = check_quality(sample_dict)
            if reason:
                rejected += 1
                newly_done.append(segment_id)
                continue

            pt_path = TEST_FEATURES_DIR / f"{segment_id}.pt"
            torch.save(sample_dict, pt_path)
            newly_done.append(segment_id)

            manifest.append({
                "segment_id": segment_id,
                "dataset": "libritts_r_test_clean",
                "num_tokens": token_ids.shape[0],
                "duration_frames": total_frames,
                "feature_path": str(pt_path),
            })

        except Exception as e:
            print(f"  [error] {segment_id}: {e}")
            continue

        if len(newly_done) % 200 == 0 and newly_done:
            all_done = list(already_done | set(newly_done))
            processed_log.write_text(json.dumps(all_done))

    # Final checkpoint
    all_done = list(already_done | set(newly_done))
    processed_log.write_text(json.dumps(all_done))

    # Write manifest
    manifest_path = TEST_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[test] {len(manifest)} samples kept, {rejected} rejected by quality filter")
    print(f"[test] Manifest → {manifest_path}")
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Step 1: Download
    raw_dir = download_test_clean()

    # Step 2: Discover samples with transcripts
    samples = discover_test_samples(raw_dir)

    # Step 3: Load encoder and process
    print(f"[init] Device: {args.device}")
    print("[init] Loading TADA encoder + aligner...")
    encoder = Encoder.from_pretrained(TADA_CODEC_REPO, subfolder=ENCODER_SUBFOLDER).to(args.device)
    encoder.eval()

    manifest = process_test_set(samples, encoder, args.device)
    print(f"[done] Test set processing complete: {len(manifest)} samples")


if __name__ == "__main__":
    main()

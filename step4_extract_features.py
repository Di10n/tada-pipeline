#!/usr/bin/env python3
"""Steps 4-7: Load TADA models, tokenize, extract alignments & encoder features.

Uses the TADA Encoder (which internally runs the Aligner) to extract:
  - token_ids: text token IDs from Llama 3.2 tokenizer
  - token_positions: frame-level alignment positions (50Hz)
  - token_values: encoder acoustic features (L, 512)
  - f_before / f_after: blank frame counts around each token
"""

import csv
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer

# Add TADA repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tada"))

from tada.modules.encoder import Encoder

from config import (
    SEGMENTS_DIR,
    FEATURES_DIR,
    LIBRITTS_R,
    TADA_CODEC_REPO,
    ENCODER_SUBFOLDER,
    TOKENIZER_NAME,
    SAMPLE_RATE,
    BATCH_SIZE,
    FRAME_RATE,
)


def read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_frame_gaps(positions: torch.Tensor, total_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute f_before and f_after for each token from positions.

    positions: (L,) 0-indexed frame positions for each token.
    total_frames: total number of encoder frames in the audio.

    f_before[i] = positions[i] - positions[i-1] - 1   (for i>0, else positions[0])
    f_after[i]  = positions[i+1] - positions[i] - 1   (for i<L-1, else total_frames - 1 - positions[-1])
    """
    L = positions.shape[0]
    f_before = torch.zeros(L, dtype=torch.long)
    f_after = torch.zeros(L, dtype=torch.long)

    # f_before[0] = number of blank frames before the first token
    f_before[0] = positions[0]
    for i in range(1, L):
        f_before[i] = positions[i] - positions[i - 1] - 1

    # f_after[i] = number of blank frames after token i
    for i in range(L - 1):
        f_after[i] = positions[i + 1] - positions[i] - 1
    f_after[L - 1] = total_frames - 1 - positions[L - 1]

    # Clamp to non-negative (should already be, but safety)
    f_before = f_before.clamp(min=0)
    f_after = f_after.clamp(min=0)

    return f_before, f_after


def process_dataset(dataset_name: str, encoder: Encoder, tokenizer, device: str):
    """Extract features for all segments in a dataset."""
    seg_dir = SEGMENTS_DIR / dataset_name
    manifest_path = seg_dir / "manifest_with_text.csv"

    if not manifest_path.exists():
        print(f"[skip] No manifest_with_text.csv for {dataset_name}")
        return

    rows = read_manifest(manifest_path)
    out_dir = FEATURES_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_log = out_dir / "_processed.json"
    already_done = set()
    if processed_log.exists():
        already_done = set(json.loads(processed_log.read_text()))

    print(f"[extract] {dataset_name}: {len(rows)} segments, {len(already_done)} already done")

    newly_done = []
    t0 = time.time()
    processed_count = 0

    for i, row in enumerate(rows):
        segment_id = row["segment_id"]
        if segment_id in already_done:
            continue

        audio_path = row["audio_path"]
        transcript = row["transcript_text"]

        processed_count += 1
        if processed_count % 100 == 0:
            elapsed = time.time() - t0
            remaining = len(rows) - len(already_done) - processed_count
            eta = elapsed / processed_count * remaining
            print(f"  [{i}/{len(rows)}] {segment_id}  ETA {eta/60:.1f}m")

        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.to(device)

            # Run encoder (handles alignment + feature extraction internally)
            with torch.no_grad():
                enc_out = encoder(
                    waveform,
                    text=[transcript],
                    sample_rate=sr,
                    sample=False,  # No noise during extraction — save deterministic mean
                )

            # Extract outputs (squeeze batch dim)
            token_ids = enc_out.text_tokens[0].cpu()  # (L,)
            token_positions = enc_out.token_positions[0].cpu()  # (L,)
            token_values = enc_out.token_values[0].cpu()  # (L, 512)
            L = token_ids.shape[0]

            # Trim to actual token length
            if enc_out.text_tokens_len is not None:
                tlen = enc_out.text_tokens_len[0].item()
                token_ids = token_ids[:tlen]
                token_positions = token_positions[:tlen]
                token_values = token_values[:tlen]
                L = tlen

            # Compute total frames from audio duration
            dur_sec = waveform.shape[1] / sr
            total_frames = int(dur_sec * FRAME_RATE)

            # Compute f_before, f_after
            f_before, f_after = compute_frame_gaps(token_positions, total_frames)

            # Duration in frames for the full audio
            duration_frames = total_frames

            # Save as .pt
            sample_dict = {
                "token_ids": token_ids,                # (L,) int
                "encoder_features": token_values,      # (L, 512) float
                "positions": token_positions,          # (L,) int
                "f_before": f_before,                  # (L,) int
                "f_after": f_after,                    # (L,) int
                "duration_frames": duration_frames,    # int
            }
            torch.save(sample_dict, out_dir / f"{segment_id}.pt")
            newly_done.append(segment_id)

        except Exception as e:
            print(f"  [error] {segment_id}: {e}")
            continue

        # Checkpoint progress every 500 segments
        if len(newly_done) % 500 == 0 and newly_done:
            all_done = list(already_done | set(newly_done))
            processed_log.write_text(json.dumps(all_done))

    # Final checkpoint
    all_done = list(already_done | set(newly_done))
    processed_log.write_text(json.dumps(all_done))
    print(f"[extract] {dataset_name}: {len(newly_done)} new, {len(all_done)} total done")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"[init] Device: {device}")

    # Load TADA encoder (includes aligner)
    print("[init] Loading TADA encoder + aligner...")
    encoder = Encoder.from_pretrained(TADA_CODEC_REPO, subfolder=ENCODER_SUBFOLDER).to(device)
    encoder.eval()

    # Load tokenizer (for verification only — encoder tokenizes internally)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    process_dataset(LIBRITTS_R, encoder, tokenizer, device)

    print("[done] Feature extraction complete.")


if __name__ == "__main__":
    main()

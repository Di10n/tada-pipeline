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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer

# Add TADA repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tada"))

from tada.modules.encoder import Encoder
from tada.utils.text import normalize_text

from config import (
    SEGMENTS_DIR,
    FEATURES_DIR,
    LIBRITTS_R,
    TADA_CODEC_REPO,
    ENCODER_SUBFOLDER,
    TOKENIZER_NAME,
    SAMPLE_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
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


def _load_audio(row):
    """Load and preprocess a single audio file. Returns (row, waveform, sr) or (row, None, error)."""
    try:
        waveform, sr = torchaudio.load(row["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return row, waveform.squeeze(0), sr  # (samples,)
    except Exception as e:
        return row, None, str(e)


def _load_batch_audio(batch_rows):
    """Load audio for a batch using parallel threads."""
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        return list(pool.map(_load_audio, batch_rows))


def _extract_cpu(enc_out, j, audio_len_samples, sr):
    """Extract features for sample j from batched encoder output, returning CPU tensors."""
    token_ids = enc_out.text_tokens[j].cpu()
    token_positions = enc_out.token_positions[j].cpu()
    token_values = enc_out.token_values[j].cpu()

    if enc_out.text_tokens_len is not None:
        tlen = enc_out.text_tokens_len[j].item()
        token_ids = token_ids[:tlen]
        token_positions = token_positions[:tlen]
        token_values = token_values[:tlen]

    total_frames = int(audio_len_samples / sr * FRAME_RATE)
    f_before, f_after = compute_frame_gaps(token_positions, total_frames)

    return {
        "token_ids": token_ids,
        "encoder_features": token_values,
        "positions": token_positions,
        "f_before": f_before,
        "f_after": f_after,
        "duration_frames": total_frames,
    }


def _save_pt(sample_dict, path):
    """Save a sample dict to disk (intended for background thread)."""
    torch.save(sample_dict, path)


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

    # Filter to unprocessed rows and sort by duration to minimize padding waste
    todo_rows = [row for row in rows if row["segment_id"] not in already_done]
    todo_rows.sort(key=lambda r: float(r.get("duration_seconds", 0)))
    total_todo = len(todo_rows)

    print(f"[extract] {dataset_name}: {len(rows)} segments, {len(already_done)} already done, {total_todo} to process")

    if total_todo == 0:
        return

    newly_done = []
    t0 = time.time()
    processed_count = 0
    last_progress = 0
    last_checkpoint = 0
    save_futures = []

    # Split into batches
    batches = [todo_rows[i:i + BATCH_SIZE] for i in range(0, total_todo, BATCH_SIZE)]

    # Prefetch: load first batch while we set up
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    save_pool = ThreadPoolExecutor(max_workers=4)
    next_loaded = prefetch_pool.submit(_load_batch_audio, batches[0])

    for batch_idx, batch_rows in enumerate(batches):
        # Wait for current batch audio (already loading in background)
        loaded = next_loaded.result()

        # Start loading next batch while GPU processes this one
        if batch_idx + 1 < len(batches):
            next_loaded = prefetch_pool.submit(_load_batch_audio, batches[batch_idx + 1])

        # Separate successes and failures
        valid = []
        for row, wav, sr_or_err in loaded:
            if wav is not None:
                valid.append((row, wav, sr_or_err))
            else:
                print(f"  [error] {row['segment_id']}: {sr_or_err}")

        if not valid:
            processed_count += len(batch_rows)
            continue

        # Build padded batch
        waveforms = [wav for _, wav, _ in valid]
        audio_lengths = torch.tensor([w.shape[0] for w in waveforms])
        texts = [row["transcript_text"] for row, _, _ in valid]
        sr = valid[0][2]  # LibriTTS-R is all 24kHz

        max_len = audio_lengths.max().item()
        padded = torch.zeros(len(waveforms), max_len)
        for j, w in enumerate(waveforms):
            padded[j, :w.shape[0]] = w

        padded = padded.to(device)
        audio_lengths_dev = audio_lengths.to(device).unsqueeze(1)  # (B,1) for aligner broadcast

        # Pre-tokenize: the encoder's internal pad_sequence is buggy for
        # batch>1 (2D tensors with varying dim 1). Bypass by passing
        # text_tokens and text_token_len directly.
        normalized = [normalize_text(t) for t in texts]
        tok = encoder.aligner.tokenizer
        token_seqs = [
            tok.encode(t, add_special_tokens=False, return_tensors="pt").squeeze(0)
            for t in normalized
        ]
        text_token_len = torch.tensor([t.shape[0] for t in token_seqs], device=device)
        text_tokens = torch.nn.utils.rnn.pad_sequence(
            token_seqs, batch_first=True, padding_value=tok.eos_token_id,
        ).to(device)

        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled="cuda" in device):
                enc_out = encoder(
                    padded,
                    text_tokens=text_tokens,
                    text_token_len=text_token_len,
                    audio_length=audio_lengths_dev,
                    sample_rate=sr,
                    sample=False,
                )

            for j, (row, wav, _) in enumerate(valid):
                try:
                    sample_dict = _extract_cpu(enc_out, j, audio_lengths[j].item(), sr)
                    save_futures.append(save_pool.submit(
                        _save_pt, sample_dict, out_dir / f"{row['segment_id']}.pt"
                    ))
                    newly_done.append(row["segment_id"])
                except Exception as e:
                    print(f"  [error] {row['segment_id']}: {e}")

        except Exception as e:
            print(f"  [warn] Batch failed ({e}), falling back to sequential")
            for row, wav, sr_i in valid:
                try:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled="cuda" in device):
                        enc_out = encoder(
                            wav.unsqueeze(0).to(device),
                            text=[row["transcript_text"]],
                            sample_rate=sr_i,
                            sample=False,
                        )
                    sample_dict = _extract_cpu(enc_out, 0, wav.shape[0], sr_i)
                    save_futures.append(save_pool.submit(
                        _save_pt, sample_dict, out_dir / f"{row['segment_id']}.pt"
                    ))
                    newly_done.append(row["segment_id"])
                except Exception as e2:
                    print(f"  [error] {row['segment_id']}: {e2}")

        processed_count += len(batch_rows)

        # Progress every ~100 samples
        if processed_count - last_progress >= 100:
            last_progress = processed_count
            elapsed = time.time() - t0
            eta = elapsed / processed_count * (total_todo - processed_count)
            print(f"  [{len(already_done) + processed_count}/{len(rows)}]  ETA {eta/60:.1f}m")

        # Checkpoint every ~500 segments
        if len(newly_done) - last_checkpoint >= 500:
            last_checkpoint = len(newly_done)
            all_done = list(already_done | set(newly_done))
            processed_log.write_text(json.dumps(all_done))

    prefetch_pool.shutdown()

    # Wait for all background saves to finish
    for fut in save_futures:
        exc = fut.exception()
        if exc is not None:
            print(f"  [error] background save failed: {exc}")

    save_pool.shutdown()

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
    print("[init] Loading TADA encoder + aligner...")
    encoder = Encoder.from_pretrained(TADA_CODEC_REPO, subfolder=ENCODER_SUBFOLDER).to(device)
    encoder.eval()
    if "cuda" in device:
        import logging
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        encoder = torch.compile(encoder, mode="default")
        print("[init] torch.compile enabled (first batch will be slow)")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    process_dataset(LIBRITTS_R, encoder, tokenizer, device)

    print("[done] Feature extraction complete.")


if __name__ == "__main__":
    main()

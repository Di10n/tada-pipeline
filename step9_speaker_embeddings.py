#!/usr/bin/env python3
"""Step 9: Extract ECAPA-TDNN speaker embeddings for all splits.

Uses SpeechBrain's pretrained ECAPA-TDNN (spkrec-ecapa-voxceleb) to extract
192-dim speaker embeddings from original audio segments. Embeddings are saved
into each sample's .pt feature file under the key "speaker_embedding".

Processes train, val, and test splits. Expects:
  - /root/work/data/processed/manifest_train.json
  - /root/work/data/processed/manifest_val.json
  - /root/work/data/test/manifest.json
  - Original audio segments accessible (LibriTTS-R WAVs)
"""

import json
import sys
import time
from pathlib import Path

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

from config import DATA_ROOT, SEGMENTS_DIR, SAMPLE_RATE

PROCESSED_DIR = DATA_ROOT / "processed"
TEST_DIR = DATA_ROOT / "test"

# ECAPA-TDNN expects 16kHz input
ECAPA_SR = 16000

BATCH_SIZE = 64
SAVE_EVERY = 64  # flush pending saves every N embeddings


def load_speaker_model(device: str) -> EncoderClassifier:
    """Load pretrained ECAPA-TDNN speaker encoder."""
    print("[init] Loading ECAPA-TDNN speaker model...")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return spk_model


def build_audio_index(split: str) -> dict[str, Path]:
    """Walk audio directories once and return {stem: path} for all WAVs."""
    index: dict[str, Path] = {}
    if split == "test":
        raw_test_dir = DATA_ROOT / "raw" / "libritts_r_test"
        if raw_test_dir.exists():
            for p in raw_test_dir.rglob("*.wav"):
                index[p.stem] = p
    else:
        # Segments dir takes priority over raw dir
        raw_dir = DATA_ROOT / "raw" / "libritts_r"
        if raw_dir.exists():
            for p in raw_dir.rglob("*.wav"):
                index[p.stem] = p
        seg_dir = SEGMENTS_DIR / "libritts_r"
        if seg_dir.exists():
            for p in seg_dir.rglob("*.wav"):
                index[p.stem] = p  # overwrites raw — segments take priority
    return index


def flush_pending_saves(pending: list[tuple[Path, dict]]) -> None:
    """Write accumulated embeddings back to .pt files."""
    for feature_path, data in pending:
        torch.save(data, feature_path)
    pending.clear()


def extract_embeddings_for_manifest(
    manifest: list[dict],
    split_name: str,
    spk_model: EncoderClassifier,
    device: str,
):
    """Extract speaker embeddings for all samples in a manifest."""
    print(f"\n[embed] Processing {split_name}: {len(manifest)} samples")

    print(f"[index] Building audio file index for {split_name}...")
    audio_index = build_audio_index(split_name)
    print(f"[index] Found {len(audio_index)} audio files")

    done = 0
    skipped = 0
    t0 = time.time()

    # Accumulate a batch of (waveform, feature_path, data_dict) for batched inference
    batch_waveforms: list[torch.Tensor] = []  # each (1, time) at ECAPA_SR, on device
    batch_meta: list[tuple[Path, dict]] = []  # (feature_path, loaded data dict)

    # Pending saves: (feature_path, data_dict with embedding added)
    pending_saves: list[tuple[Path, dict]] = []

    def run_batch() -> int:
        """Run encoder on accumulated batch, return number of embeddings extracted."""
        if not batch_waveforms:
            return 0

        # Pad waveforms to same length for batched inference
        lengths = [w.shape[1] for w in batch_waveforms]
        max_len = max(lengths)
        padded = torch.zeros(len(batch_waveforms), max_len, device=device)
        for j, w in enumerate(batch_waveforms):
            padded[j, :w.shape[1]] = w[0]

        # Relative lengths for SpeechBrain (each in [0, 1])
        rel_lengths = torch.tensor(
            [l / max_len for l in lengths], device=device, dtype=torch.float32,
        )

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled="cuda" in device,
        ):
            embeddings = spk_model.encode_batch(padded, rel_lengths)  # (B, 1, 192)

        embeddings = embeddings.squeeze(1).cpu()  # (B, 192)

        for j, (feature_path, data) in enumerate(batch_meta):
            data["speaker_embedding"] = embeddings[j]
            pending_saves.append((feature_path, data))

        count = len(batch_waveforms)
        batch_waveforms.clear()
        batch_meta.clear()
        return count

    for entry in manifest:
        segment_id = entry["segment_id"]
        feature_path = Path(entry["feature_path"])

        if not feature_path.exists():
            skipped += 1
            continue

        # Check if embedding already exists
        data = torch.load(feature_path, map_location="cpu", weights_only=True)
        if "speaker_embedding" in data:
            done += 1
            continue

        # Find audio via cached index
        audio_path = audio_index.get(segment_id)
        if audio_path is None:
            if skipped < 5:
                print(f"  [warn] No audio found for {segment_id}")
            skipped += 1
            continue

        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.to(device)

            if sr != ECAPA_SR:
                waveform = torchaudio.functional.resample(waveform, sr, ECAPA_SR)

            batch_waveforms.append(waveform)
            batch_meta.append((feature_path, data))

        except Exception as e:
            if skipped < 5:
                print(f"  [error] {segment_id}: {e}")
            skipped += 1
            continue

        # Run batch when full
        if len(batch_waveforms) >= BATCH_SIZE:
            done += run_batch()

            # Flush saves periodically
            if len(pending_saves) >= SAVE_EVERY:
                flush_pending_saves(pending_saves)

            if (done + skipped) % 100 < BATCH_SIZE:
                elapsed = time.time() - t0
                eta = elapsed / (done + skipped) * (len(manifest) - done - skipped)
                print(f"  [{done + skipped}/{len(manifest)}] {done} done, {skipped} skipped  ETA {eta/60:.1f}m")

    # Process remaining samples
    done += run_batch()
    flush_pending_saves(pending_saves)

    print(f"[embed] {split_name}: {done} embeddings extracted, {skipped} skipped")
    return done, skipped


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split(s) to process",
    )
    args = parser.parse_args()

    spk_model = load_speaker_model(args.device)

    splits = {}
    if args.split in ("train", "all"):
        path = PROCESSED_DIR / "manifest_train.json"
        if path.exists():
            with open(path) as f:
                splits["train"] = json.load(f)
        else:
            print(f"[skip] {path} not found")

    if args.split in ("val", "all"):
        path = PROCESSED_DIR / "manifest_val.json"
        if path.exists():
            with open(path) as f:
                splits["val"] = json.load(f)
        else:
            print(f"[skip] {path} not found")

    if args.split in ("test", "all"):
        path = TEST_DIR / "manifest.json"
        if path.exists():
            with open(path) as f:
                splits["test"] = json.load(f)
        else:
            print(f"[skip] {path} not found")

    total_done = 0
    total_skipped = 0
    for split_name, manifest in splits.items():
        done, skipped = extract_embeddings_for_manifest(
            manifest, split_name, spk_model, args.device
        )
        total_done += done
        total_skipped += skipped

    print(f"\n[done] Speaker embeddings complete: {total_done} extracted, {total_skipped} skipped")


if __name__ == "__main__":
    main()

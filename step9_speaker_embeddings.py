#!/usr/bin/env python3
"""Step 9: Extract ECAPA-TDNN speaker embeddings for all splits.

Uses SpeechBrain's pretrained ECAPA-TDNN (spkrec-ecapa-voxceleb) to extract
192-dim speaker embeddings from original audio segments. Embeddings are saved
into each sample's .pt feature file under the key "speaker_embedding".

Processes train, val, and test splits. Expects:
  - /workspace/data/processed/manifest_train.json
  - /workspace/data/processed/manifest_val.json
  - /workspace/data/test/manifest.json
  - Original audio segments accessible (LibriTTS-R WAVs)
"""

import json
import sys
from pathlib import Path

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

from config import DATA_ROOT, SEGMENTS_DIR, SAMPLE_RATE

PROCESSED_DIR = DATA_ROOT / "processed"
TEST_DIR = DATA_ROOT / "test"

# ECAPA-TDNN expects 16kHz input
ECAPA_SR = 16000


def load_speaker_model(device: str) -> EncoderClassifier:
    """Load pretrained ECAPA-TDNN speaker encoder."""
    print("[init] Loading ECAPA-TDNN speaker model...")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return spk_model


def find_audio_for_segment(segment_id: str, split: str) -> Path | None:
    """Locate the original audio file for a segment.

    For train/val: look in LibriTTS-R raw/segments directories.
    For test: look in the downloaded test-clean raw audio.
    """
    if split == "test":
        raw_test_dir = DATA_ROOT / "raw" / "libritts_r_test"
        matches = list(raw_test_dir.rglob(f"{segment_id}.wav"))
        if matches:
            return matches[0]
    else:
        # LibriTTS-R segments are the original files — check segments dir first
        candidate = SEGMENTS_DIR / "libritts_r" / f"{segment_id}.wav"
        if candidate.exists():
            return candidate
        # Fall back to raw dir
        raw_dir = DATA_ROOT / "raw" / "libritts_r"
        matches = list(raw_dir.rglob(f"{segment_id}.wav"))
        if matches:
            return matches[0]

    return None


def extract_embeddings_for_manifest(
    manifest: list[dict],
    split_name: str,
    spk_model: EncoderClassifier,
    device: str,
):
    """Extract speaker embeddings for all samples in a manifest."""
    print(f"\n[embed] Processing {split_name}: {len(manifest)} samples")

    done = 0
    skipped = 0

    for i, entry in enumerate(manifest):
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

        # Find audio
        audio_path = find_audio_for_segment(segment_id, split_name)
        if audio_path is None:
            if skipped < 5:
                print(f"  [warn] No audio found for {segment_id}")
            skipped += 1
            continue

        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz for ECAPA-TDNN
            if sr != ECAPA_SR:
                waveform = torchaudio.functional.resample(waveform, sr, ECAPA_SR)

            # Extract embedding — SpeechBrain expects (batch, time)
            waveform = waveform.to(device)
            with torch.no_grad():
                embedding = spk_model.encode_batch(waveform)  # (1, 1, 192)
                embedding = embedding.squeeze().cpu()  # (192,)

            # Save back into .pt file
            data["speaker_embedding"] = embedding
            torch.save(data, feature_path)
            done += 1

        except Exception as e:
            if skipped < 5:
                print(f"  [error] {segment_id}: {e}")
            skipped += 1
            continue

        if (done + skipped) % 500 == 0:
            print(f"  [{done + skipped}/{len(manifest)}] {done} done, {skipped} skipped")

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

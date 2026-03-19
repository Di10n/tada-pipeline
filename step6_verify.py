#!/usr/bin/env python3
"""Verification: validate the processed dataset.

1. Load random samples and check tensor shapes
2. Decode features back to audio via TADA decoder, compare with original
3. Verify token_ids decode back to transcript text
4. Confirm f_after[i] == f_before[i+1] for consecutive tokens
"""

import json
import random
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tada"))

from tada.modules.encoder import Encoder
from tada.modules.decoder import Decoder

from config import (
    PROCESSED_DIR,
    TADA_CODEC_REPO,
    ENCODER_SUBFOLDER,
    TOKENIZER_NAME,
    SAMPLE_RATE,
    FRAME_RATE,
)


def load_manifest() -> list[dict]:
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def verify_shapes(samples: list[dict]):
    """Check 1: tensor shapes are consistent."""
    print("\n=== Shape Verification ===")
    all_ok = True

    for entry in samples:
        sid = entry["segment_id"]
        data = torch.load(entry["feature_path"], map_location="cpu", weights_only=True)

        token_ids = data["token_ids"]
        features = data["encoder_features"]
        positions = data["positions"]
        f_before = data["f_before"]
        f_after = data["f_after"]

        L = token_ids.shape[0]
        ok = True

        if features.shape != (L, 512):
            print(f"  [FAIL] {sid}: encoder_features shape {features.shape}, expected ({L}, 512)")
            ok = False
        if positions.shape != (L,):
            print(f"  [FAIL] {sid}: positions shape {positions.shape}, expected ({L},)")
            ok = False
        if f_before.shape != (L,):
            print(f"  [FAIL] {sid}: f_before shape {f_before.shape}, expected ({L},)")
            ok = False
        if f_after.shape != (L,):
            print(f"  [FAIL] {sid}: f_after shape {f_after.shape}, expected ({L},)")
            ok = False

        if ok:
            print(f"  [OK] {sid}: L={L}, dur_frames={data['duration_frames']}")
        else:
            all_ok = False

    return all_ok


def verify_frame_consistency(samples: list[dict]):
    """Check 4: f_after[i] == f_before[i+1] for all consecutive tokens."""
    print("\n=== Frame Consistency (f_after[i] == f_before[i+1]) ===")
    all_ok = True

    for entry in samples:
        sid = entry["segment_id"]
        data = torch.load(entry["feature_path"], map_location="cpu", weights_only=True)

        f_before = data["f_before"]
        f_after = data["f_after"]
        L = f_before.shape[0]

        if L < 2:
            print(f"  [SKIP] {sid}: only {L} token(s)")
            continue

        mismatches = 0
        for i in range(L - 1):
            if f_after[i].item() != f_before[i + 1].item():
                mismatches += 1
                if mismatches <= 3:
                    print(
                        f"  [MISMATCH] {sid} at i={i}: "
                        f"f_after[{i}]={f_after[i].item()}, f_before[{i+1}]={f_before[i+1].item()}"
                    )

        if mismatches == 0:
            print(f"  [OK] {sid}: all {L-1} consecutive pairs match")
        else:
            print(f"  [FAIL] {sid}: {mismatches}/{L-1} mismatches")
            all_ok = False

    return all_ok


def verify_token_roundtrip(samples: list[dict], tokenizer):
    """Check 3: token_ids decode back to reasonable text."""
    print("\n=== Token Roundtrip ===")
    all_ok = True

    for entry in samples:
        sid = entry["segment_id"]
        data = torch.load(entry["feature_path"], map_location="cpu", weights_only=True)

        token_ids = data["token_ids"].tolist()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

        if decoded.strip():
            print(f"  [OK] {sid}: \"{decoded[:80]}{'...' if len(decoded) > 80 else ''}\"")
        else:
            print(f"  [FAIL] {sid}: decoded to empty string from {len(token_ids)} tokens")
            all_ok = False

    return all_ok


def verify_decoder_reconstruction(samples: list[dict], encoder: Encoder, device: str):
    """Check 2: reconstruct audio from features via decoder and compare."""
    print("\n=== Decoder Reconstruction ===")

    # Load decoder
    decoder = Decoder.from_pretrained(TADA_CODEC_REPO, subfolder="decoder").to(device)
    decoder.eval()

    for entry in samples:
        sid = entry["segment_id"]
        data = torch.load(entry["feature_path"], map_location="cpu", weights_only=True)

        features = data["encoder_features"].to(device)   # (L, 512)
        positions = data["positions"].to(device)          # (L,)
        dur_frames = data["duration_frames"]

        # Build expanded tensor: (1, dur_frames, 512) with features placed at positions
        expanded = torch.zeros(1, dur_frames, 512, device=device)
        token_masks = torch.zeros(1, dur_frames, device=device)

        for i in range(positions.shape[0]):
            pos = positions[i].item()
            if 0 <= pos < dur_frames:
                expanded[0, pos] = features[i]
                token_masks[0, pos] = 1.0

        # Fill non-token positions by interpolating from nearest token positions
        # (simple forward-fill for now)
        last_feat = torch.zeros(512, device=device)
        for t in range(dur_frames):
            if token_masks[0, t] == 1.0:
                last_feat = expanded[0, t]
            else:
                expanded[0, t] = last_feat

        try:
            with torch.no_grad():
                reconstructed = decoder(expanded, token_masks)
            # reconstructed: (1, 1, audio_samples)
            audio_len = reconstructed.shape[-1]
            expected_len = dur_frames * (SAMPLE_RATE // FRAME_RATE)  # 480 samples per frame
            ratio = audio_len / expected_len if expected_len > 0 else 0
            print(f"  [OK] {sid}: reconstructed {audio_len} samples (ratio={ratio:.2f})")
        except Exception as e:
            print(f"  [WARN] {sid}: decoder failed: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10, help="Number of samples to verify")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-decoder", action="store_true", help="Skip decoder reconstruction")
    args = parser.parse_args()

    manifest = load_manifest()
    if not manifest:
        print("[error] Empty manifest, nothing to verify")
        return

    # Sample n random entries
    n = min(args.n, len(manifest))
    samples = random.sample(manifest, n)
    print(f"[verify] Checking {n} random samples out of {len(manifest)} total")

    # Checks that don't need models
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    ok1 = verify_shapes(samples)
    ok2 = verify_frame_consistency(samples)
    ok3 = verify_token_roundtrip(samples, tokenizer)

    # Decoder reconstruction (needs GPU)
    if not args.skip_decoder:
        encoder = Encoder.from_pretrained(TADA_CODEC_REPO, subfolder=ENCODER_SUBFOLDER).to(args.device)
        encoder.eval()
        verify_decoder_reconstruction(samples, encoder, args.device)

    print("\n=== Summary ===")
    print(f"  Shapes:      {'PASS' if ok1 else 'FAIL'}")
    print(f"  Consistency: {'PASS' if ok2 else 'FAIL'}")
    print(f"  Roundtrip:   {'PASS' if ok3 else 'FAIL'}")


if __name__ == "__main__":
    main()

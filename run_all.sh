#!/usr/bin/env bash
# Full TADA data pipeline — run on RunPod with GPU.
# Uses LibriTTS-R only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "TADA Data Pipeline"
echo "DATA_ROOT=${DATA_ROOT:-/workspace/data}"
echo "============================================"

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[setup] Installing dependencies..."
pip install -q hume-tada torchaudio transformers webdataset speechbrain 2>/dev/null || true

# ── Step 1: Download ──────────────────────────────────────────────────────────
echo ""
echo "[step 1] Downloading LibriTTS-R..."
python step1_download.py

# ── Step 2: Build manifest ────────────────────────────────────────────────────
echo ""
echo "[step 2] Building manifest..."
python step2_vad_segment.py

# ── Step 3: Transcription ────────────────────────────────────────────────────
echo ""
echo "[step 3] Mapping transcripts..."
python step3_transcribe.py

# ── Step 4: Feature extraction ────────────────────────────────────────────────
echo ""
echo "[step 4] Extracting alignments & encoder features..."
python step4_extract_features.py

# ── Step 5: Filter and build ─────────────────────────────────────────────────
echo ""
echo "[step 5] Filtering and building final dataset..."
python step5_filter_and_build.py --shards

# ── Step 6: Verification ─────────────────────────────────────────────────────
echo ""
echo "[step 6] Running verification..."
python step6_verify.py -n 10

# ── Step 7: Download & preprocess test-clean ──────────────────────────────────
echo ""
echo "[step 7] Download & preprocess LibriTTS-R test-clean..."
python step7_download_test.py

# ── Step 8: Train/val split by speaker ────────────────────────────────────────
echo ""
echo "[step 8] Splitting train/val by speaker..."
python step8_split_train_val.py

# ── Step 9: Speaker embeddings ────────────────────────────────────────────────
echo ""
echo "[step 9] Extracting speaker embeddings (all splits)..."
python step9_speaker_embeddings.py

# ── Step 10: Final verification ───────────────────────────────────────────────
echo ""
echo "[step 10] Running final verification..."
python step10_verify_final.py

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "  Train manifest: ${DATA_ROOT:-/workspace/data}/processed/manifest_train.json"
echo "  Val manifest:   ${DATA_ROOT:-/workspace/data}/processed/manifest_val.json"
echo "  Test manifest:  ${DATA_ROOT:-/workspace/data}/test/manifest.json"
echo "============================================"

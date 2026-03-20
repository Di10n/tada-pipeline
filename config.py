"""Shared configuration for the TADA data pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# When running on RunPod, set DATA_ROOT=/root/work/data via env var.
import os

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/root/work/data"))

RAW_DIR = DATA_ROOT / "raw"
SEGMENTS_DIR = DATA_ROOT / "segments"
PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
SHARDS_DIR = DATA_ROOT / "shards"

# Dataset name
LIBRITTS_R = "libritts_r"

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 24_000
MAX_DURATION_SEC = 30.0
MIN_DURATION_SEC = 1.0
FRAME_RATE = 50  # TADA encoder frame rate (Hz)

# ── TADA model identifiers ────────────────────────────────────────────────────
TADA_CODEC_REPO = "HumeAI/tada-codec"
ENCODER_SUBFOLDER = "encoder"
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

# ── Quality filtering thresholds ──────────────────────────────────────────────
MAX_CONSECUTIVE_FRAMES = 3  # max consecutive aligned positions span
MAX_POSITION_GAP_FRAMES = 150  # ~3 seconds at 50Hz

# ── Processing ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64  # batch size for encoder / ASR inference
NUM_WORKERS = 4

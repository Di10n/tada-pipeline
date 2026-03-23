"""TADA Data Pipeline on Modal Serverless.

Fused architecture: each GPU worker processes one recording end-to-end
(VAD → transcribe → features) with zero intermediate
I/O. Only final .pt files are written to the volume.

Setup (one-time):
    pip install modal
    modal setup

    Ensure .env has: R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET, HF_TOKEN

Run:
    modal run modal_pipeline.py
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24_000
MAX_DURATION_SEC = 30.0
MIN_DURATION_SEC = 1.0
FRAME_RATE = 50
BATCH_SIZE = 64
MAX_CONSECUTIVE_FRAMES = 3
MAX_POSITION_GAP_FRAMES = 150
DATASET_NAME = "r2_audio"
R2_PREFIX = "en-sample-20260322T191419Z-0amulojz"
TADA_CODEC_REPO = "HumeAI/tada-codec"
ENCODER_SUBFOLDER = "encoder"
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

MERGE_GAP_SEC = 0.5
SPLIT_TARGET_SEC = 25.0

TRAIN_RATIO = 0.95
SPLIT_SEED = 42

DATA_ROOT = Path("/data")
RAW_DIR = DATA_ROOT / "raw"
SEGMENTS_DIR = DATA_ROOT / "segments"
PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"

# ── Modal plumbing ────────────────────────────────────────────────────────────

app = modal.App("tada-pipeline")

volume = modal.Volume.from_name("tada-pipeline-data", create_if_missing=True)
VOLUME_MOUNT = {"/data": volume}

env_secret = modal.Secret.from_dotenv(path=Path(__file__).parent / ".env")

GPU_TYPE = "A100-80GB"

# Slim image for lightweight tasks (list files, download, upload).
slim_image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3")

# Full image for GPU work and anything that needs torch.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox")
    .pip_install("torch", "torchaudio")
    .pip_install("nemo_toolkit[asr]")
    .pip_install(
        "hume-tada",
        "pyannote-audio",
        "transformers",
        "boto3",
    )
    # Fix protobuf/onnx LAST — NeMo installs incompatible versions.
    .run_commands(
        "pip install 'protobuf>=4.21,<5' 'onnx>=1.14,<1.17' && "
        "python -c 'from google.protobuf.internal import builder; print(\"protobuf OK\")'"
    )
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
    )


def _check_quality(sample: dict) -> str | None:
    """Return rejection reason or None if sample passes."""
    positions = sample["positions"]
    duration_frames = sample["duration_frames"]
    L = positions.shape[0]

    if L == 0:
        return "empty_tokens"

    dur_sec = duration_frames / FRAME_RATE
    if dur_sec < MIN_DURATION_SEC:
        return "too_short"
    if dur_sec > MAX_DURATION_SEC:
        return "too_long"

    if L > 1:
        diffs = positions[1:] - positions[:-1]
        run_len = 1
        for d in diffs:
            if d.item() == 0:
                run_len += 1
                if run_len > MAX_CONSECUTIVE_FRAMES:
                    return "consecutive_frames"
            else:
                run_len = 1
        if diffs.max().item() > MAX_POSITION_GAP_FRAMES:
            return "large_gap"

    return None


def _extract_cpu(enc_out, j, audio_len_samples, sr):
    """Pull features for sample *j* out of a batched encoder output → CPU dict."""
    import torch

    token_ids = enc_out.text_tokens[j].cpu()
    token_positions = enc_out.token_positions[j].cpu()
    token_values = enc_out.token_values[j].cpu()

    if enc_out.text_tokens_len is not None:
        tlen = enc_out.text_tokens_len[j].item()
        token_ids = token_ids[:tlen]
        token_positions = token_positions[:tlen]
        token_values = token_values[:tlen]

    total_frames = int(audio_len_samples / sr * FRAME_RATE)
    f_before, f_after = _compute_frame_gaps(token_positions, total_frames)

    return {
        "token_ids": token_ids,
        "encoder_features": token_values,
        "positions": token_positions,
        "f_before": f_before,
        "f_after": f_after,
        "duration_frames": total_frames,
    }


def _compute_frame_gaps(positions, total_frames):
    import torch

    L = positions.shape[0]
    f_before = torch.zeros(L, dtype=torch.long)
    f_after = torch.zeros(L, dtype=torch.long)

    f_before[0] = positions[0]
    for i in range(1, L):
        f_before[i] = positions[i] - positions[i - 1] - 1

    for i in range(L - 1):
        f_after[i] = positions[i + 1] - positions[i] - 1
    f_after[L - 1] = total_frames - 1 - positions[L - 1]

    return f_before.clamp(min=0), f_after.clamp(min=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Download from R2
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(image=slim_image, secrets=[env_secret], volumes=VOLUME_MOUNT, timeout=300)
def list_r2_files() -> list[dict]:
    """Return [{key, size}, ...] for every audio object in R2."""
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict] = []
    for page in paginator.paginate(
        Bucket=os.environ["R2_BUCKET"], Prefix=f"{R2_PREFIX}/audio/"
    ):
        for obj in page.get("Contents", []):
            objects.append({"key": obj["Key"], "size": obj["Size"]})
    print(f"[list] {len(objects)} audio files in R2")
    return objects


@app.function(
    image=slim_image, secrets=[env_secret], volumes=VOLUME_MOUNT, timeout=600, retries=2
)
def download_file(file_info: dict) -> str:
    """Download one file from R2 → volume."""
    volume.reload()
    key = file_info["key"]
    filename = key.split("/")[-1]
    dest = RAW_DIR / DATASET_NAME / filename

    if dest.exists() and dest.stat().st_size == file_info["size"]:
        return "skip"

    dest.parent.mkdir(parents=True, exist_ok=True)
    _get_s3_client().download_file(os.environ["R2_BUCKET"], key, str(dest))
    volume.commit()
    return "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Fused per-recording processing
#   VAD → transcribe → feature extraction
#   Audio stays in memory. Only .pt files touch the volume.
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(image=slim_image, volumes=VOLUME_MOUNT, timeout=120)
def list_recordings() -> list[str]:
    """Return MP3 filenames on the volume."""
    volume.reload()
    raw_dir = RAW_DIR / DATASET_NAME
    if not raw_dir.exists():
        return []
    mp3s = sorted(p.name for p in raw_dir.glob("*.mp3"))
    print(f"[list] {len(mp3s)} MP3 files")
    return mp3s


@app.function(image=slim_image, volumes=VOLUME_MOUNT, timeout=120)
def list_done_recordings() -> dict:
    """Return {recording_id: n_segments} for recordings with a .done marker."""
    volume.reload()
    done_dir = FEATURES_DIR / DATASET_NAME
    done: dict[str, int] = {}
    if done_dir.exists():
        for p in done_dir.glob("*.done"):
            try:
                data = json.loads(p.read_text())
                done[p.stem] = data.get("n_segments", 0)
            except Exception:
                pass
    return done


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    secrets=[env_secret],
    volumes=VOLUME_MOUNT,
    timeout=3600,
)
class FusedWorker:
    """Loads all four models once, then processes recordings end-to-end."""

    @modal.enter()
    def load_models(self):
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        import nemo.collections.asr as nemo_asr
        from tada.modules.encoder import Encoder

        volume.reload()
        t0 = time.time()

        # Pyannote checkpoints fail with torch 2.6+'s weights_only=True.
        # Patch torch.load during model loading only, then restore.
        import functools
        _orig_load = torch.load

        @functools.wraps(_orig_load)
        def _permissive_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _permissive_load
        try:
            self.vad = PyannotePipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=os.environ["HF_TOKEN"],
            )
            self.vad.to(torch.device("cuda"))

            self.asr = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2"
            )
            self.asr.eval()

            # DAC's snake activation is @torch.jit.script compiled.
            # The JIT kernel produces "CUDA driver error: invalid argument"
            # on torch 2.7 with batched inputs. Replace with identical
            # eager-mode version before loading the encoder (which uses DAC).
            import dac.nn.layers

            def _eager_snake(x, alpha):
                shape = x.shape
                x = x.reshape(shape[0], shape[1], -1)
                x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
                x = x.reshape(shape)
                return x

            dac.nn.layers.snake = _eager_snake

            self.encoder = Encoder.from_pretrained(
                TADA_CODEC_REPO, subfolder=ENCODER_SUBFOLDER
            ).to("cuda")
            self.encoder.eval()
            self.tok = self.encoder.aligner.tokenizer
        finally:
            torch.load = _orig_load

        print(f"[init] All 3 models loaded  ({time.time() - t0:.0f}s)")

    @modal.method()
    def process(self, mp3_filename: str) -> list[dict]:
        """Full pipeline for one recording. Returns manifest entries."""
        import shutil
        import tempfile

        import torch
        import torchaudio
        from tada.utils.text import normalize_text

        t_rec = time.time()
        mp3_path = RAW_DIR / DATASET_NAME / mp3_filename
        if not mp3_path.exists():
            print(f"[error] {mp3_filename} not found")
            return []

        recording_id = mp3_filename.split("_")[0]
        out_dir = FEATURES_DIR / DATASET_NAME
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume check ───────────────────────────────────────────────────
        done_marker = out_dir / f"{recording_id}.done"
        if done_marker.exists():
            print(f"[{recording_id}] already done, skipping")
            return []

        # Clean up .pt files from any partial previous run so we don't leave
        # orphans if VAD produces different segment boundaries this time.
        for stale in out_dir.glob(f"{recording_id}_*.pt"):
            stale.unlink()

        def _mark_done(n: int = 0) -> list[dict]:
            """Write .done marker, commit, return empty list."""
            done_marker.write_text(json.dumps({"n_segments": n}))
            volume.commit()
            return []

        # ── 1. Load & resample ─────────────────────────────────────────────
        try:
            waveform, sr = torchaudio.load(str(mp3_path))
        except Exception as e:
            print(f"[error] {recording_id}: load failed: {e}")
            return _mark_done()

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        audio_dur = waveform.shape[1] / sr

        # ── 2. VAD ─────────────────────────────────────────────────────────
        vad_output = self.vad({"waveform": waveform, "sample_rate": sr})
        raw_segs = [(t.start, t.end) for t in vad_output.get_timeline()]
        if not raw_segs:
            return _mark_done()

        # Merge close segments
        merged = [raw_segs[0]]
        for s, e in raw_segs[1:]:
            if s - merged[-1][1] < MERGE_GAP_SEC:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Split long, discard short
        final_segs: list[tuple[float, float]] = []
        for s, e in merged:
            dur = e - s
            if dur <= MAX_DURATION_SEC:
                if dur >= MIN_DURATION_SEC:
                    final_segs.append((s, e))
            else:
                n = max(1, round(dur / SPLIT_TARGET_SEC))
                cd = dur / n
                for i in range(n):
                    cs, ce = s + i * cd, s + (i + 1) * cd
                    if ce - cs >= MIN_DURATION_SEC:
                        final_segs.append((cs, ce))

        if not final_segs:
            return _mark_done()

        # Slice waveforms (stay in memory — never written as WAV to volume)
        # Each element: (segment_id, waveform_1ch_tensor, duration_sec)
        segments: list[tuple[str, "torch.Tensor", float]] = []
        for idx, (s, e) in enumerate(final_segs):
            ss = int(s * sr)
            es = min(int(e * sr), waveform.shape[1])
            seg_wav = waveform[:, ss:es]  # (1, samples)
            dur = seg_wav.shape[1] / sr
            if dur < MIN_DURATION_SEC:
                continue
            segments.append((f"{recording_id}_{idx:04d}", seg_wav, dur))

        if not segments:
            return _mark_done()

        # ── 3. Transcribe (NeMo needs file paths → /tmp, not volume) ──────
        tmpdir = tempfile.mkdtemp()
        tmp_paths = []
        for sid, seg_wav, _ in segments:
            p = os.path.join(tmpdir, f"{sid}.wav")
            torchaudio.save(p, seg_wav, sr)
            tmp_paths.append(p)

        try:
            texts = self.asr.transcribe(
                tmp_paths, batch_size=min(len(tmp_paths), BATCH_SIZE)
            )
            if isinstance(texts, tuple):
                texts = texts[0]
            if hasattr(texts[0], "text"):
                texts = [t.text for t in texts]
        except Exception as e:
            print(f"[error] ASR batch {recording_id}: {e}; sequential fallback")
            texts = []
            for p in tmp_paths:
                try:
                    r = self.asr.transcribe([p], batch_size=1)
                    if isinstance(r, tuple):
                        r = r[0]
                    texts.append(r[0].text if hasattr(r[0], "text") else r[0])
                except Exception:
                    texts.append("")

        shutil.rmtree(tmpdir, ignore_errors=True)

        # Free NeMo's GPU cache before feature extraction
        torch.cuda.empty_cache()

        # Pair segments with transcriptions, drop empty
        transcribed: list[tuple[str, "torch.Tensor", float, str]] = []
        for (sid, wav, dur), text in zip(segments, texts):
            text = text.strip() if isinstance(text, str) else ""
            if text:
                transcribed.append((sid, wav, dur, text))

        if not transcribed:
            return _mark_done()

        # ── 4. Feature extraction (TADA encoder, batched) ──────────────────
        # Accumulate: (segment_id, waveform, duration, sample_dict)
        featured: list[tuple[str, "torch.Tensor", float, dict]] = []

        for b_start in range(0, len(transcribed), BATCH_SIZE):
            batch = transcribed[b_start : b_start + BATCH_SIZE]

            wavs = [wav.squeeze(0) for _, wav, _, _ in batch]  # list of (samples,)
            txts = [text for _, _, _, text in batch]
            a_lens = torch.tensor([w.shape[0] for w in wavs])

            max_len = int(a_lens.max().item())
            padded = torch.zeros(len(wavs), max_len)
            for j, w in enumerate(wavs):
                padded[j, : w.shape[0]] = w
            padded = padded.to("cuda")
            a_lens_dev = a_lens.to("cuda").unsqueeze(1)

            normed = [normalize_text(t) for t in txts]
            tseqs = [
                self.tok.encode(
                    t, add_special_tokens=False, return_tensors="pt"
                ).squeeze(0)
                for t in normed
            ]
            ttl = torch.tensor([t.shape[0] for t in tseqs], device="cuda")
            tt = torch.nn.utils.rnn.pad_sequence(
                tseqs, batch_first=True, padding_value=self.tok.eos_token_id
            ).to("cuda")

            batch_ok = True
            try:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    enc_out = self.encoder(
                        padded,
                        text_tokens=tt,
                        text_token_len=ttl,
                        audio_length=a_lens_dev,
                        sample_rate=sr,
                        sample=False,
                    )
                # Force async CUDA errors to surface here as a catchable
                # exception, before they corrupt GPU state and cause SIGABRT.
                torch.cuda.synchronize()
                for j, (sid, wav, dur, _text) in enumerate(batch):
                    try:
                        sd = _extract_cpu(enc_out, j, int(a_lens[j].item()), sr)
                        featured.append((sid, wav, dur, sd))
                    except Exception as e:
                        print(f"[error] extract {sid}: {e}")
            except Exception as e:
                print(f"[warn] Feature batch failed ({e}), sequential fallback")
                # Reset CUDA state so the sequential fallback starts clean.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                batch_ok = False

            if not batch_ok:
                for sid, wav, dur, text in batch:
                    try:
                        with torch.no_grad(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            enc_out = self.encoder(
                                wav.to("cuda"),  # (1, samples)
                                text=[text],
                                sample_rate=sr,
                                sample=False,
                            )
                        torch.cuda.synchronize()
                        sd = _extract_cpu(enc_out, 0, wav.shape[1], sr)
                        featured.append((sid, wav, dur, sd))
                    except Exception as e2:
                        print(f"[error] {sid}: {e2}")
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

        if not featured:
            return _mark_done()

        # ── 5. Save .pt files to volume ────────────────────────────────────
        manifest_entries: list[dict] = []
        for sid, _wav, _dur, sd in featured:
            pt_path = out_dir / f"{sid}.pt"
            torch.save(sd, pt_path)
            manifest_entries.append(
                {
                    "segment_id": sid,
                    "dataset": DATASET_NAME,
                    "num_tokens": int(sd["token_ids"].shape[0]),
                    "duration_frames": int(sd["duration_frames"]),
                    "feature_path": str(pt_path),
                }
            )

        # Write completion marker — this is the source of truth for resume.
        # Written after all .pt files, before commit, so the commit is
        # all-or-nothing: either everything is persisted or nothing is.
        done_marker.write_text(json.dumps({"n_segments": len(manifest_entries)}))

        volume.commit()
        elapsed = time.time() - t_rec
        print(
            f"[{recording_id}] {audio_dur:.0f}s audio → "
            f"{len(manifest_entries)} features  ({elapsed:.1f}s)"
        )
        return manifest_entries


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Quality filter & manifest build
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(image=gpu_image, volumes=VOLUME_MOUNT, timeout=3600)
def filter_and_build() -> list[dict]:
    """Scan .pt files, quality-filter, write manifest.json. Returns kept entries."""
    import torch

    volume.reload()

    feat_dir = FEATURES_DIR / DATASET_NAME
    if not feat_dir.exists():
        print("[error] Feature directory missing")
        return []

    pt_files = sorted(feat_dir.glob("*.pt"))
    n_total = len(pt_files)
    print(f"[filter] {n_total} feature files to check")
    t0 = time.time()

    manifest: list[dict] = []
    stats: Counter = Counter()

    for i, pt_path in enumerate(pt_files):
        sid = pt_path.stem
        if sid.startswith("_"):
            continue
        try:
            sample = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            stats["load_error"] += 1
            continue

        reason = _check_quality(sample)
        if reason:
            stats[reason] += 1
            continue

        manifest.append(
            {
                "segment_id": sid,
                "dataset": DATASET_NAME,
                "num_tokens": int(sample["token_ids"].shape[0]),
                "duration_frames": int(sample["duration_frames"]),
                "feature_path": str(pt_path),
            }
        )
        stats["kept"] += 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    total = sum(stats.values())
    kept_pct = stats["kept"] / total * 100 if total else 0
    total_hours = sum(e["duration_frames"] for e in manifest) / FRAME_RATE / 3600
    total_tokens = sum(e["num_tokens"] for e in manifest)

    print(f"[filter] {total} processed in {elapsed:.1f}s:")
    for reason, count in stats.most_common():
        pct = count / total * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print(
        f"[manifest] {len(manifest)} samples ({kept_pct:.0f}% kept) → {manifest_path}"
    )
    print(f"  ~{total_hours:.1f} hours, {total_tokens:,} tokens")
    volume.commit()
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Train / val split
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(image=gpu_image, volumes=VOLUME_MOUNT, timeout=3600)
def split_train_val():
    """Split manifest by recording → manifest_train.json + manifest_val.json."""
    volume.reload()

    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        print("[error] manifest.json not found")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[split] {len(manifest)} samples to split (ratio {TRAIN_RATIO:.0%} train)")

    recording_samples: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        rec_id = entry["segment_id"].rsplit("_", 1)[0]
        recording_samples[rec_id].append(entry)

    recordings = sorted(recording_samples.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(recordings)

    n_train = int(len(recordings) * TRAIN_RATIO)
    train_recs = set(recordings[:n_train])

    train_manifest: list[dict] = []
    val_manifest: list[dict] = []
    for rec_id in recordings:
        target = train_manifest if rec_id in train_recs else val_manifest
        target.extend(recording_samples[rec_id])

    for name, data in [
        ("manifest_train.json", train_manifest),
        ("manifest_val.json", val_manifest),
    ]:
        path = PROCESSED_DIR / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[write] {len(data)} samples → {path}")

    train_h = sum(e.get("duration_frames", 0) for e in train_manifest) / FRAME_RATE / 3600
    val_h = sum(e.get("duration_frames", 0) for e in val_manifest) / FRAME_RATE / 3600
    print(f"[split] Train: ~{train_h:.0f}h ({len(train_recs)} recs), Val: ~{val_h:.0f}h")
    volume.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Final verification
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(image=gpu_image, volumes=VOLUME_MOUNT, timeout=3600)
def verify_final():
    """Check keys, shapes, and recording disjointness."""
    import torch

    volume.reload()

    REQUIRED_KEYS = [
        "token_ids",
        "encoder_features",
        "positions",
        "f_before",
        "f_after",
    ]
    results: dict[str, bool] = {}

    for split_name in ["train", "val"]:
        path = PROCESSED_DIR / f"manifest_{split_name}.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        with open(path) as f:
            manifest = json.load(f)
        if not manifest:
            continue

        # Keys & shapes
        rng = random.Random(123)
        samples = rng.sample(manifest, min(5, len(manifest)))
        keys_ok = True
        print(f"\n=== Keys & Shapes: {split_name} ===")
        for entry in samples:
            sid = entry["segment_id"]
            pt_path = Path(entry["feature_path"])
            if not pt_path.exists():
                print(f"  [FAIL] {sid}: file not found")
                keys_ok = False
                continue
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            missing = [k for k in REQUIRED_KEYS if k not in data]
            if missing:
                print(f"  [FAIL] {sid}: missing {missing}")
                keys_ok = False
                continue
            L = data["token_ids"].shape[0]
            errors = []
            if data["encoder_features"].shape != (L, 512):
                errors.append(f"encoder_features {data['encoder_features'].shape}")
            if errors:
                print(f"  [FAIL] {sid}: {'; '.join(errors)}")
                keys_ok = False
            else:
                print(f"  [OK]   {sid}: L={L}")
        results[f"{split_name}_keys"] = keys_ok

    # Recording disjointness
    train_path = PROCESSED_DIR / "manifest_train.json"
    val_path = PROCESSED_DIR / "manifest_val.json"
    if train_path.exists() and val_path.exists():
        with open(train_path) as f:
            train = json.load(f)
        with open(val_path) as f:
            val = json.load(f)
        train_recs = {e["segment_id"].rsplit("_", 1)[0] for e in train}
        val_recs = {e["segment_id"].rsplit("_", 1)[0] for e in val}
        overlap = train_recs & val_recs
        results["disjoint"] = len(overlap) == 0
        print(f"\n=== Disjointness ===")
        print(
            f"  Train: {len(train_recs)} recs, Val: {len(val_recs)} recs, "
            f"Overlap: {len(overlap)}"
        )

    print("\n=== Results ===")
    all_pass = True
    for check, passed in results.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Upload results to R2
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(image=slim_image, secrets=[env_secret], volumes=VOLUME_MOUNT, timeout=300)
def list_upload_files() -> list[dict]:
    """Inventory files to upload (only manifest-referenced features)."""
    volume.reload()
    files: list[dict] = []
    output_prefix = f"{R2_PREFIX}/processed"

    for name in ["manifest.json", "manifest_train.json", "manifest_val.json"]:
        path = PROCESSED_DIR / name
        if path.exists():
            files.append({"local": str(path), "key": f"{output_prefix}/{name}"})

    manifest_path = PROCESSED_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            pt_path = Path(entry["feature_path"])
            if pt_path.exists():
                files.append(
                    {
                        "local": str(pt_path),
                        "key": f"{output_prefix}/features/{pt_path.name}",
                    }
                )

    print(f"[upload] {len(files)} files to upload")
    return files


@app.function(
    image=slim_image, secrets=[env_secret], volumes=VOLUME_MOUNT, timeout=600, retries=2
)
def upload_one_file(file_info: dict) -> str:
    """Upload one file to R2."""
    volume.reload()
    _get_s3_client().upload_file(
        file_info["local"], os.environ["R2_BUCKET"], file_info["key"]
    )
    return "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def _collect(generator) -> list:
    """Collect all results from a .map() generator."""
    return list(generator)


@app.local_entrypoint()
def main():
    pipeline_start = time.time()

    print("=" * 60)
    print("TADA Data Pipeline (Modal Serverless)")
    print("=" * 60)

    # ── Step 1: Download ─────────────────────────────────────────────────
    print("\n[step 1/6] Downloading from R2...")
    t0 = time.time()
    files = list_r2_files.remote()
    if not files:
        print("[error] No audio files in R2 — aborting.")
        return
    total_size_mb = sum(f["size"] for f in files) / 1024**2
    print(f"  {len(files)} files, {total_size_mb:.0f} MB")

    results = _collect(download_file.map(files, order_outputs=False))
    ok = sum(1 for r in results if r == "ok")
    skip = sum(1 for r in results if r == "skip")
    print(f"  {ok} downloaded, {skip} skipped  ({time.time() - t0:.0f}s)")

    # ── Step 2: Fused processing ─────────────────────────────────────────
    print("\n[step 2/6] Processing (VAD → ASR → features)...")
    t0 = time.time()
    mp3s = list_recordings.remote()
    if not mp3s:
        print("[error] No MP3 files on volume — aborting.")
        return

    done = list_done_recordings.remote()
    already_done_segments = sum(done.values())
    todo = [f for f in mp3s if f.split("_")[0] not in done]

    print(
        f"  {len(mp3s)} recordings: {len(done)} done ({already_done_segments} segments), "
        f"{len(todo)} to process"
    )

    new_segments = 0
    if todo:
        all_results = _collect(
            FusedWorker().process.map(todo, order_outputs=False)
        )
        new_segments = sum(len(batch) for batch in all_results)

    total_segments = already_done_segments + new_segments
    print(
        f"  {total_segments} total features ({new_segments} new)  "
        f"({time.time() - t0:.0f}s)"
    )

    if total_segments == 0:
        print("[error] No features produced — aborting.")
        return

    # ── Step 3: Quality filter ───────────────────────────────────────────
    print("\n[step 3/6] Quality filtering...")
    t0 = time.time()
    manifest = filter_and_build.remote()
    print(f"  {len(manifest)}/{total_segments} kept  ({time.time() - t0:.0f}s)")

    if not manifest:
        print("[error] No samples survived filtering — aborting.")
        return

    # ── Step 4: Split ────────────────────────────────────────────────────
    print("\n[step 4/6] Train/val split...")
    t0 = time.time()
    split_train_val.remote()
    print(f"  Done  ({time.time() - t0:.0f}s)")

    # ── Step 5: Verify ───────────────────────────────────────────────────
    print("\n[step 5/6] Final verification...")
    t0 = time.time()
    verify_final.remote()
    print(f"  Done  ({time.time() - t0:.0f}s)")

    # ── Step 6: Upload ───────────────────────────────────────────────────
    print("\n[step 6/6] Uploading to R2...")
    t0 = time.time()
    upload_files = list_upload_files.remote()
    if upload_files:
        results = _collect(
            upload_one_file.map(upload_files, order_outputs=False)
        )
        print(f"  {len(results)} files uploaded  ({time.time() - t0:.0f}s)")
    else:
        print("  Nothing to upload")

    # ── Summary ──────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!  Total: {_fmt_eta(total)}")
    print(f"{'=' * 60}")

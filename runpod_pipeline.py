#!/usr/bin/env python3
"""TADA Data Pipeline — RunPod multi-GPU version.

Replaces the Modal serverless pipeline with a single script optimized for
a multi-GPU RunPod pod. Each GPU runs an independent worker process that
loads models once and processes recordings end-to-end (download → VAD →
ASR → TADA features → save). Downloads and uploads are parallelized
with a thread pool.

Setup:
    # SSH into your RunPod pod, then:
    apt-get update && apt-get install -y ffmpeg libsndfile1 sox
    pip install torch==2.7.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt
    pip install --no-deps hume-tada==0.1.8 descript-audio-codec==1.0.0 descript-audiotools==0.7.2

    # Copy .env to the pod (or export the variables)
    # Required: R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET, HF_TOKEN

Run:
    python runpod_pipeline.py
"""

import functools
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue, SimpleQueue
from pathlib import Path
from typing import Optional

# ── Load .env if present ─────────────────────────────────────────────────────

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and val:
            os.environ.setdefault(key, val)

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24_000
MIN_DURATION_SEC = 5.0
TARGET_DURATION_SEC = 15.0
SOFT_MAX_DURATION_SEC = 20.0
HARD_MAX_DURATION_SEC = 30.0
FRAME_RATE = 50
BATCH_SIZE = 64
MAX_CONSECUTIVE_FRAMES = 3
MAX_POSITION_GAP_FRAMES = 150
DATASET_NAME = "r2_audio"
R2_PREFIX = "en-sample-20260322T191419Z-0amulojz"
TADA_CODEC_REPO = "HumeAI/tada-codec"
ENCODER_SUBFOLDER = "encoder"
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"


TRAIN_RATIO = 0.95
SPLIT_SEED = 42

DATA_ROOT = Path("/root/data")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"

# Download / upload parallelism
DOWNLOAD_WORKERS = 64
UPLOAD_WORKERS = 128

# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class _LinearETABar:
    """Progress bar with ETA = elapsed / fraction_done (linear extrapolation)."""

    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        from tqdm import tqdm

        self._t0 = time.time()
        self._bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        )
        self._bar.set_postfix_str("--")

    def update(self, n: int = 1):
        self._bar.update(n)
        done = self._bar.n
        total = self._bar.total
        if done > 0 and total > 0:
            elapsed = time.time() - self._t0
            frac = done / total
            eta = elapsed / frac - elapsed
            self._bar.set_postfix_str(f"ETA {_fmt_time(eta)}")

    def close(self):
        self._bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _get_s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
    )


def _check_quality(sample: dict) -> Optional[str]:
    """Return rejection reason or None if sample passes."""
    import torch

    positions = sample["positions"]
    duration_frames = sample["duration_frames"]
    L = positions.shape[0]

    if L == 0:
        return "empty_tokens"

    dur_sec = duration_frames / FRAME_RATE
    if dur_sec < MIN_DURATION_SEC:
        return "too_short"
    if dur_sec > HARD_MAX_DURATION_SEC:
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


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Download from R2 (threaded)
# ═════════════════════════════════════════════════════════════════════════════


def step_download() -> list[str]:
    """Download all audio files from R2 in parallel. Returns list of local paths."""
    print("\n[step 1/6] Downloading from R2...")
    t0 = time.time()

    s3 = _get_s3_client()
    bucket = os.environ["R2_BUCKET"]

    # List objects
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{R2_PREFIX}/audio/"):
        for obj in page.get("Contents", []):
            objects.append({"key": obj["Key"], "size": obj["Size"]})

    if not objects:
        print("  [error] No audio files found in R2")
        return []

    total_mb = sum(o["size"] for o in objects) / 1024**2
    print(f"  {len(objects)} files, {total_mb:.0f} MB")

    raw_dir = RAW_DIR / DATASET_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(obj: dict) -> str:
        key = obj["key"]
        filename = key.split("/")[-1]
        dest = raw_dir / filename
        if dest.exists() and dest.stat().st_size == obj["size"]:
            return "skip"
        # Each thread gets its own S3 client (boto3 clients aren't thread-safe)
        for attempt in range(3):
            try:
                client = _get_s3_client()
                client.download_file(bucket, key, str(dest))
                return "ok"
            except Exception as e:
                if attempt == 2:
                    print(f"  [error] Download failed after 3 attempts: {filename}: {e}")
                    return "error"
                time.sleep(1 * (attempt + 1))

    ok = skip = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_one, obj): obj for obj in objects}
        with _LinearETABar(total=len(objects), desc="  Downloading", unit="file") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result == "ok":
                    ok += 1
                else:
                    skip += 1
                pbar.update(1)

    print(f"  {ok} downloaded, {skip} skipped  ({_fmt_time(time.time() - t0)})")

    return sorted(str(p) for p in raw_dir.glob("*.mp3"))


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — GPU processing: VAD → ASR → TADA features (multi-GPU)
# ═════════════════════════════════════════════════════════════════════════════


def _gpu_worker(gpu_id: int, file_queue: SimpleQueue, result_queue: Queue):
    """Worker process: owns one GPU, processes recordings until queue is empty.

    Each worker loads all models onto its assigned GPU once, then pulls
    recordings from the shared queue. Results are sent back via result_queue.
    """
    import sys
    import traceback

    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    print(f"  [GPU {gpu_id}] Worker started (pid={os.getpid()})", flush=True)

    try:
        _gpu_worker_inner(gpu_id, file_queue, result_queue)
    except BaseException as e:
        traceback.print_exc()
        print(f"\n  [GPU {gpu_id}] FATAL ERROR: {e}", flush=True)
        result_queue.put(("done", None, 0))


def _gpu_worker_inner(gpu_id: int, file_queue: SimpleQueue, result_queue: Queue):
    import torch
    import torchaudio
    from tada.utils.text import normalize_text

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    # ── Load models ──────────────────────────────────────────────────────
    t_load = time.time()

    # Pyannote needs weights_only=False for its checkpoints
    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _permissive_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _permissive_load
    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        vad = PyannotePipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=os.environ["HF_TOKEN"],
        )
        vad.to(device)

        import nemo.collections.asr as nemo_asr

        asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        asr.eval()

        # DAC snake activation JIT fix
        import dac.nn.layers

        def _eager_snake(x, alpha):
            shape = x.shape
            x = x.reshape(shape[0], shape[1], -1)
            x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
            x = x.reshape(shape)
            return x

        dac.nn.layers.snake = _eager_snake

        from tada.modules.encoder import Encoder

        encoder = Encoder.from_pretrained(
            TADA_CODEC_REPO, subfolder=ENCODER_SUBFOLDER
        ).to(device)
        encoder.eval()
        tok = encoder.aligner.tokenizer
    finally:
        torch.load = _orig_load

    print(f"  [GPU {gpu_id}] Models loaded ({time.time() - t_load:.0f}s)")

    out_dir = FEATURES_DIR / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    # ── Process loop ─────────────────────────────────────────────────────
    while True:
        mp3_path_str = file_queue.get()
        if mp3_path_str is None:  # Sentinel — no more work
            break

        mp3_path = Path(mp3_path_str)
        mp3_filename = mp3_path.name
        recording_id = mp3_filename.split("_")[0]
        t_rec = time.time()

        # Resume check
        done_marker = out_dir / f"{recording_id}.done"
        if done_marker.exists():
            result_queue.put(("skip", recording_id, 0))
            continue

        # Clean partial previous run
        for stale in out_dir.glob(f"{recording_id}_*.pt"):
            stale.unlink()

        # ── 1. Load & resample ───────────────────────────────────────────
        try:
            waveform, sr = torchaudio.load(str(mp3_path))
        except Exception as e:
            print(f"  [GPU {gpu_id}] {recording_id}: load failed: {e}")
            done_marker.write_text(json.dumps({"n_segments": 0}))
            result_queue.put(("error", recording_id, 0))
            continue

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        audio_dur = waveform.shape[1] / sr

        # ── 2. VAD ───────────────────────────────────────────────────────
        vad_output = vad({"waveform": waveform, "sample_rate": sr})
        raw_segs = [(t.start, t.end) for t in vad_output.get_timeline()]
        if not raw_segs:
            done_marker.write_text(json.dumps({"n_segments": 0}))
            result_queue.put(("empty", recording_id, 0))
            continue

        # Step 1: Greedy accumulation (soft max = 20s)
        # Accumulate consecutive VAD intervals into segments targeting ~15s
        buffers: list[list[tuple[float, float]]] = [[raw_segs[0]]]
        for s, e in raw_segs[1:]:
            buf = buffers[-1]
            candidate_dur = e - buf[0][0]
            if candidate_dur <= SOFT_MAX_DURATION_SEC:
                buf.append((s, e))
            else:
                buffers.append([(s, e)])

        accumulated = [(buf[0][0], buf[-1][1]) for buf in buffers]

        # Step 2: Split segments exceeding hard max (30s)
        split_segs: list[tuple[float, float]] = []
        for s, e in accumulated:
            dur = e - s
            if dur <= HARD_MAX_DURATION_SEC:
                split_segs.append((s, e))
            else:
                n = max(1, round(dur / TARGET_DURATION_SEC))
                cd = dur / n
                for i in range(n):
                    split_segs.append((s + i * cd, s + (i + 1) * cd))

        # Step 3: Handle runts (< 5s) — merge with neighbor or discard
        final_segs: list[tuple[float, float]] = []
        for s, e in split_segs:
            dur = e - s
            if dur >= MIN_DURATION_SEC:
                final_segs.append((s, e))
            elif final_segs:
                # Try merging with previous segment
                ps, pe = final_segs[-1]
                if e - ps <= HARD_MAX_DURATION_SEC:
                    final_segs[-1] = (ps, e)
                # else discard
            # leading runt with no previous neighbor — discard

        # Second pass: merge trailing runts that couldn't merge forward
        # (a runt at position i that was kept, check if next seg can absorb it)
        cleaned: list[tuple[float, float]] = []
        for i, (s, e) in enumerate(final_segs):
            dur = e - s
            if dur < MIN_DURATION_SEC and cleaned:
                ps, pe = cleaned[-1]
                if e - ps <= HARD_MAX_DURATION_SEC:
                    cleaned[-1] = (ps, e)
                    continue
            cleaned.append((s, e))
        final_segs = [(s, e) for s, e in cleaned if e - s >= MIN_DURATION_SEC]

        if not final_segs:
            done_marker.write_text(json.dumps({"n_segments": 0}))
            result_queue.put(("empty", recording_id, 0))
            continue

        seg_durs = [e - s for s, e in final_segs]
        print(
            f"  [GPU {gpu_id}] {recording_id}: {len(seg_durs)} segs, "
            f"mean={sum(seg_durs)/len(seg_durs):.1f}s, "
            f"min={min(seg_durs):.1f}s, max={max(seg_durs):.1f}s, "
            f"total={sum(seg_durs):.1f}s"
        )

        # Slice waveforms
        segments: list[tuple[str, torch.Tensor, float]] = []
        for idx, (s, e) in enumerate(final_segs):
            ss = int(s * sr)
            es = min(int(e * sr), waveform.shape[1])
            seg_wav = waveform[:, ss:es]
            dur = seg_wav.shape[1] / sr
            if dur < MIN_DURATION_SEC:
                continue
            segments.append((f"{recording_id}_{idx:06d}", seg_wav, dur))

        if not segments:
            done_marker.write_text(json.dumps({"n_segments": 0}))
            result_queue.put(("empty", recording_id, 0))
            continue

        # ── 3. Transcribe ────────────────────────────────────────────────
        tmpdir = tempfile.mkdtemp()
        tmp_paths = []
        for sid, seg_wav, _ in segments:
            p = os.path.join(tmpdir, f"{sid}.wav")
            torchaudio.save(p, seg_wav, sr)
            tmp_paths.append(p)

        try:
            texts = asr.transcribe(
                tmp_paths, batch_size=min(len(tmp_paths), BATCH_SIZE)
            )
            if isinstance(texts, tuple):
                texts = texts[0]
            if hasattr(texts[0], "text"):
                texts = [t.text for t in texts]
        except Exception as e:
            print(f"  [GPU {gpu_id}] ASR batch {recording_id}: {e}; sequential fallback")
            texts = []
            for p in tmp_paths:
                try:
                    r = asr.transcribe([p], batch_size=1)
                    if isinstance(r, tuple):
                        r = r[0]
                    texts.append(r[0].text if hasattr(r[0], "text") else r[0])
                except Exception:
                    texts.append("")

        shutil.rmtree(tmpdir, ignore_errors=True)
        torch.cuda.empty_cache()

        # Pair with transcriptions, drop empty
        transcribed: list[tuple[str, torch.Tensor, float, str]] = []
        for (sid, wav, dur), text in zip(segments, texts):
            text = text.strip() if isinstance(text, str) else ""
            if text:
                transcribed.append((sid, wav, dur, text))

        if not transcribed:
            done_marker.write_text(json.dumps({"n_segments": 0}))
            result_queue.put(("empty", recording_id, 0))
            continue

        # ── 4. Feature extraction (batched) ──────────────────────────────
        featured: list[tuple[str, torch.Tensor, float, dict]] = []

        for b_start in range(0, len(transcribed), BATCH_SIZE):
            batch = transcribed[b_start : b_start + BATCH_SIZE]

            wavs = [wav.squeeze(0) for _, wav, _, _ in batch]
            txts = [text for _, _, _, text in batch]
            a_lens = torch.tensor([w.shape[0] for w in wavs])

            max_len = int(a_lens.max().item())
            padded = torch.zeros(len(wavs), max_len)
            for j, w in enumerate(wavs):
                padded[j, : w.shape[0]] = w
            padded = padded.to(device)
            a_lens_dev = a_lens.to(device).unsqueeze(1)

            normed = [normalize_text(t) for t in txts]
            tseqs = [
                tok.encode(t, add_special_tokens=False, return_tensors="pt").squeeze(0)
                for t in normed
            ]
            ttl = torch.tensor([t.shape[0] for t in tseqs], device=device)
            tt = torch.nn.utils.rnn.pad_sequence(
                tseqs, batch_first=True, padding_value=tok.eos_token_id
            ).to(device)

            batch_ok = True
            try:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    enc_out = encoder(
                        padded,
                        text_tokens=tt,
                        text_token_len=ttl,
                        audio_length=a_lens_dev,
                        sample_rate=sr,
                        sample=False,
                    )
                torch.cuda.synchronize()
                for j, (sid, wav, dur, _text) in enumerate(batch):
                    try:
                        sd = _extract_cpu(enc_out, j, int(a_lens[j].item()), sr)
                        featured.append((sid, wav, dur, sd))
                    except Exception as e:
                        print(f"  [GPU {gpu_id}] extract {sid}: {e}")
            except Exception as e:
                print(f"  [GPU {gpu_id}] Feature batch failed ({e}), sequential fallback")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                batch_ok = False

            if not batch_ok:
                for sid, wav, dur, text in batch:
                    try:
                        with torch.no_grad(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            enc_out = encoder(
                                wav.to(device),
                                text=[text],
                                sample_rate=sr,
                                sample=False,
                            )
                        torch.cuda.synchronize()
                        sd = _extract_cpu(enc_out, 0, wav.shape[1], sr)
                        featured.append((sid, wav, dur, sd))
                    except Exception as e2:
                        print(f"  [GPU {gpu_id}] {sid}: {e2}")
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

        if not featured:
            done_marker.write_text(json.dumps({"n_segments": 0}))
            result_queue.put(("empty", recording_id, 0))
            continue

        # ── 5. Save .pt ──────────────────────────────────────────────────
        n_saved = 0
        for sid, _wav, _dur, sd in featured:
            pt_path = out_dir / f"{sid}.pt"
            torch.save(sd, pt_path)
            n_saved += 1

        done_marker.write_text(json.dumps({"n_segments": n_saved}))
        elapsed = time.time() - t_rec
        processed_count += 1
        print(
            f"  [GPU {gpu_id}] {recording_id}: {audio_dur:.0f}s audio → "
            f"{n_saved} features  ({elapsed:.1f}s)"
        )
        result_queue.put(("ok", recording_id, n_saved))

    result_queue.put(("done", f"gpu_{gpu_id}", processed_count))


def _count_gpus() -> int:
    """Count available NVIDIA GPUs via NVML without initializing CUDA runtime.

    This is critical: if CUDA runtime is initialized before fork(), child
    processes inherit a corrupted CUDA context and CUDA_VISIBLE_DEVICES
    set in the child is silently ignored.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return len(out.strip().splitlines())
    except Exception:
        return 0


def step_process(mp3_paths: list[str]) -> int:
    """Distribute recordings across all available GPUs. Returns total features."""
    print("\n[step 2/6] Processing (VAD → ASR → features)...")
    t0 = time.time()

    n_gpus = _count_gpus()
    if n_gpus == 0:
        print("  [error] No GPUs available")
        return 0

    # Check which recordings are already done
    out_dir = FEATURES_DIR / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    already_done_segments = 0
    for p in out_dir.glob("*.done"):
        try:
            data = json.loads(p.read_text())
            done.add(p.stem)
            already_done_segments += data.get("n_segments", 0)
        except Exception:
            pass

    todo = [p for p in mp3_paths if Path(p).name.split("_")[0] not in done]
    print(
        f"  {len(mp3_paths)} recordings: {len(done)} done "
        f"({already_done_segments} segments), {len(todo)} to process on {n_gpus} GPU(s)"
    )

    if not todo:
        return already_done_segments

    # Fill work queue
    file_queue: SimpleQueue = SimpleQueue()
    for path in todo:
        file_queue.put(path)
    # Add sentinels (one per GPU)
    for _ in range(n_gpus):
        file_queue.put(None)

    result_queue: Queue = Queue()

    # Launch one process per GPU
    workers = []
    for gpu_id in range(n_gpus):
        p = Process(target=_gpu_worker, args=(gpu_id, file_queue, result_queue))
        p.start()
        workers.append(p)

    # Collect results with progress bar
    new_segments = 0
    finished_workers = 0
    with _LinearETABar(total=len(todo), desc="  Processing", unit="rec") as pbar:
        while finished_workers < n_gpus:
            status, rec_id, n = result_queue.get()
            if status == "done":
                finished_workers += 1
            else:
                pbar.update(1)
                if status == "ok":
                    new_segments += n

    for w in workers:
        w.join()

    total = already_done_segments + new_segments
    print(
        f"  {total} total features ({new_segments} new)  "
        f"({_fmt_time(time.time() - t0)})"
    )
    return total


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Quality filter & manifest build
# ═════════════════════════════════════════════════════════════════════════════


def _filter_one_file(pt_path_str: str) -> tuple[Optional[dict], Optional[str]]:
    """Filter a single .pt file. Returns (manifest_entry_or_None, rejection_reason_or_None)."""
    import torch

    pt_path = Path(pt_path_str)
    sid = pt_path.stem
    if sid.startswith("_") or sid.endswith(".done"):
        return None, None

    try:
        sample = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception:
        return None, "load_error"

    reason = _check_quality(sample)
    if reason:
        return None, reason

    entry = {
        "segment_id": sid,
        "dataset": DATASET_NAME,
        "num_tokens": int(sample["token_ids"].shape[0]),
        "duration_frames": int(sample["duration_frames"]),
        "feature_path": pt_path_str,
    }
    return entry, None


def step_filter() -> list[dict]:
    """Scan .pt files, quality-filter, write manifest.json. Uses all CPU cores."""
    from multiprocessing import Pool

    print("\n[step 3/6] Quality filtering...")
    t0 = time.time()

    feat_dir = FEATURES_DIR / DATASET_NAME
    if not feat_dir.exists():
        print("  [error] Feature directory missing")
        return []

    pt_files = sorted(str(p) for p in feat_dir.glob("*.pt"))
    print(f"  {len(pt_files)} feature files to check")

    manifest: list[dict] = []
    stats: Counter = Counter()

    n_workers = min(os.cpu_count() or 1, len(pt_files))
    pbar = _LinearETABar(total=len(pt_files), desc="  Filtering", unit="file")
    with Pool(processes=n_workers) as pool:
        for entry, reason in pool.imap_unordered(_filter_one_file, pt_files, chunksize=64):
            if entry is not None:
                manifest.append(entry)
                stats["kept"] += 1
            elif reason is not None:
                stats[reason] += 1
            pbar.update(1)
    pbar.close()

    # Sort by segment_id to restore temporal order within each recording
    manifest.sort(key=lambda e: e["segment_id"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(stats.values())
    kept_pct = stats["kept"] / total * 100 if total else 0
    total_hours = sum(e["duration_frames"] for e in manifest) / FRAME_RATE / 3600
    total_tokens = sum(e["num_tokens"] for e in manifest)

    print(f"  {total} processed in {time.time() - t0:.1f}s:")
    for reason, count in stats.most_common():
        pct = count / total * 100
        print(f"    {reason}: {count} ({pct:.1f}%)")
    print(f"  {len(manifest)} samples ({kept_pct:.0f}% kept)")
    print(f"  ~{total_hours:.1f} hours, {total_tokens:,} tokens")
    return manifest


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Train / val split
# ═════════════════════════════════════════════════════════════════════════════


def step_split():
    """Split manifest by recording → manifest_train.json + manifest_val.json."""
    print("\n[step 4/6] Train/val split...")
    t0 = time.time()

    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        print("  [error] manifest.json not found")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  {len(manifest)} samples (ratio {TRAIN_RATIO:.0%} train)")

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
        print(f"  {len(data)} samples → {path}")

    train_h = sum(e.get("duration_frames", 0) for e in train_manifest) / FRAME_RATE / 3600
    val_h = sum(e.get("duration_frames", 0) for e in val_manifest) / FRAME_RATE / 3600
    print(f"  Train: ~{train_h:.1f}h ({len(train_recs)} recs), Val: ~{val_h:.1f}h")
    print(f"  Done  ({_fmt_time(time.time() - t0)})")


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Final verification
# ═════════════════════════════════════════════════════════════════════════════


def step_verify():
    """Check keys, shapes, and recording disjointness."""
    import torch

    print("\n[step 5/6] Final verification...")
    t0 = time.time()

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
            print(f"  [skip] {path} not found")
            continue
        with open(path) as f:
            manifest = json.load(f)
        if not manifest:
            continue

        rng = random.Random(123)
        samples = rng.sample(manifest, min(5, len(manifest)))
        keys_ok = True
        print(f"\n  === Keys & Shapes: {split_name} ===")
        for entry in samples:
            sid = entry["segment_id"]
            pt_path = Path(entry["feature_path"])
            if not pt_path.exists():
                print(f"    [FAIL] {sid}: file not found")
                keys_ok = False
                continue
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            missing = [k for k in REQUIRED_KEYS if k not in data]
            if missing:
                print(f"    [FAIL] {sid}: missing {missing}")
                keys_ok = False
                continue
            L = data["token_ids"].shape[0]
            errors = []
            if data["encoder_features"].shape != (L, 512):
                errors.append(f"encoder_features {data['encoder_features'].shape}")
            if errors:
                print(f"    [FAIL] {sid}: {'; '.join(errors)}")
                keys_ok = False
            else:
                print(f"    [OK]   {sid}: L={L}")
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
        print(f"\n  === Disjointness ===")
        print(
            f"    Train: {len(train_recs)} recs, Val: {len(val_recs)} recs, "
            f"Overlap: {len(overlap)}"
        )

    print(f"\n  === Results ===")
    all_pass = True
    for check, passed in results.items():
        print(f"    {check}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
    print(f"    Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"  Done  ({_fmt_time(time.time() - t0)})")


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — Upload results to R2 (threaded)
# ═════════════════════════════════════════════════════════════════════════════


def step_upload():
    """Upload manifests and feature files to R2 in parallel."""
    print("\n[step 6/6] Uploading to R2...")
    t0 = time.time()

    bucket = os.environ["R2_BUCKET"]
    output_prefix = f"{R2_PREFIX}/processed"

    files: list[dict] = []
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

    if not files:
        print("  Nothing to upload")
        return

    # Build index of existing remote objects for resume support
    s3 = _get_s3_client()
    remote_sizes: dict[str, int] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix + "/"):
        for obj in page.get("Contents", []):
            remote_sizes[obj["Key"]] = obj["Size"]

    # Skip files already uploaded with matching size
    todo: list[dict] = []
    skipped = 0
    for f in files:
        local_size = Path(f["local"]).stat().st_size
        if remote_sizes.get(f["key"]) == local_size:
            skipped += 1
        else:
            todo.append(f)

    print(
        f"  {len(files)} files: {skipped} already uploaded, {len(todo)} to upload"
    )

    if not todo:
        print(f"  Nothing new to upload  ({_fmt_time(time.time() - t0)})")
        return

    def _upload_one(file_info: dict) -> str:
        for attempt in range(3):
            try:
                client = _get_s3_client()
                client.upload_file(file_info["local"], bucket, file_info["key"])
                return "ok"
            except Exception as e:
                if attempt == 2:
                    print(f"  [error] Upload failed after 3 attempts: {file_info['key']}: {e}")
                    return "error"
                time.sleep(1 * (attempt + 1))

    uploaded = 0
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        futures = {pool.submit(_upload_one, f): f for f in todo}
        with _LinearETABar(total=len(todo), desc="  Uploading", unit="file") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result == "ok":
                    uploaded += 1
                pbar.update(1)

    print(
        f"  {uploaded} uploaded, {skipped} skipped  ({_fmt_time(time.time() - t0)})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════


def main():
    pipeline_start = time.time()

    print("=" * 60)
    print("TADA Data Pipeline (RunPod multi-GPU)")
    print("=" * 60)

    # Step 1: Download
    mp3_paths = step_download()
    if not mp3_paths:
        print("[error] No audio files — aborting.")
        return

    # Step 2: GPU processing (multi-GPU parallel)
    total_features = step_process(mp3_paths)
    if total_features == 0:
        print("[error] No features produced — aborting.")
        return

    # Step 3: Quality filter
    manifest = step_filter()
    if not manifest:
        print("[error] No samples survived filtering — aborting.")
        return

    # Step 4: Split
    step_split()

    # Step 5: Verify
    step_verify()

    # Step 6: Upload
    step_upload()

    # Summary
    total = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!  Total: {_fmt_time(total)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()

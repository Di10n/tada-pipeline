#!/usr/bin/env python3
"""Step 1: Download LibriTTS-R dataset.

Downloads and extractions run in parallel using a thread pool.
Each split is an independent (download → extract → cleanup) task.
"""

import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import RAW_DIR, LIBRITTS_R

LIBRITTS_R_BASE = "https://www.openslr.org/resources/141"
LIBRITTS_R_SPLITS = [
    "train_clean_100",
    "train_clean_360",
    "train_other_500",
]


def download_file(url: str, dest: Path) -> Path:
    """Download a file with wget (resume-capable)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", url, "-O", str(dest)],
        check=True,
    )
    return dest


def _download_and_extract_libritts_split(split: str) -> str:
    """Download and extract a single LibriTTS-R split. Returns status string."""
    out_dir = RAW_DIR / LIBRITTS_R
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / split).exists():
        return f"[skip] {split} already extracted"

    tar_name = f"{split}.tar.gz"
    url = f"{LIBRITTS_R_BASE}/{tar_name}"
    tar_path = RAW_DIR / tar_name

    print(f"[download] {url}")
    download_file(url, tar_path)

    print(f"[extract] {tar_name} → {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)

    tar_path.unlink()
    return f"[done] {split}"


def download_all_parallel():
    """Run all downloads in parallel using a thread pool."""
    tasks = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for split in LIBRITTS_R_SPLITS:
            fut = pool.submit(_download_and_extract_libritts_split, split)
            tasks[fut] = f"libritts_r/{split}"

        for fut in as_completed(tasks):
            name = tasks[fut]
            try:
                result = fut.result()
                print(f"  {name}: {result}")
            except Exception as e:
                print(f"  {name}: [FAILED] {e}")
                raise


if __name__ == "__main__":
    download_all_parallel()
    print("[done] Downloads complete.")

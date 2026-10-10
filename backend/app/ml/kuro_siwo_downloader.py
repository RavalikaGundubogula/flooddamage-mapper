import io
import json
import logging
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/resolve/main"
REQUIRED_SAMPLE_FILES = [
    "flood_vv.npy",
    "flood_vh.npy",
    "sec1_vv.npy",
    "sec1_vh.npy",
    "mask.npy",
    "info.json",
]


def download_official_kuro_siwo_partition(
    target_dir: Path,
    split: str = "train_GRD",
    shard_name: str = "shard-00000.tar",
    max_bytes: int = 80 * 1024 * 1024,  # ~80 MB default partition per shard
) -> int:
    """
    Download and extract real official Kuro Siwo multi-temporal SAR samples
    from the official Hugging Face repository (orion-ai-lab/Kuro-Siwo-Webdataset).
    
    Args:
        target_dir: Root dataset directory (e.g. backend/data/datasets/kuro_siwo).
        split: 'train_GRD' or 'test_GRD'.
        shard_name: Specific shard tar archive.
        max_bytes: Byte range to fetch via HTTP 206 Partial Content.
        
    Returns:
        Number of verified complete samples successfully saved.
    """
    url = f"{HF_BASE_URL}/{split}/{shard_name}"
    headers = {
        "User-Agent": "FloodDamageMapper/1.0 (EarthObservation; Research)",
        "Range": f"bytes=0-{max_bytes}",
    }
    req = urllib.request.Request(url, headers=headers)
    
    logger.info("Downloading official Kuro Siwo %s from %s (up to %d MB)...", split, url, max_bytes // (1024 * 1024))
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
    except Exception as e:
        logger.error("Failed to download from %s: %s", url, e)
        raise RuntimeError(f"Could not connect to official Kuro Siwo repository at {url}: {e}")

    logger.info("Received %.2f MB of raw Kuro Siwo tar data. Extracting samples...", len(data) / (1024 * 1024))
    bio = io.BytesIO(data)
    
    try:
        tf = tarfile.TarFile.open(fileobj=bio, mode="r|*")
    except Exception as e:
        raise RuntimeError(f"Failed to open downloaded Kuro Siwo tar archive: {e}")

    extracted_files: Dict[str, bytes] = {}
    try:
        for m in tf:
            if not m.isfile():
                continue
            extracted_files[m.name] = tf.extractfile(m).read()
    except tarfile.ReadError:
        # Expected for partial range stream boundary
        pass

    out_split_dir = target_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # Group by sample ID prefix
    sample_prefixes = sorted(list(set(k.split(".")[0] for k in extracted_files.keys())))
    saved_count = 0

    for prefix in sample_prefixes:
        # Verify all required arrays exist for this sample
        has_all_required = all(f"{prefix}.{req_f}" in extracted_files for req_f in REQUIRED_SAMPLE_FILES)
        if not has_all_required:
            continue

        for k, content in extracted_files.items():
            if k.startswith(f"{prefix}."):
                out_file = out_split_dir / k
                with open(out_file, "wb") as f:
                    f.write(content)
        saved_count += 1

    logger.info("Extracted %d complete real Kuro Siwo samples into %s", saved_count, out_split_dir)
    return saved_count


def acquire_real_kuro_siwo_dataset(
    target_dir: Path,
    train_bytes: int = 80 * 1024 * 1024,  # ~80 MB (~40-45 real samples)
    test_bytes: int = 40 * 1024 * 1024,   # ~40 MB (~20-25 real samples)
) -> Tuple[int, int]:
    """
    Acquire real official Kuro Siwo training and testing splits.
    Returns (train_samples_count, test_samples_count).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    train_count = download_official_kuro_siwo_partition(
        target_dir=target_dir,
        split="train_GRD",
        shard_name="shard-00000.tar",
        max_bytes=train_bytes,
    )

    test_count = download_official_kuro_siwo_partition(
        target_dir=target_dir,
        split="test_GRD",
        shard_name="shard-00000.tar",
        max_bytes=test_bytes,
    )

    return train_count, test_count

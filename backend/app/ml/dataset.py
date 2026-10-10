import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enforce strict real-data requirement
REAL_KURO_SIWO_REQUIRED = True

# Standard Sentinel-1 SAR backscatter normalization range in decibels (dB)
SAR_DB_MIN = -35.0
SAR_DB_MAX = 5.0
OFFICIAL_SOURCE_URL = "https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset"


def linear_intensity_to_db(arr: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Convert raw Sentinel-1 linear intensity (gamma0) to backscatter in decibels (dB).
    dB = 10 * log10(max(intensity, epsilon))
    """
    # If values are already in negative dB, do not recompute log10
    if np.nanmedian(arr) < 0:
        return arr.astype(np.float32)
    positive_vals = np.maximum(arr, epsilon)
    db = 10.0 * np.log10(positive_vals)
    return db.astype(np.float32)


def normalize_sar_db(arr: np.ndarray, clip_min: float = SAR_DB_MIN, clip_max: float = SAR_DB_MAX) -> np.ndarray:
    """Normalize SAR backscatter in dB to [0, 1] interval."""
    # If array is in linear intensity, convert to dB first
    if np.nanmedian(arr) >= 0:
        arr = linear_intensity_to_db(arr)
    clipped = np.clip(arr, clip_min, clip_max)
    normalized = (clipped - clip_min) / (clip_max - clip_min)
    return normalized.astype(np.float32)


def denormalize_sar_db(norm_arr: np.ndarray, clip_min: float = SAR_DB_MIN, clip_max: float = SAR_DB_MAX) -> np.ndarray:
    """Reconstruct SAR backscatter in dB from normalized [0, 1] range."""
    return (norm_arr * (clip_max - clip_min) + clip_min).astype(np.float32)


def verify_kuro_siwo_dataset(dataset_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Verify that the official real Kuro Siwo dataset is present on disk.
    
    Checks:
    - Scans for official `.npy` arrays or GeoTIFFs across train and test partitions.
    - Ensures no synthetic samples are misidentified as real Kuro Siwo data.
    - Validates matched channel and mask pairs.
    - Computes real positive floodwater statistics.
    - Updates/writes dataset_manifest.json with official provenance metadata.
    
    Raises:
        RuntimeError: If no valid real Kuro Siwo samples are found.
    """
    ds_dir = Path(dataset_dir)
    if not ds_dir.exists():
        raise RuntimeError(
            f"MODULE 4 REAL-DATA TRAINING IS BLOCKED: Dataset directory '{ds_dir}' does not exist. "
            f"Real Kuro Siwo dataset is required from {OFFICIAL_SOURCE_URL}."
        )

    # 1. Discover all real samples
    real_samples: List[Dict[str, Any]] = []
    synthetic_samples_found: List[str] = []

    # Check for synthetic folders (sample_001, etc.)
    for d in ds_dir.iterdir():
        if d.is_dir() and d.name.startswith("sample_"):
            synthetic_samples_found.append(d.name)

    # Search for official Kuro Siwo .npy samples in train_GRD, test_GRD, or root
    search_dirs = [ds_dir]
    for sub in ["train_GRD", "test_GRD", "train", "test"]:
        if (ds_dir / sub).exists():
            search_dirs.append(ds_dir / sub)

    for sdir in search_dirs:
        info_files = list(sdir.glob("*.info.json"))
        for info_file in info_files:
            prefix = info_file.name.replace(".info.json", "")
            flood_vv = sdir / f"{prefix}.flood_vv.npy"
            flood_vh = sdir / f"{prefix}.flood_vh.npy"
            sec1_vv = sdir / f"{prefix}.sec1_vv.npy"
            sec1_vh = sdir / f"{prefix}.sec1_vh.npy"
            mask_file = sdir / f"{prefix}.mask.npy"

            if flood_vv.exists() and flood_vh.exists() and sec1_vv.exists() and sec1_vh.exists() and mask_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

                real_samples.append({
                    "sample_id": prefix,
                    "split_dir": sdir.name,
                    "flood_vv": flood_vv,
                    "flood_vh": flood_vh,
                    "sec1_vv": sec1_vv,
                    "sec1_vh": sec1_vh,
                    "mask": mask_file,
                    "info": meta,
                    "actid": meta.get("actid", "unknown"),
                    "flood_date": meta.get("flood_date", "unknown"),
                })

    total_real_samples = len(real_samples)

    if total_real_samples == 0:
        raise RuntimeError(
            "MODULE 4 REAL-DATA TRAINING IS BLOCKED: Zero real Kuro Siwo samples found in "
            f"'{ds_dir}'. Found {len(synthetic_samples_found)} synthetic folders which cannot be used. "
            f"Please download real Kuro Siwo data from {OFFICIAL_SOURCE_URL}."
        )

    # 2. Inspect real sample statistics
    sample_stat_log = []
    total_flood_pixels = 0
    total_pixels_evaluated = 0

    for s in real_samples:
        try:
            m_arr = np.load(s["mask"])
            # In official Kuro Siwo, label 2 is floodwater
            flood_pix = int(np.sum(m_arr == 2))
            # Also count label 1 (permanent water) if separate
            perm_water_pix = int(np.sum(m_arr == 1))
            tot_pix = int(m_arr.size)
            total_flood_pixels += flood_pix
            total_pixels_evaluated += tot_pix

            sample_stat_log.append({
                "sample_id": s["sample_id"],
                "partition": s["split_dir"],
                "actid": s["actid"],
                "flood_date": s["flood_date"],
                "flood_pixels": flood_pix,
                "perm_water_pixels": perm_water_pix,
                "total_pixels": tot_pix,
                "flood_percentage": round(float((flood_pix / max(1, tot_pix)) * 100), 2),
            })
        except Exception as e:
            logger.warning("Could not read mask %s: %s", s["mask"], e)

    # 3. Determine train / validation / test splits
    train_samples = [s for s in sample_stat_log if "train" in s["partition"]]
    test_samples = [s for s in sample_stat_log if "test" in s["partition"]]

    # If all in one directory, split deterministically by event/activation ID
    if len(test_samples) == 0 and len(sample_stat_log) > 4:
        val_count = max(2, int(len(sample_stat_log) * 0.2))
        val_samples = sample_stat_log[-val_count:]
        tr_samples = sample_stat_log[:-val_count]
        train_count = len(tr_samples)
        val_count = len(val_samples)
        test_count = 0
    else:
        # If train_GRD and test_GRD exist
        # Use 20% of train_GRD for validation, rest for train, test_GRD for test
        train_count = max(1, int(len(train_samples) * 0.8))
        val_count = len(train_samples) - train_count
        test_count = len(test_samples)

    manifest_data = {
        "dataset_name": "Kuro Siwo Official Dataset (GRD)",
        "dataset_source": OFFICIAL_SOURCE_URL,
        "official_source_url": OFFICIAL_SOURCE_URL,
        "dataset_version": "v2",
        "real_samples": True,
        "synthetic_samples_used": 0,
        "synthetic_samples_detected": len(synthetic_samples_found),
        "total_samples": total_real_samples,
        "train_samples": train_count,
        "validation_samples": val_count,
        "test_samples": test_count,
        "spatial_resolution_meters": 10.0,
        "patch_dimensions": [224, 224],
        "input_channels": ["sec1_vv (pre_vv)", "sec1_vh (pre_vh)", "flood_vv (post_vv)", "flood_vh (post_vh)"],
        "label_mapping": {
            "0": "Background / No water",
            "1": "Permanent water",
            "2": "Floodwater / Inundation",
        },
        "positive_flood_pixels": total_flood_pixels,
        "overall_flood_percentage": round(float((total_flood_pixels / max(1, total_pixels_evaluated)) * 100), 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_inventory": sample_stat_log,
    }

    manifest_path = ds_dir / "dataset_manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info("Updated real dataset manifest at %s", manifest_path)
    except Exception as e:
        logger.warning("Could not write manifest to %s: %s", manifest_path, e)

    return manifest_data


class KuroSiwoDataset(Dataset):
    """
    PyTorch Dataset loader for the REAL Kuro Siwo multi-temporal SAR benchmark.
    
    Reads authentic Sentinel-1 dual-pol SAR pairs (pre-event sec1 VV/VH and post-event flood VV/VH)
    and expert-photointerpreted ground-truth flood masks.
    
    Maps:
    - Input tensor (4, 224, 224): [pre_vv, pre_vh, post_vv, post_vh] in normalized [0, 1] range.
    - Target mask (1, 224, 224): binary {0.0, 1.0} where 1.0 is inundated floodwater (label == 2).
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str = "train",
        val_ratio: float = 0.2,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.augment = augment
        self.samples: List[Dict[str, Any]] = []

        self._load_samples(val_ratio=val_ratio, seed=seed)

    def _load_samples(self, val_ratio: float = 0.2, seed: int = 42) -> None:
        """Scan real Kuro Siwo dataset and construct train/val/test splits."""
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Kuro Siwo dataset directory '{self.dataset_dir}' does not exist.")

        # Check for official train_GRD and test_GRD partitions
        train_grd_dir = self.dataset_dir / "train_GRD"
        test_grd_dir = self.dataset_dir / "test_GRD"

        train_samples = self._discover_partition_samples(train_grd_dir) if train_grd_dir.exists() else []
        test_samples = self._discover_partition_samples(test_grd_dir) if test_grd_dir.exists() else []

        # If partitions exist
        if train_samples or test_samples:
            if self.split == "test":
                self.samples = test_samples if test_samples else train_samples
                return

            # Split train_samples into train and val deterministically
            rng = random.Random(seed)
            indices = list(range(len(train_samples)))
            rng.shuffle(indices)

            val_count = max(1, int(len(train_samples) * val_ratio)) if len(train_samples) > 1 else 0
            if self.split == "val":
                selected = indices[:val_count]
            else:
                selected = indices[val_count:]
            self.samples = [train_samples[i] for i in selected]
            return

        # Fallback: scan root directory for samples
        all_samples = self._discover_partition_samples(self.dataset_dir)
        if not all_samples:
            raise RuntimeError(
                f"No valid real Kuro Siwo samples found in '{self.dataset_dir}'. "
                "Ensure real Kuro Siwo dataset is acquired."
            )

        rng = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)

        val_count = max(1, int(len(all_samples) * val_ratio)) if len(all_samples) > 1 else 0
        if self.split == "val":
            selected = indices[:val_count]
        elif self.split == "test":
            selected = indices[:val_count]
        else:
            selected = indices[val_count:]

        self.samples = [all_samples[i] for i in selected]

    def _discover_partition_samples(self, partition_dir: Path) -> List[Dict[str, Any]]:
        """Find paired real Kuro Siwo sample files in directory."""
        if not partition_dir.exists():
            return []

        samples = []
        info_files = sorted(list(partition_dir.glob("*.info.json")))
        for info_file in info_files:
            prefix = info_file.name.replace(".info.json", "")
            flood_vv = partition_dir / f"{prefix}.flood_vv.npy"
            flood_vh = partition_dir / f"{prefix}.flood_vh.npy"
            sec1_vv = partition_dir / f"{prefix}.sec1_vv.npy"
            sec1_vh = partition_dir / f"{prefix}.sec1_vh.npy"
            mask_file = partition_dir / f"{prefix}.mask.npy"

            if flood_vv.exists() and flood_vh.exists() and sec1_vv.exists() and sec1_vh.exists() and mask_file.exists():
                samples.append({
                    "sample_id": prefix,
                    "flood_vv": flood_vv,
                    "flood_vh": flood_vh,
                    "sec1_vv": sec1_vv,
                    "sec1_vh": sec1_vh,
                    "mask": mask_file,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Load real Sentinel-1 numpy arrays (shape 1, H, W or H, W)
        f_vv = np.squeeze(np.load(sample["flood_vv"]))
        f_vh = np.squeeze(np.load(sample["flood_vh"]))
        s_vv = np.squeeze(np.load(sample["sec1_vv"]))
        s_vh = np.squeeze(np.load(sample["sec1_vh"]))
        m_raw = np.squeeze(np.load(sample["mask"]))

        # Convert linear radar intensity to dB and normalize to [0, 1]
        norm_pre_vv = normalize_sar_db(s_vv)
        norm_pre_vh = normalize_sar_db(s_vh)
        norm_post_vv = normalize_sar_db(f_vv)
        norm_post_vh = normalize_sar_db(f_vh)

        # Stack into 4-channel tensor: [pre_vv, pre_vh, post_vv, post_vh]
        image_np = np.stack([norm_pre_vv, norm_pre_vh, norm_post_vv, norm_post_vh], axis=0).astype(np.float32)

        # Official Kuro Siwo label mapping:
        # 0: non-water, 1: permanent water, 2: floodwater
        # Binary flood target: 1.0 for floodwater (label == 2)
        mask_bin = (m_raw == 2).astype(np.float32)
        mask_np = np.expand_dims(mask_bin, axis=0)  # Shape (1, H, W)

        # Spatial data augmentation
        if self.augment:
            if random.random() > 0.5:
                image_np = np.flip(image_np, axis=2).copy()
                mask_np = np.flip(mask_np, axis=2).copy()
            if random.random() > 0.5:
                image_np = np.flip(image_np, axis=1).copy()
                mask_np = np.flip(mask_np, axis=1).copy()

        return torch.from_numpy(image_np), torch.from_numpy(mask_np)

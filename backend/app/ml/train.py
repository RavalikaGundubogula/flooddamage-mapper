import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from app.ml.dataset import (
        KuroSiwoDataset,
        OFFICIAL_SOURCE_URL,
        REAL_KURO_SIWO_REQUIRED,
        verify_kuro_siwo_dataset,
    )
    from app.ml.loss import CombinedSegmentationLoss
    from app.ml.metrics import evaluate_segmentation
    from app.ml.model import FloodUNet
    TORCH_READY = True
except ImportError:
    TORCH_READY = False

from app.core.config import Settings, get_settings
from app.schemas.segmentation import SegmentationMetrics, TrainModelResponse

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """
    Supervised Semantic Segmentation Trainer for Sentinel-1 / REAL Kuro Siwo Flood Mapping.
    
    Strictly trained on real Kuro Siwo benchmark data (no synthetic fallback).
    
    Manages:
    - Real dataset verification and inventory validation
    - Train / Validation / Test splits
    - BCEWithLogits + Dice Loss optimization
    - Evaluation of IoU, Dice, Precision, Recall metrics
    - Checkpoint provenance recording (real_samples=True, synthetic_samples_used=0)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "flood_unet_s1",
        dataset_dir: Optional[Union[str, Path]] = None,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.seed = seed

        # Resolve dataset directory
        if dataset_dir:
            self.dataset_dir = Path(dataset_dir)
        else:
            self.dataset_dir = self.settings.get_data_path(self.settings.KUROSIWO_DATA_DIR)

        # Output models directory
        self.models_dir = self.settings.get_data_path(self.settings.MODEL_ARTIFACTS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if not TORCH_READY:
            raise RuntimeError("PyTorch is required for SegmentationTrainer but is not installed.")

        # Determine compute device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        logger.info("SegmentationTrainer initialized on device: %s", self.device)

    def _prepare_data(self) -> Tuple[KuroSiwoDataset, KuroSiwoDataset, Dict[str, Any]]:
        """
        Verify real Kuro Siwo dataset and construct train and validation datasets.
        Fails loudly if real Kuro Siwo samples are unavailable.
        """
        # 1. Strong real dataset verification
        manifest = verify_kuro_siwo_dataset(self.dataset_dir)
        total_real = manifest.get("total_samples", 0)

        if total_real == 0:
            raise RuntimeError(
                f"MODULE 4 REAL-DATA TRAINING IS BLOCKED: Zero real Kuro Siwo samples available in '{self.dataset_dir}'. "
                "Synthetic data generation is strictly disabled. Download real data from "
                f"{OFFICIAL_SOURCE_URL} before training."
            )

        logger.info(
            "Verified REAL Kuro Siwo dataset at %s: %d total real samples (%d train, %d val, %d test)",
            self.dataset_dir,
            total_real,
            manifest.get("train_samples", 0),
            manifest.get("validation_samples", 0),
            manifest.get("test_samples", 0),
        )

        train_ds = KuroSiwoDataset(self.dataset_dir, split="train", val_ratio=0.2, augment=True, seed=self.seed)
        val_ds = KuroSiwoDataset(self.dataset_dir, split="val", val_ratio=0.2, augment=False, seed=self.seed)

        if len(train_ds) == 0:
            raise RuntimeError(f"Real training set is empty in '{self.dataset_dir}'.")
        if len(val_ds) == 0:
            val_ds = train_ds

        return train_ds, val_ds, manifest

    def train(self) -> TrainModelResponse:
        """
        Execute training and validation loop on REAL Kuro Siwo data.
        Saves best checkpoint and provenance sidecar.
        """
        start_time = time.time()
        train_ds, val_ds, manifest = self._prepare_data()

        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=min(self.batch_size, len(val_ds)), shuffle=False)

        # Initialize FloodUNet
        model = FloodUNet(in_channels=4, out_channels=1, features=[32, 64, 128, 256]).to(self.device)
        criterion = CombinedSegmentationLoss(bce_weight=0.5, dice_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_epoch = 1
        best_metrics: Dict[str, float] = {"iou": 0.0, "dice_score": 0.0, "precision": 0.0, "recall": 0.0}
        best_state = None
        history = []

        logger.info(
            "Starting REAL Kuro Siwo training for %s: %d epochs, %d train, %d val samples, %d params on %s",
            self.model_name,
            self.epochs,
            len(train_ds),
            len(val_ds),
            model.count_parameters(),
            self.device,
        )

        for epoch in range(1, self.epochs + 1):
            # Training pass
            model.train()
            train_losses = []
            for imgs, masks in train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))

            # Validation pass
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)

                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    val_losses.append(loss.item())

                    probs = torch.sigmoid(logits).cpu().numpy()
                    targets = masks.cpu().numpy()
                    all_preds.append(probs)
                    all_targets.append(targets)

            val_loss = float(np.mean(val_losses)) if val_losses else train_loss
            y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((1, 1, 16, 16))
            y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((1, 1, 16, 16))
            metrics = evaluate_segmentation(y_pred, y_true, threshold=0.5)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "metrics": metrics,
            })

            logger.info(
                "Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f | Val IoU: %.4f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                metrics["dice_score"],
                metrics["iou"],
            )

            # Checkpoint selection (lowest validation loss)
            if val_loss < best_val_loss or best_state is None:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = metrics
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        duration = round(time.time() - start_time, 2)

        # Save checkpoint to disk
        checkpoint_path = self.models_dir / f"{self.model_name}.pt"
        checkpoint_payload = {
            "model_name": self.model_name,
            "architecture": "FloodUNet",
            "parameters_count": model.count_parameters(),
            "in_channels": 4,
            "features": [32, 64, 128, 256],
            "model_state_dict": best_state,
            "dataset_name": manifest.get("dataset_name", "Kuro Siwo Official Dataset (GRD)"),
            "dataset_source": OFFICIAL_SOURCE_URL,
            "official_source_url": OFFICIAL_SOURCE_URL,
            "dataset_version": manifest.get("dataset_version", "v2"),
            "dataset_path": str(self.dataset_dir.resolve()),
            "real_samples": True,
            "synthetic_samples_used": 0,
            "total_samples": manifest.get("total_samples", len(train_ds) + len(val_ds)),
            "train_samples": len(train_ds),
            "validation_samples": len(val_ds),
            "test_samples": manifest.get("test_samples", 0),
            "epochs": self.epochs,
            "best_epoch": best_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "seed": self.seed,
            "best_val_loss": round(best_val_loss, 4),
            "val_metrics": best_metrics,
            "iou": best_metrics["iou"],
            "dice_score": best_metrics["dice_score"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "duration_seconds": duration,
            "compute_device": str(self.device),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint_payload, checkpoint_path)

        # Save metadata sidecar JSON
        meta_path = self.models_dir / f"{self.model_name}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            # Save everything except state dict weights
            json_meta = {k: v for k, v in checkpoint_payload.items() if k != "model_state_dict"}
            json_meta["training_history"] = history
            json.dump(json_meta, f, indent=2)

        logger.info(
            "Model checkpoint saved to %s (Val Dice: %.4f, IoU: %.4f, 0 synthetic samples)",
            checkpoint_path,
            best_metrics["dice_score"],
            best_metrics["iou"],
        )

        return TrainModelResponse(
            model_name=self.model_name,
            checkpoint_path=str(checkpoint_path.resolve()),
            epochs_trained=self.epochs,
            train_loss=round(train_loss, 4),
            val_loss=round(best_val_loss, 4),
            val_metrics=SegmentationMetrics(**best_metrics),
            samples_count=len(train_ds) + len(val_ds),
            train_split=len(train_ds),
            val_split=len(val_ds),
            dataset_source=OFFICIAL_SOURCE_URL,
            status="completed",
            trained_at=datetime.now(timezone.utc),
            message=(
                f"Model '{self.model_name}' successfully trained on REAL Kuro Siwo dataset for {self.epochs} epochs. "
                f"Validation Dice: {best_metrics['dice_score']:.4f}, IoU: {best_metrics['iou']:.4f}. "
                "Zero synthetic samples used."
            ),
        )

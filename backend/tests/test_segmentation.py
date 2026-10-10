import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.ml.dataset import (
    KuroSiwoDataset,
    OFFICIAL_SOURCE_URL,
    REAL_KURO_SIWO_REQUIRED,
    normalize_sar_db,
    verify_kuro_siwo_dataset,
)
from app.ml.loss import CombinedSegmentationLoss, DiceLoss
from app.ml.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_precision,
    calculate_recall,
    evaluate_segmentation,
)
from app.ml.model import FloodUNet
from app.ml.train import SegmentationTrainer
from app.schemas.ard import PrepareArdRequest
from app.schemas.segmentation import SegmentationRequest
from app.services.ard_service import get_ard_service
from app.services.segmentation_service import get_segmentation_service

client = TestClient(app)


# ==========================================
# 1. Metrics Calculation Tests
# ==========================================

def test_metrics_perfect_overlap():
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)

    iou = calculate_iou(pred, target, threshold=0.5)
    dice = calculate_dice(pred, target, threshold=0.5)
    prec = calculate_precision(pred, target, threshold=0.5)
    rec = calculate_recall(pred, target, threshold=0.5)

    assert round(iou, 2) == 1.0
    assert round(dice, 2) == 1.0
    assert round(prec, 2) == 1.0
    assert round(rec, 2) == 1.0


def test_metrics_zero_overlap():
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[0.1, 0.2], [0.9, 0.8]], dtype=np.float32)

    iou = calculate_iou(pred, target, threshold=0.5)
    dice = calculate_dice(pred, target, threshold=0.5)

    assert iou < 0.01
    assert dice < 0.01


def test_metrics_partial_overlap():
    # Target: 2 ones. Pred: 1 true positive, 1 false positive, 1 false negative.
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    metrics = evaluate_segmentation(pred, target, threshold=0.5)
    assert 0.3 < metrics["iou"] < 0.4  # TP=1, Union=3 -> 1/3 ~ 0.3333
    assert 0.45 < metrics["dice_score"] < 0.55  # 2*1 / (2+2) = 0.5
    assert metrics["precision"] == 0.5  # TP=1, TP+FP=2 -> 0.5
    assert metrics["recall"] == 0.5  # TP=1, TP+FN=2 -> 0.5


# ==========================================
# 2. Model Architecture & Forward Pass Tests
# ==========================================

def test_flood_unet_instantiation():
    import torch
    model = FloodUNet(in_channels=4, out_channels=1, features=[32, 64, 128, 256])
    params = model.count_parameters()
    # Lightweight model: between 1M and 10M params
    assert 1_000_000 < params < 10_000_000

    # Test standard forward pass
    x = torch.randn(2, 4, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_flood_unet_arbitrary_spatial_dimensions():
    import torch
    model = FloodUNet(in_channels=4, out_channels=1)
    # Non-power-of-2 spatial dimensions
    x = torch.randn(1, 4, 137, 149)
    out = model(x)
    assert out.shape == (1, 1, 137, 149)

    probs, mask = model.predict_mask(x, threshold=0.5)
    assert probs.shape == (1, 1, 137, 149)
    assert mask.shape == (1, 1, 137, 149)
    assert torch.all((mask == 0) | (mask == 1))


# ==========================================
# 3. Loss Function Tests
# ==========================================

def test_segmentation_losses():
    import torch
    criterion = CombinedSegmentationLoss(bce_weight=0.5, dice_weight=0.5)

    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = criterion(logits, targets)
    assert loss.item() > 0.0

    # Backpropagation gradient test
    loss.backward()
    assert logits.grad is not None


# ==========================================
# 4. Real Kuro Siwo Dataset Verification Tests
# ==========================================

def test_real_kuro_siwo_dataset_verification():
    settings = get_settings()
    kuro_dir = settings.get_data_path(settings.KUROSIWO_DATA_DIR)
    
    assert kuro_dir.exists(), f"Kuro Siwo dataset directory '{kuro_dir}' must exist."
    manifest = verify_kuro_siwo_dataset(kuro_dir)
    
    assert manifest["real_samples"] is True
    assert manifest["synthetic_samples_used"] == 0
    assert manifest["total_samples"] >= 40
    assert manifest["train_samples"] > 0
    assert manifest["validation_samples"] > 0
    assert manifest["positive_flood_pixels"] > 0
    assert manifest["dataset_source"] == OFFICIAL_SOURCE_URL


def test_real_kuro_siwo_dataset_loader():
    settings = get_settings()
    kuro_dir = settings.get_data_path(settings.KUROSIWO_DATA_DIR)
    
    train_ds = KuroSiwoDataset(kuro_dir, split="train", augment=False)
    val_ds = KuroSiwoDataset(kuro_dir, split="val", augment=False)
    
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    
    # Check tensor shape and properties of first sample
    img, mask = train_ds[0]
    assert img.shape == (4, 224, 224)
    assert mask.shape == (1, 224, 224)
    # SAR input values are normalized to [0, 1]
    assert img.min().item() >= -0.1
    assert img.max().item() <= 1.1
    # Binary mask values are 0 or 1
    unique_vals = set(mask.unique().numpy().tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_real_kuro_siwo_empty_fails_loudly():
    empty_dir = Path(tempfile.mkdtemp())
    try:
        with pytest.raises(RuntimeError, match="MODULE 4 REAL-DATA TRAINING IS BLOCKED"):
            verify_kuro_siwo_dataset(empty_dir)
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)


# ==========================================
# 5. Checkpoint Provenance & Metadata Tests
# ==========================================

def test_model_checkpoint_provenance():
    settings = get_settings()
    models_dir = settings.get_data_path(settings.MODEL_ARTIFACTS_DIR)
    ckpt_path = models_dir / "flood_unet_s1.pt"
    meta_path = models_dir / "flood_unet_s1.json"
    
    assert ckpt_path.exists(), f"Model checkpoint '{ckpt_path}' must exist."
    assert meta_path.exists(), f"Model metadata sidecar '{meta_path}' must exist."
    
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Assert genuine real data training provenance
    assert ckpt["real_samples"] is True
    assert ckpt["synthetic_samples_used"] == 0
    assert ckpt["dataset_source"] == OFFICIAL_SOURCE_URL
    assert ckpt["train_samples"] > 0
    assert ckpt["validation_samples"] > 0
    assert ckpt["val_metrics"]["dice_score"] > 0.4
    assert ckpt["val_metrics"]["iou"] > 0.3
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["real_samples"] is True
        assert meta["synthetic_samples_used"] == 0
        assert meta["dataset_source"] == OFFICIAL_SOURCE_URL
        assert meta["val_metrics"]["dice_score"] > 0.4


# ==========================================
# 6. End-to-End Segmentation Inference Service Test
# ==========================================

def test_segmentation_inference_with_ard():
    ard_service = get_ard_service()
    seg_service = get_segmentation_service()

    # 1. Prepare ARD product for test job
    job_id = "test_seg_job_001"
    req = PrepareArdRequest(
        job_id=job_id,
        flood_date="2024-08-26",
        resolution_meters=100.0,
    )
    ard_pkg = ard_service.prepare_ard(req)
    assert ard_pkg.status in ["ready", "contract_ready"]

    # 2. Run AI segmentation
    seg_req = SegmentationRequest(
        job_id=job_id,
        confidence_threshold=0.5,
        min_connected_pixels=2,
    )
    resp = seg_service.segment_flood(seg_req)

    assert resp.job_id == job_id
    assert resp.status == "completed"
    assert Path(resp.mask_raster_path).exists()
    assert resp.geojson_features_count >= 0
    assert resp.geojson["type"] == "FeatureCollection"
    assert resp.summary.total_valid_pixels > 0
    assert resp.summary.flood_pixel_count >= 0
    assert resp.summary.flood_area_sqkm >= 0.0

    # 3. Retrieve stored result
    stored = seg_service.get_segmentation_result(job_id)
    assert stored is not None
    assert stored.job_id == job_id


# ==========================================
# 7. FastAPI API Endpoints Tests
# ==========================================

def test_api_segmentation_endpoints():
    ard_service = get_ard_service()
    job_id = "test_api_seg_job"

    # Prepare ARD first
    ard_service.prepare_ard(PrepareArdRequest(job_id=job_id, flood_date="2024-08-26", resolution_meters=100.0))

    # POST /analysis/{job_id}/segment
    post_resp = client.post(
        f"/analysis/{job_id}/segment",
        json={"confidence_threshold": 0.5, "min_connected_pixels": 2},
    )
    assert post_resp.status_code == 200
    data = post_resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert "summary" in data
    assert "geojson" in data

    # GET /analysis/{job_id}/segment
    get_resp = client.get(f"/analysis/{job_id}/segment")
    assert get_resp.status_code == 200
    assert get_resp.json()["job_id"] == job_id

    # GET /analysis/{job_id}/segment/mask (GeoTIFF stream)
    mask_resp = client.get(f"/analysis/{job_id}/segment/mask")
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "image/tiff"
    assert len(mask_resp.content) > 0

    # GET /models
    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    assert isinstance(models_resp.json(), list)

    # GET /analysis/nonexistent_job/segment -> 404
    err_resp = client.get("/analysis/nonexistent_job_9999/segment")
    assert err_resp.status_code == 404

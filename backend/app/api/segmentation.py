import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.schemas.segmentation import (
    ModelInfo,
    SegmentationRequest,
    SegmentationResponse,
    TrainModelRequest,
    TrainModelResponse,
)
from app.services.segmentation_service import (
    SegmentationInferenceService,
    get_segmentation_service,
)
from app.ml.train import SegmentationTrainer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Flood Segmentation"])


@router.post(
    "/analysis/{job_id}/segment",
    response_model=SegmentationResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute AI Flood Segmentation on Analysis-Ready Imagery",
    description=(
        "Consumes the prepared Sentinel ARD package for the specified job, "
        "runs semantic segmentation using FloodUNet, and generates both pixel-level "
        "GeoTIFF flood masks and vectorized GeoJSON polygon geometries with spatial damage metrics."
    ),
)
def run_segmentation(
    job_id: str,
    request: Optional[SegmentationRequest] = None,
    service: SegmentationInferenceService = Depends(get_segmentation_service),
) -> SegmentationResponse:
    if request is None:
        req = SegmentationRequest(job_id=job_id)
    else:
        req = request
        req.job_id = job_id

    try:
        return service.segment_flood(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("AI segmentation failed for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}",
        )


@router.get(
    "/analysis/{job_id}/segment",
    response_model=SegmentationResponse,
    summary="Get Stored AI Segmentation Results",
    description="Retrieve existing segmentation output, vector polygons, and spatial damage summary.",
)
def get_segmentation_results(
    job_id: str,
    service: SegmentationInferenceService = Depends(get_segmentation_service),
) -> SegmentationResponse:
    res = service.get_segmentation_result(job_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No segmentation results found for job '{job_id}'. Run POST /analysis/{job_id}/segment first.",
        )
    return res


@router.get(
    "/analysis/{job_id}/segment/mask",
    summary="Download GeoTIFF Flood Mask",
    description="Returns the calibrated, georeferenced binary GeoTIFF raster mask.",
)
def download_flood_mask(
    job_id: str,
    service: SegmentationInferenceService = Depends(get_segmentation_service),
):
    res = service.get_segmentation_result(job_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No segmentation results found for job '{job_id}'.",
        )
    mask_file = Path(res.mask_raster_path)
    if not mask_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Flood mask raster file missing from disk.",
        )
    return FileResponse(
        path=str(mask_file),
        media_type="image/tiff",
        filename=f"{job_id}_flood_mask.tif",
    )


@router.post(
    "/models/train",
    response_model=TrainModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train Semantic Segmentation Model on Kuro Siwo Dataset",
    description=(
        "Launches training of FloodUNet on the Kuro Siwo multi-temporal flood dataset "
        "(or synthetic benchmark). Optimizes BCE + Dice loss and records IoU/Dice metrics."
    ),
)
def train_model(
    request: TrainModelRequest,
) -> TrainModelResponse:
    try:
        trainer = SegmentationTrainer(
            model_name=request.model_name,
            dataset_dir=request.dataset_dir,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            seed=request.seed,
        )
        return trainer.train()
    except Exception as e:
        logger.exception("Model training failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )


@router.get(
    "/models",
    response_model=List[ModelInfo],
    summary="List Registered AI Segmentation Models",
    description="List all available trained FloodUNet model checkpoints and validation performance.",
)
def list_models(
    service: SegmentationInferenceService = Depends(get_segmentation_service),
) -> List[ModelInfo]:
    return service.list_available_models()

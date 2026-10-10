import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.connectivity import (
    ConnectivityAnalysisRequest,
    ConnectivityAnalysisResponse,
)
from app.services.connectivity_service import (
    ConnectivityAnalysisService,
    get_connectivity_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Road Network & Settlement Connectivity Analysis"])


@router.post(
    "/analysis/{job_id}/connectivity",
    response_model=ConnectivityAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Road Network & Settlement Accessibility Analysis",
    description=(
        "Constructs a routable graph from pre-event OpenStreetMap roads, identifies estimated blocked "
        "edges from Module 5 spatial exposure, builds a post-flood estimated graph, and evaluates "
        "reachability, flood isolation, road detours, and healthcare facility access."
    ),
)
def run_connectivity_analysis(
    job_id: str,
    request: Optional[ConnectivityAnalysisRequest] = None,
    service: ConnectivityAnalysisService = Depends(get_connectivity_service),
) -> ConnectivityAnalysisResponse:
    req = request or ConnectivityAnalysisRequest()
    req.job_id = job_id

    try:
        return service.analyze_connectivity(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Connectivity analysis failed for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connectivity analysis failed: {str(e)}",
        )


@router.get(
    "/analysis/{job_id}/connectivity",
    response_model=ConnectivityAnalysisResponse,
    summary="Get Stored Connectivity & Accessibility Results",
    description="Retrieve structured road network metrics, settlement isolation counts, detour statistics, and hospital access summary.",
)
def get_connectivity_results(
    job_id: str,
    service: ConnectivityAnalysisService = Depends(get_connectivity_service),
) -> ConnectivityAnalysisResponse:
    res = service.get_connectivity_result(job_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No connectivity analysis results found for job '{job_id}'. Run POST /analysis/{job_id}/connectivity first.",
        )
    return res


@router.get(
    "/analysis/{job_id}/connectivity/roads",
    summary="Get Blocked Road Segments GeoJSON",
    description="Retrieve estimated blocked road segments (>= 50% flood exposure) in GeoJSON FeatureCollection format.",
)
def get_blocked_roads(
    job_id: str,
    service: ConnectivityAnalysisService = Depends(get_connectivity_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "blocked_roads")
    if not layer:
        res = service.get_connectivity_result(job_id)
        if res:
            return res.blocked_roads
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Road connectivity data not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/connectivity/settlements",
    summary="Get Settlement Connectivity Status GeoJSON",
    description="Retrieve evaluated settlements with baseline/post-flood reachability, route distances, detours, and hospital access in GeoJSON FeatureCollection format.",
)
def get_settlements_connectivity(
    job_id: str,
    service: ConnectivityAnalysisService = Depends(get_connectivity_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "settlements_connectivity")
    if not layer:
        res = service.get_connectivity_result(job_id)
        if res:
            return res.settlements_connectivity
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Settlement connectivity data not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/connectivity/isolated",
    summary="Get Flood-Isolated Settlements GeoJSON",
    description="Retrieve settlements strictly classified as flood-isolated (baseline connected, post-flood disconnected) in GeoJSON FeatureCollection format.",
)
def get_isolated_settlements(
    job_id: str,
    service: ConnectivityAnalysisService = Depends(get_connectivity_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "isolated_settlements")
    if not layer:
        res = service.get_connectivity_result(job_id)
        if res:
            return res.isolated_settlements
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Isolated settlement data not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/connectivity/hospitals",
    summary="Get Hospital Access Lost Settlements GeoJSON",
    description="Retrieve settlements that lost road-network access to healthcare facilities due to estimated blocked roads.",
)
def get_hospital_accessibility(
    job_id: str,
    service: ConnectivityAnalysisService = Depends(get_connectivity_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "hospital_access_lost_settlements")
    if not layer:
        res = service.get_connectivity_result(job_id)
        if res:
            return res.hospital_access_lost_settlements
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hospital accessibility data not found for job '{job_id}'.",
        )
    return layer

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.geospatial import (
    GeospatialAnalysisRequest,
    GeospatialAnalysisResponse,
)
from app.services.geospatial_service import (
    GeospatialDamageAnalysisService,
    get_geospatial_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Geospatial Damage Analysis"])


@router.post(
    "/analysis/{job_id}/geospatial",
    response_model=GeospatialAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Geospatial Infrastructure Damage & Exposure Analysis",
    description=(
        "Consumes Module 4 AI flood segmentation predictions, measures geodesic inundation extents, "
        "extracts pre-event OpenStreetMap assets (buildings, roads, bridges), and performs spatial "
        "overlay analysis to classify infrastructure exposure and estimate potential damage."
    ),
)
def run_geospatial_analysis(
    job_id: str,
    request: Optional[GeospatialAnalysisRequest] = None,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> GeospatialAnalysisResponse:
    req = request or GeospatialAnalysisRequest()
    req.job_id = job_id

    try:
        return service.analyze_job_impact(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Geospatial analysis failed for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Geospatial analysis failed: {str(e)}",
        )


@router.get(
    "/analysis/{job_id}/geospatial",
    response_model=GeospatialAnalysisResponse,
    summary="Get Stored Geospatial Asset Analysis Results",
    description="Retrieve existing geospatial infrastructure damage summary, metrics, and asset counts.",
)
def get_geospatial_results(
    job_id: str,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> GeospatialAnalysisResponse:
    res = service.get_geospatial_result(job_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No geospatial analysis results found for job '{job_id}'. Run POST /analysis/{job_id}/geospatial first.",
        )
    return res


@router.get(
    "/analysis/{job_id}/geospatial/flood-regions",
    summary="Get Vectorized Flood Regions GeoJSON",
    description="Retrieve discrete polygonized floodwater extent regions in GeoJSON FeatureCollection format.",
)
def get_flood_regions(
    job_id: str,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "flood_regions")
    if not layer:
        # Check in memory response
        res = service.get_geospatial_result(job_id)
        if res:
            return res.flood_regions
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flood regions not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/geospatial/buildings",
    summary="Get Affected Buildings GeoJSON",
    description="Retrieve evaluated OpenStreetMap buildings with exposure classifications and overlap percentages.",
)
def get_affected_buildings(
    job_id: str,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "buildings")
    if not layer:
        res = service.get_geospatial_result(job_id)
        if res:
            return res.buildings
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Building impact data not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/geospatial/roads",
    summary="Get Affected Roads GeoJSON",
    description="Retrieve evaluated OpenStreetMap road segments with flooded lengths and blockage classifications.",
)
def get_affected_roads(
    job_id: str,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "roads")
    if not layer:
        res = service.get_geospatial_result(job_id)
        if res:
            return res.roads
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Road impact data not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/geospatial/bridges",
    summary="Get Affected Bridges GeoJSON",
    description="Retrieve evaluated OpenStreetMap bridges with flood intersections and exposure classifications.",
)
def get_affected_bridges(
    job_id: str,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "bridges")
    if not layer:
        res = service.get_geospatial_result(job_id)
        if res:
            return res.bridges
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bridge impact data not found for job '{job_id}'.",
        )
    return layer


@router.get(
    "/analysis/{job_id}/geospatial/change-zones",
    summary="Get Candidate Terrain Change / Debris Zones GeoJSON",
    description="Retrieve candidate terrain change and potential debris zones derived from SAR backscatter differencing.",
)
def get_change_zones(
    job_id: str,
    service: GeospatialDamageAnalysisService = Depends(get_geospatial_service),
) -> Dict[str, Any]:
    layer = service.get_layer_geojson(job_id, "change_zones")
    if not layer:
        res = service.get_geospatial_result(job_id)
        if res:
            return res.change_zones
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Change zone data not found for job '{job_id}'.",
        )
    return layer

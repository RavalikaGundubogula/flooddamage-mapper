from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from app.schemas.analysis import AnalysisCreateRequest, AnalysisResponse
from app.schemas.pipeline import PipelineRunRequest, PipelineStatusResponse, PipelineStage
from app.services.analysis_service import AnalysisService, get_analysis_service
from app.services.pipeline_service import PipelineExecutionService, get_pipeline_service

router = APIRouter(prefix="/analysis", tags=["Analysis & Pipeline Orchestration"])


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new flood damage analysis job",
    description="Submits an Area of Interest (AOI) and event date to create a new analysis job.",
)
def create_analysis(
    payload: AnalysisCreateRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """
    Create a new analysis job:
    - Validates coordinates and geometry
    - Normalizes AOI to GeoJSON Polygon format
    - Generates a unique job_id
    - Initializes job status to 'created'
    """
    record = service.create_analysis(payload)
    return AnalysisResponse(
        job_id=record.job_id,
        status=record.status.value,
        aoi=record.aoi,
        flood_date=record.flood_date,
        report_language=record.report_language,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get(
    "",
    summary="List available analysis jobs",
    description="Lists saved and pre-computed flood damage analysis jobs.",
)
def list_analyses(
    service: AnalysisService = Depends(get_analysis_service),
    pipeline_service: PipelineExecutionService = Depends(get_pipeline_service),
):
    """Returns available analysis jobs, highlighting real case studies and completed runs."""
    jobs = []
    nepal_dir = pipeline_service.outputs_dir / "job_nepal_2026_case"
    if nepal_dir.exists():
        jobs.append({
            "job_id": "job_nepal_2026_case",
            "title": "Northern Nepal Trishuli / Bhote Koshi Basin (26 Aug 2026)",
            "flood_date": "2026-08-26",
            "status": "completed",
            "is_real_case": True,
            "description": "Real Sentinel-1 SAR dual-polarization & OSM infrastructure analysis",
        })

    seen_ids = {"job_nepal_2026_case"}
    for record in service.repository.list_all():
        if record.job_id not in seen_ids:
            seen_ids.add(record.job_id)
            jobs.append({
                "job_id": record.job_id,
                "title": f"Analysis {record.job_id}",
                "flood_date": record.flood_date,
                "status": record.status.value,
                "is_real_case": False,
                "description": f"Analysis job created at {record.created_at}",
            })

    return {"jobs": jobs}


@router.get(
    "/{job_id}",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Get analysis job status and metadata",
    description="Retrieves the current status, normalized AOI, and timestamps for a given job_id.",
)
def get_analysis_status(
    job_id: str,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """Retrieve status and details for an existing analysis job."""
    record = service.get_analysis(job_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found",
        )

    return AnalysisResponse(
        job_id=record.job_id,
        status=record.status.value,
        aoi=record.aoi,
        flood_date=record.flood_date,
        report_language=record.report_language,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post(
    "/run",
    response_model=PipelineStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run End-to-End Flood Analysis Pipeline",
    description=(
        "Executes complete disaster pipeline: satellite discovery, smart router, "
        "scene selection, ARD preparation, FloodUNet segmentation, OSM geospatial analysis, "
        "road network connectivity, structured results aggregation, and multilingual situation report."
    ),
)
def run_analysis_pipeline(
    payload: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    sync: bool = False,
    pipeline_service: PipelineExecutionService = Depends(get_pipeline_service),
) -> PipelineStatusResponse:
    """Initiate or execute full pipeline."""
    import uuid
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    if sync:
        return pipeline_service.execute_pipeline(request=payload, job_id=job_id)

    # Initialize state
    init_state = pipeline_service._update_stage(
        job_id=job_id,
        stage=PipelineStage.CREATED,
        progress_percent=0,
        message="Analysis pipeline queued for background execution",
        status_code="created",
    )
    init_state.flood_date = payload.flood_date
    init_state.report_language = payload.report_language
    init_state.demo_mode = payload.demo_mode

    # Schedule background execution
    background_tasks.add_task(pipeline_service.execute_pipeline, payload, job_id)
    return init_state


@router.get(
    "/{job_id}/status",
    response_model=PipelineStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Pipeline Execution Status & Stepper Progress",
    description="Retrieves current pipeline stage, percentage completion, blocker information, and layer status.",
)
def get_pipeline_status(
    job_id: str,
    pipeline_service: PipelineExecutionService = Depends(get_pipeline_service),
) -> PipelineStatusResponse:
    state = pipeline_service.get_status(job_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis pipeline job '{job_id}' not found",
        )
    return state


@router.get(
    "/{job_id}/layers/{layer_name}",
    status_code=status.HTTP_200_OK,
    summary="Get GeoJSON Layer for Interactive Web Map",
    description="Fetches specific GeoJSON FeatureCollection (aoi, flood, buildings, roads, bridges, blocked_roads, isolated_settlements, hospitals, change_zones).",
)
def get_analysis_layer(
    job_id: str,
    layer_name: str,
    pipeline_service: PipelineExecutionService = Depends(get_pipeline_service),
):
    geojson = pipeline_service.get_layer_geojson(job_id, layer_name)
    if geojson is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Layer '{layer_name}' not available for job '{job_id}'",
        )
    return geojson


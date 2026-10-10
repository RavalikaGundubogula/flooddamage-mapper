import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.results import (
    ReportGenerateRequest,
    ReportResponse,
    StructuredAnalysisResult,
)
from app.services.report_service import (
    SituationReportService,
    get_report_service,
)
from app.services.results_service import (
    ResultsAggregationService,
    get_results_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Structured Results & Multilingual Situation Reports"])


@router.post(
    "/analysis/{job_id}/results",
    response_model=StructuredAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Aggregate Central Structured Analysis Results",
    description=(
        "Consolidates outputs across Modules 1 through 6 into a single canonical record. "
        "Explicitly marks satellite blockers, preserves factual trust classifications "
        "(VERIFIED, ESTIMATED, BLOCKED, UNAVAILABLE), and enforces strict data provenance."
    ),
)
def aggregate_analysis_results(
    job_id: str,
    service: ResultsAggregationService = Depends(get_results_service),
) -> StructuredAnalysisResult:
    try:
        return service.build_analysis_result(job_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Failed to aggregate results for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Result aggregation failed: {str(e)}",
        )


@router.get(
    "/analysis/{job_id}/results",
    response_model=StructuredAnalysisResult,
    summary="Get Stored Central Structured Analysis Results",
    description="Retrieve canonical machine-readable analysis results, asset summaries, and factuality registry.",
)
def get_analysis_results(
    job_id: str,
    service: ResultsAggregationService = Depends(get_results_service),
) -> StructuredAnalysisResult:
    res = service.get_analysis_result(job_id)
    if not res:
        # Build if not already stored
        try:
            return service.build_analysis_result(job_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis results found for job '{job_id}': {e}",
            )
    return res


@router.post(
    "/analysis/{job_id}/report",
    response_model=ReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Multilingual Humanitarian Situation Report",
    description=(
        "Synthesizes structured factual analysis findings into an 11-section situation brief. "
        "Supports English ('en') and Nepali ('ne') with 100% numerical consistency. "
        "Uses cost-aware LLM routing when configured and deterministic fallback templates when offline."
    ),
)
def generate_situation_report(
    job_id: str,
    request: Optional[ReportGenerateRequest] = None,
    service: SituationReportService = Depends(get_report_service),
) -> ReportResponse:
    req = request or ReportGenerateRequest()
    try:
        return service.generate_report(job_id, req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Report generation failed for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )


@router.get(
    "/analysis/{job_id}/report",
    response_model=ReportResponse,
    summary="Get Cached or Generated Situation Report",
    description="Retrieve situation report in requested language ('en' by default).",
)
def get_situation_report(
    job_id: str,
    language: str = Query("en", description="Target ISO language code ('en' or 'ne')"),
    service: SituationReportService = Depends(get_report_service),
) -> ReportResponse:
    req = ReportGenerateRequest(language=language)
    try:
        return service.generate_report(job_id, req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to retrieve report for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve report: {str(e)}",
        )


@router.get(
    "/analysis/{job_id}/report/{language}",
    response_model=ReportResponse,
    summary="Get Situation Report in Specific Language",
    description="Retrieve situation report for specific language code (e.g. 'en', 'ne').",
)
def get_situation_report_by_language(
    job_id: str,
    language: str,
    service: SituationReportService = Depends(get_report_service),
) -> ReportResponse:
    req = ReportGenerateRequest(language=language)
    try:
        return service.generate_report(job_id, req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to retrieve report for job %s: %s", job_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve report: {str(e)}",
        )

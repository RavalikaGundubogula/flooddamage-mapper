from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.analysis import router as analysis_router
from app.api.satellites import router as satellites_router
from app.api.segmentation import router as segmentation_router
from app.api.geospatial import router as geospatial_router
from app.api.connectivity import router as connectivity_router
from app.api.reports import router as reports_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "FastAPI Backend Foundation for Track B: Mapping Flood Damage from Space. "
        "Provides AOI normalization, satellite discovery, smart routing, AI segmentation, "
        "geospatial damage analysis, road network connectivity, and multilingual situation reports."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(analysis_router)
app.include_router(satellites_router)
app.include_router(segmentation_router)
app.include_router(geospatial_router)
app.include_router(connectivity_router)
app.include_router(reports_router)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
def dashboard_view():
    """Serves the interactive Web GIS disaster dashboard."""
    html_file = Path(__file__).parent / "static" / "dashboard.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Flood Damage Mapper Dashboard</h1><p>Static dashboard file not found.</p>")


@app.get("/", tags=["General"])
def root(request: Request):
    """Root entrypoint returning interactive dashboard for browsers, or API metadata for JSON clients."""
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        html_file = Path(__file__).parent / "static" / "dashboard.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text(encoding="utf-8"))

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "dashboard_url": "/dashboard",
        "endpoints": {
            "run_pipeline": "POST /analysis/run",
            "get_pipeline_status": "GET /analysis/{job_id}/status",
            "get_results": "GET /analysis/{job_id}/results",
            "get_report": "GET /analysis/{job_id}/report",
            "create_analysis": "POST /analysis",
            "get_analysis_status": "GET /analysis/{job_id}",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.VERSION}


# FloodScope — Flood Damage Mapper (Multimodal AI 2026)

Geospatial AI disaster mapping system for the **Amazon Multimodal AI Hackathon 2026, Track B: Mapping Flood Damage from Space**.

This repository integrates:
1. **Frontend (`src/`)**: Modern React + TypeScript + Vite spatial intelligence dashboard.
2. **Backend (`backend/`)**: FastAPI pipeline orchestrating Copernicus Data Space Ecosystem (CDSE) Sentinel-1 SAR & Sentinel-2 optical data, PyTorch FloodUNet segmentation, OpenStreetMap infrastructure exposure, and NetworkX road/hospital accessibility analysis.

---

## Architecture Overview

```
User / Analyst (React UI)
       │
       ▼ (AOI + Event Date)
FastAPI Backend (backend/app/main.py)
       │
       ├─► 1. Smart Satellite Resource Router (Sentinel-1 SAR / Sentinel-2 Optical)
       ├─► 2. Before/After ARD Preprocessing
       ├─► 3. PyTorch FloodUNet Inundation Segmentation
       ├─► 4. OpenStreetMap Infrastructure Overlays (Buildings, Roads, Bridges, Hospitals)
       ├─► 5. NetworkX Road Graph & Critical Facility Accessibility
       └─► 6. Multilingual Situation Report Synthesis (English & Nepali)
```

---

## Quick Start Guide

### 1. Prerequisites
- **Node.js**: v18+ and `npm`
- **Python**: 3.10+ (recommended in a virtual environment)
- **Git**

---

### 2. Backend Setup

```bash
cd backend

# Create and activate a Python virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Copernicus CDSE credentials if running real binary downloads:
# COPERNICUS_CLIENT_ID=...
# COPERNICUS_CLIENT_SECRET=...

# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at:
- **API Root / Health**: [http://localhost:8000/health](http://localhost:8000/health)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Built-in Fallback Dashboard**: [http://localhost:8000/](http://localhost:8000/)

#### Running Backend Tests
```bash
pytest backend/tests -v
```

---

### 3. Frontend Setup

From the repository root:

```bash
# Install NPM dependencies
npm install

# (Optional) Point to a custom backend URL in .env.local
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

# Start the Vite development server
npm run dev

# Or build for production
npm run build
```

By default, Vite runs at `http://localhost:5173`. When the backend is running on `http://localhost:8000`, the top status badge will switch to **API READY** with real-time analysis metrics.

---

## Verification & Case Studies

The repository includes pre-computed artifacts for the **26 August 2026 Nepal Flood Event** under `backend/data/outputs/job_nepal_2026_case/`:
- `analysis_result.json`: Complete metrics summary
- `flood_regions.geojson`: AI-detected flood polygon extent
- `roads.geojson` & `blocked_roads.geojson`: Road network disruption
- `bridges.geojson`: Structural asset exposure
- `situation_report_en.json` & `situation_report_ne.json`: AI situation reports

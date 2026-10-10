/**
 * FloodScope API client connecting the React frontend to the FastAPI backend.
 */

export const API_BASE_URL =
  (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export interface HealthResponse {
  status: string;
  project: string;
  version: string;
  env: string;
}

export interface AnalysisSummary {
  flooded_area_km2: number;
  total_buildings_detected: number;
  buildings_in_flood_zone: number;
  total_roads_analyzed: number;
  roads_blocked: number;
  roads_affected: number;
  isolated_settlements: number;
  settlements_lost_hospital_access: number;
  confidence_score: number;
}

export interface AnalysisJob {
  analysis_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'BLOCKED_UPSTREAM';
  progress: number;
  current_step: string;
  created_at: string;
  updated_at: string;
  summary?: AnalysisSummary;
  layers?: Record<string, any>;
  report?: any;
}

export async function checkBackendHealth(): Promise<{ connected: boolean; info?: HealthResponse; error?: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return { connected: true, info: data };
  } catch (err: any) {
    return { connected: false, error: err.message || 'Connection failed' };
  }
}

export async function fetchAnalysisJob(analysisId: string): Promise<AnalysisJob> {
  const res = await fetch(`${API_BASE_URL}/api/v1/analysis/${analysisId}`);
  if (!res.ok) throw new Error(`Failed to fetch analysis ${analysisId}: HTTP ${res.status}`);
  return await res.json();
}

export async function fetchAnalysisLayers(analysisId: string): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE_URL}/api/v1/analysis/${analysisId}/layers`);
  if (!res.ok) throw new Error(`Failed to fetch layers: HTTP ${res.status}`);
  return await res.json();
}

export async function startNewAnalysis(payload: {
  event_date: string;
  coordinates: number[][][];
  language?: string;
  mode?: 'auto' | 'sync' | 'demo';
}): Promise<AnalysisJob> {
  const endpoint = payload.mode === 'sync' ? '/api/v1/analysis/run-sync' : '/api/v1/analysis';
  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_date: payload.event_date,
      coordinates: payload.coordinates,
      report_language: payload.language || 'en',
    }),
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`API error (${res.status}): ${errBody}`);
  }
  return await res.json();
}

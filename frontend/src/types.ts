export type PipelineStage =
  | 'created'
  | 'validating'
  | 'discovering_satellites'
  | 'selecting_scenes'
  | 'preparing_ard'
  | 'running_segmentation'
  | 'running_geospatial'
  | 'running_connectivity'
  | 'building_results'
  | 'generating_report'
  | 'completed'
  | 'completed_with_limitations'
  | 'blocked'
  | 'failed';

export interface StageLogEntry {
  stage: PipelineStage;
  timestamp: string;
  message: string;
  progress_percent: number;
}

export interface PipelineSummary {
  flood_area_km2?: number | null;
  affected_buildings?: number | Record<string, any> | null;
  affected_buildings_details?: Record<string, any> | null;
  affected_roads?: number | Record<string, any> | null;
  affected_roads_km?: number | null;
  affected_roads_details?: Record<string, any> | null;
  affected_bridges?: number | Record<string, any> | null;
  affected_bridges_details?: Record<string, any> | null;
  blocked_roads?: number | null;
  isolated_settlements?: number | null;
  hospital_access_lost?: number | null;
  debris_zones?: number | null;
  [key: string]: any;
}

export interface PipelineStatus {
  job_id: string;
  status: string;
  current_stage: PipelineStage;
  progress_percent: number;
  message: string;
  timestamp: string;
  is_blocked: boolean;
  blocker?: string | null;
  error_message?: string | null;
  demo_mode?: boolean;
  flood_date?: string;
  comparison_date?: string;
  reference_date?: string;
  report_language?: string;
  available_layers?: string[];
  satellite_status?: string | null;
  segmentation_status?: string | null;
  geospatial_status?: string | null;
  connectivity_status?: string | null;
  report_status?: string | null;
  stages_history?: StageLogEntry[];
  summary?: PipelineSummary | null;
}

export interface SituationReportSection {
  section_id?: string;
  title?: string;
  heading?: string;
  content: string;
  key_metrics?: Record<string, any>;
  caveats?: string[];
}

export interface SituationReportPayload {
  title: string;
  executive_summary?: string;
  sections?: SituationReportSection[];
  metadata?: Record<string, any>;
  disclaimer?: string;
}

export interface SituationReportResponse {
  job_id: string;
  language: string;
  report_status: string;
  llm_used: boolean;
  report?: SituationReportPayload;
  metadata?: Record<string, any>;
}

export interface AnalysisJobItem {
  job_id: string;
  title: string;
  flood_date: string;
  status: string;
  is_real_case: boolean;
  description: string;
}

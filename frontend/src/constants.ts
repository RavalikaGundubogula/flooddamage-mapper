import {
  PipelineStage,
} from './types';

// Preset AOIs
export interface PresetAOI {
  id: string;
  name: string;
  region: string;
  bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number };
  defaultEventDate: string;
  defaultCompDate: string;
  approxAreaKm2: number;
  description: string;
}

export const PRESET_AOIS: PresetAOI[] = [
  {
    id: 'nepal_trishuli',
    name: 'Northern Nepal — Trishuli / Bhote Koshi Basin',
    region: 'Bagmati Province, Nepal',
    bbox: { minLon: 85.20, minLat: 27.80, maxLon: 85.50, maxLat: 28.10 },
    defaultEventDate: '2026-08-26',
    defaultCompDate: '2026-08-31',
    approxAreaKm2: 985.4,
    description: 'Reference disaster corridor: steep Himalayan gorge with Trishuli highway and suspension bridges.',
  },
  {
    id: 'nepal_kathmandu',
    name: 'Kathmandu Valley — Bagmati River Basin',
    region: 'Kathmandu, Nepal',
    bbox: { minLon: 85.25, minLat: 27.64, maxLon: 85.40, maxLat: 27.75 },
    defaultEventDate: '2026-08-26',
    defaultCompDate: '2026-08-30',
    approxAreaKm2: 180.2,
    description: 'Dense urban floodplain along Bagmati and Bishnumati rivers with high building density.',
  },
  {
    id: 'nepal_terai',
    name: 'Eastern Terai — Koshi River Floodplain',
    region: 'Koshi Province, Nepal',
    bbox: { minLon: 86.95, minLat: 26.50, maxLon: 87.25, maxLat: 26.85 },
    defaultEventDate: '2026-08-15',
    defaultCompDate: '2026-08-22',
    approxAreaKm2: 1120.0,
    description: 'Lowland braided alluvial plain susceptible to wide-area seasonal embankment overtopping.',
  },
  {
    id: 'brazil_porto_alegre',
    name: 'Porto Alegre Flood Basin',
    region: 'Rio Grande do Sul, Brazil',
    bbox: { minLon: -51.30, minLat: -30.15, maxLon: -51.05, maxLat: -29.90 },
    defaultEventDate: '2024-05-04',
    defaultCompDate: '2024-05-08',
    approxAreaKm2: 760.5,
    description: 'Severe urban inundation across Guaíba Lake tributaries and Salgado Filho Airport.',
  },
  {
    id: 'bangladesh_sylhet',
    name: 'Sylhet Division — Surma-Kushiyara Basin',
    region: 'Sylhet, Bangladesh',
    bbox: { minLon: 91.70, minLat: 24.80, maxLon: 92.20, maxLat: 25.20 },
    defaultEventDate: '2024-08-21',
    defaultCompDate: '2024-08-28',
    approxAreaKm2: 2210.0,
    description: 'Monsoon flash flood basin along Surma River and haor wetlands.',
  },
];

// Pipeline Stages definition
export const PIPELINE_STAGES: { id: PipelineStage; labelEn: string; labelNe: string; step: number }[] = [
  { id: 'validating', labelEn: '1. AOI & Boundary Validation', labelNe: '१. क्षेत्र तथा सीमा प्रमाणीकरण', step: 1 },
  { id: 'discovering_satellites', labelEn: '2. CDSE Satellite Discovery (S1/S2)', labelNe: '२. भूउपग्रह खोजी तथा प्राप्ति', step: 2 },
  { id: 'selecting_scenes', labelEn: '3. Scene Matching & Relative Orbit Router', labelNe: '३. उपग्रह जोडी छनोट', step: 3 },
  { id: 'preparing_ard', labelEn: '4. Analysis-Ready Data (ARD) Stack', labelNe: '४. विश्लेषण-योग्य डेटा तयारी', step: 4 },
  { id: 'running_segmentation', labelEn: '5. AI FloodUNet Inundation Inference', labelNe: '५. एआई बाढी वर्गीकरण', step: 5 },
  { id: 'running_geospatial', labelEn: '6. OSM Infrastructure Exposure', labelNe: '६. पूर्वाधार क्षति विश्लेषण', step: 6 },
  { id: 'running_connectivity', labelEn: '7. Road Network & Hospital Reachability', labelNe: '७. सडक सञ्जाल तथा अस्पताल पहुँच', step: 7 },
  { id: 'building_results', labelEn: '8. Structured Provenance Synthesis', labelNe: '८. संरचित तथ्यांक संकलन', step: 8 },
  { id: 'generating_report', labelEn: '9. Multilingual Situation Briefing', labelNe: '९. बहुभाषिक मानवीय प्रतिवेदन', step: 9 },
  { id: 'completed', labelEn: '10. Completed & Ready', labelNe: '१०. विश्लेषण सम्पन्न', step: 10 },
];

// Helper to format values cleanly
export function formatValue(val: any, fallback = '—'): string {
  if (val === null || val === undefined) return fallback;
  if (typeof val === 'number') return val.toLocaleString();
  if (typeof val === 'string') return val;
  if (typeof val === 'object') {
    if (val.heavily_affected !== undefined && val.potentially_affected !== undefined) {
      return (val.heavily_affected + val.potentially_affected).toLocaleString();
    }
    if (val.total !== undefined) return Number(val.total).toLocaleString();
    if (val.affected_length_km !== undefined) return `${Number(val.affected_length_km).toFixed(2)} km`;
    if (val.potentially_blocked !== undefined) return Number(val.potentially_blocked).toLocaleString();
  }
  return fallback;
}

// Calculate approximate bounding box area in km2
export function calculateBBoxArea(minLon: number, minLat: number, maxLon: number, maxLat: number): number {
  const avgLat = (minLat + maxLat) / 2.0;
  const dxKm = Math.abs(maxLon - minLon) * 111.32 * Math.cos((avgLat * Math.PI) / 180);
  const dyKm = Math.abs(maxLat - minLat) * 110.57;
  return Math.round(dxKm * dyKm * 10) / 10;
}

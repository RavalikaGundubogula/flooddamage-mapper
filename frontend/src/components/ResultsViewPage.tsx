import React, { useState, useEffect, useRef } from 'react';
import * as L from 'leaflet';
import {
  PipelineStatus,
  PipelineSummary,
  SituationReportResponse,
} from '../types';
import { DraggableLegend } from './DraggableLegend';
import { formatValue } from '../constants';
import {
  FileText,
  Map as MapIcon,
  Download,
  Printer,
  Activity,
} from 'lucide-react';

interface ResultsViewProps {
  jobId: string;
  status: PipelineStatus | null;
  reportData: SituationReportResponse | null;
  language: 'en' | 'ne';
  onNewAnalysis: () => void;
  onRefresh: () => void;
}

export const ResultsViewPage: React.FC<ResultsViewProps> = ({
  jobId,
  status,
  reportData,
  language,
  onNewAnalysis,
  onRefresh,
}) => {
  const [activeTab, setActiveTab] = useState<'map' | 'report' | 'provenance'>('map');
  const [basemapType, setBasemapType] = useState<'satellite' | 'street'>('satellite');

  // Layer Visibility
  const [layerVisibility, setLayerVisibility] = useState<Record<string, boolean>>({
    aoi: true,
    flood: true,
    blocked_roads: true,
    isolated_settlements: true,
    hospitals: true,
    bridges: false,
    buildings: false,
    roads: false,
    change_zones: false,
  });

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const leafletMapRef = useRef<L.Map | null>(null);
  const layerGroupsRef = useRef<Record<string, L.GeoJSON>>({});
  const satelliteTileRef = useRef<L.TileLayer | null>(null);
  const streetTileRef = useRef<L.TileLayer | null>(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || leafletMapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      zoomControl: true,
    }).setView([27.95, 85.35], 11);

    const satTile = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: 'Tiles &copy; Esri World Imagery',
        maxZoom: 19,
      }
    );

    const strTile = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
      }
    );

    satTile.addTo(map);
    satelliteTileRef.current = satTile;
    streetTileRef.current = strTile;
    leafletMapRef.current = map;
    (window as any).__leafletMap = map;
    (window as any).__layerGroups = layerGroupsRef.current;

    return () => {
      map.remove();
      leafletMapRef.current = null;
    };
  }, []);

  // Handle Basemap Switch
  const handleBasemapChange = (type: 'satellite' | 'street') => {
    const map = leafletMapRef.current;
    if (!map || !satelliteTileRef.current || !streetTileRef.current) return;

    if (type === 'satellite') {
      if (map.hasLayer(streetTileRef.current)) map.removeLayer(streetTileRef.current);
      if (!map.hasLayer(satelliteTileRef.current)) map.addLayer(satelliteTileRef.current);
    } else {
      if (map.hasLayer(satelliteTileRef.current)) map.removeLayer(satelliteTileRef.current);
      if (!map.hasLayer(streetTileRef.current)) map.addLayer(streetTileRef.current);
    }
    setBasemapType(type);
  };

  // Load Layers for this specific Job ID
  useEffect(() => {
    if (!leafletMapRef.current || !jobId) return;
    loadAllLayers(jobId);
  }, [jobId]);

  const loadAllLayers = async (id: string) => {
    const map = leafletMapRef.current;
    if (!map) return;

    // Clear previous layers
    Object.values(layerGroupsRef.current).forEach((l) => map.removeLayer(l));
    layerGroupsRef.current = {};

    const layerDefs: { name: string; style: L.PathOptions }[] = [
      { name: 'aoi', style: { color: '#38bdf8', weight: 2.5, fillOpacity: 0.05, dashArray: '5, 5' } },
      { name: 'flood', style: { color: '#0284c7', fillColor: '#38bdf8', fillOpacity: 0.65, weight: 1.5 } },
      { name: 'blocked_roads', style: { color: '#e11d48', weight: 4.5, opacity: 0.95 } },
      { name: 'roads', style: { color: '#64748b', weight: 1.5, opacity: 0.6 } },
      { name: 'buildings', style: { color: '#ea580c', fillColor: '#f97316', fillOpacity: 0.7, weight: 0.8 } },
      { name: 'bridges', style: { color: '#06b6d4', fillColor: '#22d3ee', fillOpacity: 1, weight: 3 } },
      { name: 'isolated_settlements', style: { color: '#a855f7', fillColor: '#c084fc', fillOpacity: 0.95, weight: 2 } },
      { name: 'hospitals', style: { color: '#10b981', fillColor: '#34d399', fillOpacity: 0.95, weight: 2 } },
      { name: 'change_zones', style: { color: '#eab308', fillColor: '#fde047', fillOpacity: 0.4, weight: 1.5, dashArray: '4, 4' } },
    ];

    let allBounds: L.LatLngBounds | null = null;

    for (const def of layerDefs) {
      try {
        const res = await fetch(`/analysis/${id}/layers/${def.name}`);
        if (!res.ok) continue;
        const geojson = await res.json();
        if (!geojson || !geojson.features || geojson.features.length === 0) continue;

        const geoLayer = L.geoJSON(geojson, {
          interactive: def.name !== 'aoi',
          style: () => def.style,
          pointToLayer: (_feat, latlng) => {
            let radius = 6;
            let fillColor = def.style.fillColor || '#38bdf8';
            if (def.name === 'hospitals') {
              radius = 8;
              fillColor = '#10b981';
            } else if (def.name === 'isolated_settlements') {
              radius = 8;
              fillColor = '#a855f7';
            } else if (def.name === 'bridges') {
              radius = 7;
              fillColor = '#06b6d4';
            }
            return L.circleMarker(latlng, {
              radius,
              fillColor,
              color: '#fff',
              weight: 2,
              opacity: 1,
              fillOpacity: 0.95,
            });
          },
          onEachFeature: (feature, layer) => {
            const props = feature.properties || {};
            let popupContent = `<div style="font-family:sans-serif;font-size:12px;color:#0f172a;padding:4px;line-height:1.4;">
              <strong style="color:#0284c7;display:block;margin-bottom:4px;text-transform:capitalize;border-bottom:1px solid #e2e8f0;padding-bottom:2px;">${def.name.replace(/_/g, ' ')} Feature</strong>`;
            for (const [k, v] of Object.entries(props)) {
              if (typeof v !== 'object' && v !== null && k !== 'job_id') {
                let displayVal = v;
                if (typeof v === 'number') {
                  if (k.endsWith('_m')) {
                    displayVal = `${v.toLocaleString()} m`;
                  } else if (k.endsWith('_km') || k.endsWith('_km2') || k.endsWith('_m2')) {
                    displayVal = `${v.toLocaleString()}`;
                  } else if (k.includes('percentage') || k.includes('pct')) {
                    displayVal = `${v.toFixed(1)}%`;
                  }
                }
                const formattedKey = k.replace(/_/g, ' ');
                popupContent += `<div><span style="color:#64748b;text-transform:capitalize;">${formattedKey}:</span> <strong style="color:#1e293b;">${displayVal}</strong></div>`;
              }
            }
            popupContent += '</div>';
            layer.bindPopup(popupContent);
          },
        });

        layerGroupsRef.current[def.name] = geoLayer;
        (window as any).__layerGroups = layerGroupsRef.current;

        if (layerVisibility[def.name]) {
          geoLayer.addTo(map);
        }

        const b = geoLayer.getBounds();
        if (b.isValid()) {
          if (!allBounds) allBounds = b;
          else allBounds.extend(b);
        }
      } catch (err) {
        console.warn(`Layer ${def.name} note:`, err);
      }
    }

    if (allBounds && allBounds.isValid()) {
      map.fitBounds(allBounds, { padding: [30, 30] });
    }
  };

  const toggleLayer = (layerName: string) => {
    const nextVis = !layerVisibility[layerName];
    setLayerVisibility((prev) => ({ ...prev, [layerName]: nextVis }));

    const layer = layerGroupsRef.current[layerName];
    const map = leafletMapRef.current;
    if (!layer || !map) return;

    if (nextVis) {
      if (!map.hasLayer(layer)) map.addLayer(layer);
    } else {
      if (map.hasLayer(layer)) map.removeLayer(layer);
    }
  };

  const focusLayer = (layerName: string) => {
    setActiveTab('map');
    setLayerVisibility((prev) => ({ ...prev, [layerName]: true }));

    const layer = layerGroupsRef.current[layerName];
    const map = leafletMapRef.current;
    if (!layer || !map) return;

    if (!map.hasLayer(layer)) map.addLayer(layer);

    try {
      if ('getBounds' in layer && typeof (layer as any).getBounds === 'function') {
        const b = (layer as any).getBounds();
        if (b && b.isValid && b.isValid()) {
          map.flyToBounds(b, { padding: [50, 50], maxZoom: 16, duration: 1.0 });
        }
      }
    } catch (e) {
      console.warn('Could not fit bounds for:', layerName, e);
    }
  };

  const summary: PipelineSummary = status?.summary || {};

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-slate-950 text-slate-100">
      {/* 1. TOP METRICS RIBBON */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 p-3 bg-slate-900 border-b border-slate-800 shrink-0">
        {/* Metric 1: Flood Inundation */}
        <div
          onClick={() => focusLayer('flood')}
          className="bg-slate-950 border border-sky-900/60 hover:border-sky-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-sky-300 uppercase font-semibold flex items-center justify-between">
            <span>Flood Extent</span>
            <span className="text-[10px] text-sky-400">🔍</span>
          </div>
          <div className="text-base font-bold text-sky-400 font-mono mt-0.5">
            {summary.flood_area_km2 !== null && summary.flood_area_km2 !== undefined
              ? `${Number(summary.flood_area_km2).toFixed(2)} km²`
              : '—'}
          </div>
          <div className="text-[9px] text-slate-500">Spatial Inundation</div>
        </div>

        {/* Metric 2: Exposed Buildings */}
        <div
          onClick={() => focusLayer('buildings')}
          className="bg-slate-950 border border-orange-900/60 hover:border-orange-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-orange-300 uppercase font-semibold flex items-center justify-between">
            <span>Exposed Bldgs</span>
            <span className="text-[10px] text-orange-400">🔍</span>
          </div>
          <div className="text-base font-bold text-orange-400 font-mono mt-0.5">
            {formatValue(summary.affected_buildings)}
          </div>
          <div className="text-[9px] text-slate-500">Inundation Zone</div>
        </div>

        {/* Metric 3: Blocked Roads */}
        <div
          onClick={() => focusLayer('blocked_roads')}
          className="bg-slate-950 border border-rose-900/60 hover:border-rose-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-rose-300 uppercase font-semibold flex items-center justify-between">
            <span>Blocked Roads</span>
            <span className="text-[10px] text-rose-400">🔍</span>
          </div>
          <div className="text-base font-bold text-rose-400 font-mono mt-0.5">
            {formatValue(summary.blocked_roads)} segs
          </div>
          <div className="text-[9px] text-slate-500">Impasse Segments</div>
        </div>

        {/* Metric 4: Affected Road Length */}
        <div
          onClick={() => focusLayer('roads')}
          className="bg-slate-950 border border-slate-700/60 hover:border-slate-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-slate-200 uppercase font-semibold flex items-center justify-between">
            <span>Road Length</span>
            <span className="text-[10px] text-slate-300">🔍</span>
          </div>
          <div className="text-base font-bold text-slate-200 font-mono mt-0.5">
            {summary.affected_roads_km !== null && summary.affected_roads_km !== undefined
              ? `${Number(summary.affected_roads_km).toFixed(2)} km`
              : '—'}
          </div>
          <div className="text-[9px] text-slate-500">Intersected</div>
        </div>

        {/* Metric 5: Exposed Bridges */}
        <div
          onClick={() => focusLayer('bridges')}
          className="bg-slate-950 border border-cyan-900/60 hover:border-cyan-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-cyan-300 uppercase font-semibold flex items-center justify-between">
            <span>Bridges</span>
            <span className="text-[10px] text-cyan-400">🔍</span>
          </div>
          <div className="text-base font-bold text-cyan-400 font-mono mt-0.5">
            {formatValue(summary.affected_bridges)}
          </div>
          <div className="text-[9px] text-slate-500">Structural Assets</div>
        </div>

        {/* Metric 6: Isolated Settlements */}
        <div
          onClick={() => focusLayer('isolated_settlements')}
          className="bg-slate-950 border border-purple-900/60 hover:border-purple-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-purple-300 uppercase font-semibold flex items-center justify-between">
            <span>Isolated Towns</span>
            <span className="text-[10px] text-purple-400">🔍</span>
          </div>
          <div className="text-base font-bold text-purple-400 font-mono mt-0.5">
            {formatValue(summary.isolated_settlements)}
          </div>
          <div className="text-[9px] text-slate-500">0 Exit Routes</div>
        </div>

        {/* Metric 7: Hospital Access Lost */}
        <div
          onClick={() => focusLayer('hospitals')}
          className="bg-slate-950 border border-emerald-900/60 hover:border-emerald-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-emerald-300 uppercase font-semibold flex items-center justify-between">
            <span>Hospital Access</span>
            <span className="text-[10px] text-emerald-400">🔍</span>
          </div>
          <div className="text-base font-bold text-emerald-400 font-mono mt-0.5">
            {formatValue(summary.hospital_access_lost)}
          </div>
          <div className="text-[9px] text-slate-500">Communities Cut Off</div>
        </div>

        {/* Metric 8: Terrain Debris */}
        <div
          onClick={() => focusLayer('change_zones')}
          className="bg-slate-950 border border-amber-900/60 hover:border-amber-400 rounded-xl p-2.5 cursor-pointer transition select-none group"
        >
          <div className="text-[10px] text-slate-400 group-hover:text-amber-300 uppercase font-semibold flex items-center justify-between">
            <span>Terrain Debris</span>
            <span className="text-[10px] text-amber-400">🔍</span>
          </div>
          <div className="text-base font-bold text-amber-300 font-mono mt-0.5">
            {formatValue(summary.debris_zones)}
          </div>
          <div className="text-[9px] text-slate-500">Candidate Zones</div>
        </div>
      </div>

      {/* 2. SUBHEADER NAVIGATION TABS */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900/90 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('map')}
            className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
              activeTab === 'map' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <MapIcon className="w-3.5 h-3.5" />
            <span>Interactive Map & Overlays</span>
          </button>
          <button
            onClick={() => setActiveTab('report')}
            className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
              activeTab === 'report' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Situation Report ({language.toUpperCase()})</span>
          </button>
          <button
            onClick={() => setActiveTab('provenance')}
            className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
              activeTab === 'provenance' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Provenance & Methodology</span>
          </button>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div className="font-mono text-slate-400">
            Active Job: <strong className="text-sky-400">{jobId}</strong>
          </div>
          <button
            onClick={onRefresh}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-200 text-xs font-medium transition"
          >
            ↻ Refresh State
          </button>
          <button
            onClick={onNewAnalysis}
            className="px-2.5 py-1 bg-sky-600 hover:bg-sky-500 rounded-lg text-white text-xs font-medium transition"
          >
            ＋ Configure New AOI
          </button>
        </div>
      </div>

      {/* 3. MAIN TAB CONTENT */}
      {/* TAB A: INTERACTIVE MAP */}
      <div className={`flex-1 relative ${activeTab === 'map' ? 'block' : 'hidden'}`}>
        <div ref={mapContainerRef} className="w-full h-full" />

        {/* Basemap Toggle (Top Right) */}
        <div className="absolute top-4 right-4 z-[1000] bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl p-1 shadow-xl flex items-center gap-1 text-xs">
          <button
            onClick={() => handleBasemapChange('satellite')}
            className={`px-3 py-1 rounded-lg font-medium transition ${
              basemapType === 'satellite' ? 'bg-sky-600 text-white' : 'text-slate-300 hover:text-white'
            }`}
          >
            🛰️ Satellite Basemap
          </button>
          <button
            onClick={() => handleBasemapChange('street')}
            className={`px-3 py-1 rounded-lg font-medium transition ${
              basemapType === 'street' ? 'bg-sky-600 text-white' : 'text-slate-300 hover:text-white'
            }`}
          >
            🗺️ Street Map
          </button>
        </div>

        {/* Draggable & Collapsible Floating Legend */}
        <DraggableLegend
          layerVisibility={layerVisibility}
          availableLayers={status?.available_layers || []}
          onToggleLayer={toggleLayer}
          language={language}
          counts={{
            flood: summary.flood_area_km2 ? `${summary.flood_area_km2.toFixed(1)} km²` : undefined,
            buildings: summary.affected_buildings ? formatValue(summary.affected_buildings) : undefined,
            blocked_roads: summary.blocked_roads ? formatValue(summary.blocked_roads) : undefined,
            isolated_settlements: summary.isolated_settlements ? formatValue(summary.isolated_settlements) : undefined,
            hospitals: summary.hospital_access_lost ? formatValue(summary.hospital_access_lost) : undefined,
            bridges: summary.affected_bridges ? formatValue(summary.affected_bridges) : undefined,
            change_zones: summary.debris_zones ? formatValue(summary.debris_zones) : undefined,
          }}
        />
      </div>

      {/* TAB B: SITUATION REPORT */}
      <div className={`flex-1 overflow-y-auto p-6 space-y-6 ${activeTab === 'report' ? 'block' : 'hidden'}`}>
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <span className="text-xs bg-sky-950 border border-sky-700/60 text-sky-300 px-3 py-1 rounded-full font-mono font-medium">
                {language === 'ne' ? 'नेपाली मानवीय स्थिति प्रतिवेदन' : 'Humanitarian Situation Report'}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `report_${jobId}_${language}.json`;
                    a.click();
                  }}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-xs text-slate-300 flex items-center gap-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download JSON</span>
                </button>
                <button
                  onClick={() => window.print()}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-xs text-slate-300 flex items-center gap-1.5 transition"
                >
                  <Printer className="w-3.5 h-3.5" />
                  <span>Print Brief</span>
                </button>
              </div>
            </div>

            <h2 className="text-xl font-bold text-white font-heading">
              {reportData?.report?.title || `Disaster Assessment Briefing (Job ID: ${jobId})`}
            </h2>

            {reportData?.report?.executive_summary && (
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs text-slate-300 leading-relaxed">
                <strong className="text-sky-400 block mb-1 uppercase tracking-wider">Executive Summary:</strong>
                {reportData.report.executive_summary}
              </div>
            )}

            {/* Sections */}
            <div className="space-y-4 pt-2">
              {(reportData?.report?.sections || []).map((sec, idx) => (
                <div key={idx} className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-4 space-y-2">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2 font-heading">
                    <span className="w-2 h-2 rounded-full bg-sky-400" />
                    <span>{sec.title || sec.heading || `Section ${idx + 1}`}</span>
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">
                    {sec.content}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* TAB C: PROVENANCE & METHODOLOGY */}
      <div className={`flex-1 overflow-y-auto p-6 space-y-6 ${activeTab === 'provenance' ? 'block' : 'hidden'}`}>
        <div className="max-w-4xl mx-auto bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6 text-xs">
          <div>
            <h2 className="text-lg font-bold text-white font-heading mb-1">
              Methodology, Provenance & Validation Framework
            </h2>
            <p className="text-slate-400">
              Audit trail of data sources, AI segmentation models, and topological network graph algorithms.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <strong className="text-sky-400 block font-heading">1. Satellite Radar & Optical Processing</strong>
              <p className="text-slate-300 leading-relaxed">
                Sentinel-1 SAR dual-polarization (VV/VH) Ground Range Detected (GRD) acquisitions calibrated to sigma0 (dB)
                and terrain-flattened to eliminate topographic distortions. Sentinel-2 multispectral Level-2A utilized for NDWI/MNDWI validation.
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <strong className="text-orange-400 block font-heading">2. FloodUNet Deep Learning Model</strong>
              <p className="text-slate-300 leading-relaxed">
                Trained on the NeurIPS 2024 Kuro Siwo global multi-temporal SAR benchmark.
                Encoder-decoder architecture with skip connections performing binary pixel-level segmentation with 0.5 confidence threshold.
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <strong className="text-rose-400 block font-heading">3. OSM Infrastructure & Road Intersections</strong>
              <p className="text-slate-300 leading-relaxed">
                Pre-event OpenStreetMap building footprints and highway LineStrings queried via Overpass API.
                Geodesic spatial intersection computes damaged/blocked segments.
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <strong className="text-emerald-400 block font-heading">4. NetworkX Reachability Analysis</strong>
              <p className="text-slate-300 leading-relaxed">
                Road network represented as a planar graph G(V, E). Inundated edges are removed to evaluate
                connected components, identifying completely isolated settlements and critical emergency hospital access loss.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

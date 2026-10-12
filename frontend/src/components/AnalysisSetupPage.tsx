import React, { useState, useEffect, useRef } from 'react';
import * as L from 'leaflet';
import {
  MapPin,
  Calendar,
  ArrowRight,
  Upload,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Satellite,
  Compass,
  Edit3,
} from 'lucide-react';
import {
  PRESET_AOIS,
  calculateBBoxArea,
} from '../constants';

interface AnalysisSetupProps {
  onStartAnalysis: (payload: {
    aoi: any;
    floodDate: string;
    comparisonDate?: string;
    referenceDate?: string;
    preWindowDays: number;
    postWindowDays: number;
    language: 'en' | 'ne';
    demoMode: boolean;
  }) => void;
  language: 'en' | 'ne';
  isConnecting?: boolean;
}

export const AnalysisSetupPage: React.FC<AnalysisSetupProps> = ({
  onStartAnalysis,
  language,
  isConnecting = false,
}) => {
  // Mode of AOI specification
  const [aoiMethod, setAoiMethod] = useState<'preset' | 'bbox' | 'geojson' | 'draw'>('preset');
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('nepal_trishuli');

  // Coordinates
  const [coords, setCoords] = useState({
    minLon: 85.20,
    minLat: 27.80,
    maxLon: 85.50,
    maxLat: 28.10,
  });

  // Raw string inputs for resilient editing without NaN clearing the map
  const [rawCoords, setRawCoords] = useState({
    minLon: '85.20',
    minLat: '27.80',
    maxLon: '85.50',
    maxLat: '28.10',
  });

  // Dates
  const [floodDate, setFloodDate] = useState<string>('2026-08-26');
  const [comparisonDate, setComparisonDate] = useState<string>('2026-08-31');
  const [referenceDate] = useState<string>('2026-08-19');
  const [preWindowDays, setPreWindowDays] = useState<number>(15);
  const [postWindowDays, setPostWindowDays] = useState<number>(7);

  // GeoJSON file upload
  const [geoJsonData, setGeoJsonData] = useState<any | null>(null);
  const [geoJsonError, setGeoJsonError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  // Execution options
  const [demoMode, setDemoMode] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Leaflet map setup for AOI preview & interactive inspection
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const aoiLayerRef = useRef<L.GeoJSON | null>(null);

  // Initialize preview map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      zoomControl: true,
      attributionControl: false,
    }).setView([27.95, 85.35], 10);

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update map AOI shape whenever coords or preset changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (aoiLayerRef.current) {
      map.removeLayer(aoiLayerRef.current);
      aoiLayerRef.current = null;
    }

    let geometry: any = null;

    if (aoiMethod === 'geojson' && geoJsonData) {
      geometry = geoJsonData;
    } else {
      const { minLon, minLat, maxLon, maxLat } = coords;
      if (
        Number.isFinite(minLon) &&
        Number.isFinite(minLat) &&
        Number.isFinite(maxLon) &&
        Number.isFinite(maxLat) &&
        minLon < maxLon &&
        minLat < maxLat
      ) {
        geometry = {
          type: 'Polygon',
          coordinates: [
            [
              [minLon, minLat],
              [maxLon, minLat],
              [maxLon, maxLat],
              [minLon, maxLat],
              [minLon, minLat],
            ],
          ],
        };
      }
    }

    if (geometry) {
      const layer = L.geoJSON(geometry, {
        style: {
          color: '#38bdf8',
          weight: 2.5,
          dashArray: '5, 5',
          fillColor: '#0284c7',
          fillOpacity: 0.15,
        },
      }).addTo(map);

      aoiLayerRef.current = layer;
      const b = layer.getBounds();
      if (b.isValid()) {
        map.fitBounds(b, { padding: [40, 40], maxZoom: 13 });
      }
    }
  }, [coords, aoiMethod, geoJsonData]);

  // Handle Preset selection
  const handleSelectPreset = (presetId: string) => {
    const p = PRESET_AOIS.find((item) => item.id === presetId);
    if (!p) return;
    setSelectedPreset(presetId);
    setCoords({ ...p.bbox });
    setRawCoords({
      minLon: String(p.bbox.minLon),
      minLat: String(p.bbox.minLat),
      maxLon: String(p.bbox.maxLon),
      maxLat: String(p.bbox.maxLat),
    });
    setFloodDate(p.defaultEventDate);
    setComparisonDate(p.defaultCompDate);
    setErrorMsg(null);
  };

  // Handle manual coord change with raw string buffer
  const handleCoordChange = (field: keyof typeof coords, valStr: string) => {
    setRawCoords((prev) => ({ ...prev, [field]: valStr }));
    setSelectedPreset('custom');
    setErrorMsg(null);

    const parsed = parseFloat(valStr);
    if (!isNaN(parsed) && isFinite(parsed)) {
      setCoords((prev) => ({ ...prev, [field]: parsed }));
    }
  };

  // Handle GeoJSON File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setGeoJsonError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = JSON.parse(text);

        // Normalize feature or collection
        let geom: any = null;
        if (parsed.type === 'FeatureCollection' && parsed.features?.length > 0) {
          geom = parsed.features[0].geometry;
        } else if (parsed.type === 'Feature') {
          geom = parsed.geometry;
        } else if (parsed.type === 'Polygon' || parsed.type === 'MultiPolygon') {
          geom = parsed;
        }

        if (!geom) {
          throw new Error('GeoJSON must contain a valid Polygon or MultiPolygon geometry.');
        }

        setGeoJsonData(geom);
        setAoiMethod('geojson');
      } catch (err: any) {
        setGeoJsonError(err.message || 'Invalid GeoJSON file format.');
        setGeoJsonData(null);
      }
    };
    reader.readAsText(file);
  };

  // Submit and launch pipeline
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate coordinates
    if (aoiMethod !== 'geojson') {
      if (coords.minLon >= coords.maxLon || coords.minLat >= coords.maxLat) {
        setErrorMsg('Invalid coordinates: min values must be less than max values.');
        return;
      }
      if (coords.minLat < -90 || coords.maxLat > 90 || coords.minLon < -180 || coords.maxLon > 180) {
        setErrorMsg('Coordinates out of range (-90 to 90 lat, -180 to 180 lon).');
        return;
      }
    } else if (!geoJsonData) {
      setErrorMsg('Please upload a valid GeoJSON polygon first.');
      return;
    }

    let aoiPayload: any;
    if (aoiMethod === 'geojson') {
      aoiPayload = geoJsonData;
    } else {
      aoiPayload = [coords.minLon, coords.minLat, coords.maxLon, coords.maxLat];
    }

    onStartAnalysis({
      aoi: aoiPayload,
      floodDate,
      comparisonDate,
      referenceDate,
      preWindowDays,
      postWindowDays,
      language,
      demoMode,
    });
  };

  const currentAreaKm2 = calculateBBoxArea(coords.minLon, coords.minLat, coords.maxLon, coords.maxLat);

  return (
    <div className="flex-1 flex flex-col lg:flex-row overflow-hidden bg-slate-950 text-slate-100">
      {/* LEFT: Setup Form (480px) */}
      <div className="w-full lg:w-[480px] bg-slate-900 border-r border-slate-800 p-6 flex flex-col overflow-y-auto space-y-6 shrink-0 shadow-2xl">
        {/* Header Hero */}
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-sky-950/80 border border-sky-800/80 text-sky-400 text-[11px] font-mono font-medium mb-3">
            <Compass className="w-3.5 h-3.5" />
            <span>DISASTER MAPPING WORKFLOW</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-heading">
            {language === 'ne' ? 'बाढी प्रभाव नक्सांकन — नयाँ विश्लेषण' : 'Map Flood Impact. Understand Risk.'}
          </h1>
          <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
            {language === 'ne'
              ? 'भूउपग्रह तस्विर (Sentinel-1 SAR तथा Sentinel-2) र खुला पूर्वाधार तथ्यांक (OSM) तुलना गरी बाढी प्रभावित क्षेत्र तथा सडक पहुँच विश्लेषण गर्नुहोस्।'
              : 'Compare pre/post disaster satellite radar (Sentinel-1) & optical imagery (Sentinel-2) with OpenStreetMap infrastructure to evaluate inundated roads and hospital access.'}
          </p>
        </div>

        {errorMsg && (
          <div className="bg-rose-950/70 border border-rose-800 p-3 rounded-xl text-rose-300 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <span>{errorMsg}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5 flex-1">
          {/* STEP 1: AOI Selection */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-3.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-heading flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-sky-400" />
                <span>1. Area of Interest (AOI)</span>
              </span>
              <span className="text-[11px] font-mono text-sky-400 font-semibold">
                ~{currentAreaKm2} km²
              </span>
            </div>

            {/* Input Method Switcher */}
            <div className="grid grid-cols-4 gap-1 p-1 bg-slate-900 rounded-xl border border-slate-800 text-xs font-medium">
              <button
                type="button"
                onClick={() => setAoiMethod('draw')}
                className={`py-1.5 rounded-lg transition text-[11px] ${
                  aoiMethod === 'draw'
                    ? 'bg-sky-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                ✏️ Draw
              </button>
              <button
                type="button"
                onClick={() => setAoiMethod('preset')}
                className={`py-1.5 rounded-lg transition text-[11px] ${
                  aoiMethod === 'preset'
                    ? 'bg-sky-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Presets
              </button>
              <button
                type="button"
                onClick={() => setAoiMethod('bbox')}
                className={`py-1.5 rounded-lg transition text-[11px] ${
                  aoiMethod === 'bbox'
                    ? 'bg-sky-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Coords
              </button>
              <button
                type="button"
                onClick={() => setAoiMethod('geojson')}
                className={`py-1.5 rounded-lg transition text-[11px] ${
                  aoiMethod === 'geojson'
                    ? 'bg-sky-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                GeoJSON
              </button>
            </div>

            {/* Method: Draw on Map */}
            {aoiMethod === 'draw' && (
              <div className="space-y-2 p-3 bg-slate-900/80 rounded-xl border border-sky-500/30">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                    <Edit3 className="w-3.5 h-3.5 text-sky-400" />
                    Interactive Map Canvas
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 font-medium">
                    Active
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Click and drag directly on the satellite map to the right to define a rectangular Area of Interest, or click below to trigger a sample drawn bounding box.
                </p>
                <div className="flex gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => {
                      const newBbox = { minLon: 85.25, minLat: 27.88, maxLon: 85.45, maxLat: 28.05 };
                      setCoords(newBbox);
                      setRawCoords({
                        minLon: '85.25',
                        minLat: '27.88',
                        maxLon: '85.45',
                        maxLat: '28.05',
                      });
                    }}
                    className="flex-1 py-1.5 px-3 rounded-lg bg-sky-600/30 hover:bg-sky-600/50 border border-sky-500/40 text-sky-200 text-xs font-medium transition"
                  >
                    Draw Sample AOI (Trishuli Valley)
                  </button>
                </div>
              </div>
            )}

            {/* Method A: Presets */}
            {aoiMethod === 'preset' && (
              <div className="space-y-2">
                <select
                  value={selectedPreset}
                  onChange={(e) => handleSelectPreset(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500 font-medium"
                >
                  {PRESET_AOIS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} (~{p.approxAreaKm2} km²)
                    </option>
                  ))}
                </select>
                {selectedPreset && (
                  <p className="text-[11px] text-slate-400 italic px-1">
                    {PRESET_AOIS.find((p) => p.id === selectedPreset)?.description}
                  </p>
                )}
              </div>
            )}

            {/* Method B: Manual Coordinates */}
            {aoiMethod === 'bbox' && (
              <div className="space-y-2">
                <div className="text-[11px] text-slate-400 font-mono flex justify-between">
                  <span>Bounding Box [W, S, E, N]</span>
                  <span>EPSG:4326</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <label className="text-[10px] text-slate-400 block mb-0.5">Min Lon (West)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={rawCoords.minLon}
                      onChange={(e) => handleCoordChange('minLon', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono focus:border-sky-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-slate-400 block mb-0.5">Min Lat (South)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={rawCoords.minLat}
                      onChange={(e) => handleCoordChange('minLat', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono focus:border-sky-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-slate-400 block mb-0.5">Max Lon (East)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={rawCoords.maxLon}
                      onChange={(e) => handleCoordChange('maxLon', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono focus:border-sky-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-slate-400 block mb-0.5">Max Lat (North)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={rawCoords.maxLat}
                      onChange={(e) => handleCoordChange('maxLat', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono focus:border-sky-500 focus:outline-none"
                    />
                  </div>
                </div>
                {coords.minLon >= coords.maxLon || coords.minLat >= coords.maxLat ? (
                  <p className="text-[11px] text-amber-400 font-medium px-1">
                    ⚠️ West must be less than East, and South must be less than North.
                  </p>
                ) : null}
              </div>
            )}

            {/* Method C: Upload GeoJSON */}
            {aoiMethod === 'geojson' && (
              <div className="space-y-2">
                <label className="border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition bg-slate-900/50 hover:bg-slate-900">
                  <Upload className="w-5 h-5 text-sky-400 mb-1" />
                  <span className="text-xs font-semibold text-slate-200">
                    {fileName ? fileName : 'Upload GeoJSON Polygon'}
                  </span>
                  <span className="text-[10px] text-slate-500 mt-0.5">.geojson, .json format</span>
                  <input
                    type="file"
                    accept=".geojson,.json"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>
                {geoJsonError && (
                  <p className="text-[11px] text-rose-400">{geoJsonError}</p>
                )}
                {geoJsonData && (
                  <p className="text-[11px] text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Polygon loaded successfully
                  </p>
                )}
              </div>
            )}
          </div>

          {/* STEP 2: Disaster & Satellite Dates */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-3.5">
            <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-heading flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-cyan-400" />
              <span>2. Disaster & Satellite Dates</span>
            </span>

            {/* Event Date vs Comparison Date */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] font-medium text-slate-300 mb-1">
                  Flood Event Date
                </label>
                <input
                  type="date"
                  value={floodDate}
                  onChange={(e) => setFloodDate(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-sky-500 font-mono"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">Reported incident date</span>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-slate-300 mb-1">
                  Target Comparison Date
                </label>
                <input
                  type="date"
                  value={comparisonDate}
                  onChange={(e) => setComparisonDate(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-sky-500 font-mono"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">Satellite observation</span>
              </div>
            </div>

            {/* Temporal Search Windows */}
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800/80 space-y-2">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Baseline Pre-Event Window:</span>
                <span className="font-mono text-sky-400 font-semibold">{preWindowDays} days prior</span>
              </div>
              <input
                type="range"
                min="5"
                max="30"
                value={preWindowDays}
                onChange={(e) => setPreWindowDays(parseInt(e.target.value))}
                className="w-full accent-sky-500 h-1 bg-slate-800 rounded-lg cursor-pointer"
              />

              <div className="flex items-center justify-between text-[11px] pt-1">
                <span className="text-slate-400">Post-Event Search Window:</span>
                <span className="font-mono text-cyan-400 font-semibold">{postWindowDays} days after</span>
              </div>
              <input
                type="range"
                min="3"
                max="21"
                value={postWindowDays}
                onChange={(e) => setPostWindowDays(parseInt(e.target.value))}
                className="w-full accent-cyan-500 h-1 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>
          </div>

          {/* STEP 3: Execution Mode */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-2">
            <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-heading flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-amber-400" />
              <span>3. Processing Mode</span>
            </span>

            <div className="space-y-1.5 text-xs">
              <label className="flex items-start gap-2.5 p-2 rounded-xl border cursor-pointer transition border-slate-800 bg-slate-900/50 hover:bg-slate-900">
                <input
                  type="radio"
                  name="execMode"
                  checked={!demoMode}
                  onChange={() => setDemoMode(false)}
                  className="accent-sky-500 mt-0.5"
                />
                <div>
                  <strong className="text-slate-200 block text-xs">Real Satellite Pipeline</strong>
                  <span className="text-[10px] text-slate-400">
                    Discovers actual CDSE scenes, queries Overpass OSM, evaluates graph accessibility.
                  </span>
                </div>
              </label>

              <label className="flex items-start gap-2.5 p-2 rounded-xl border cursor-pointer transition border-slate-800 bg-slate-900/50 hover:bg-slate-900">
                <input
                  type="radio"
                  name="execMode"
                  checked={demoMode}
                  onChange={() => setDemoMode(true)}
                  className="accent-sky-500 mt-0.5"
                />
                <div>
                  <strong className="text-slate-200 block text-xs">Demonstration Mode</strong>
                  <span className="text-[10px] text-slate-400">
                    Generates synthesized flood extent mask scaled to your AOI for testing UI layers.
                  </span>
                </div>
              </label>
            </div>
          </div>

          {/* Submit Action */}
          <button
            type="submit"
            disabled={isConnecting}
            className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-sky-600 via-sky-500 to-cyan-500 hover:from-sky-500 hover:to-cyan-400 active:scale-[0.99] text-white font-bold text-sm shadow-xl shadow-sky-600/20 transition flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {isConnecting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Creating Analysis Job...</span>
              </>
            ) : (
              <>
                <span>Launch Disaster Analysis</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>
      </div>

      {/* RIGHT: Interactive AOI Preview Map */}
      <div className="flex-1 relative flex flex-col h-64 lg:h-auto">
        <div ref={mapContainerRef} className="w-full h-full" />

        {/* Floating Map Overlay Info */}
        <div className="absolute top-4 left-4 z-[1000] bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl p-3 shadow-xl max-w-sm text-xs space-y-1">
          <div className="font-bold text-white uppercase tracking-wider font-heading flex items-center gap-2">
            <Satellite className="w-3.5 h-3.5 text-sky-400" />
            <span>AOI Spatial Footprint</span>
          </div>
          <div className="text-[11px] text-slate-300 font-mono">
            {aoiMethod === 'geojson' ? (
              <span>Geometry: Custom GeoJSON Polygon</span>
            ) : (
              <span>
                [{coords.minLon.toFixed(2)}, {coords.minLat.toFixed(2)}] → [{coords.maxLon.toFixed(2)}, {coords.maxLat.toFixed(2)}]
              </span>
            )}
          </div>
          <div className="text-[10px] text-slate-400">
            The selected AOI defines the analysis region. Available satellite scenes may cover a larger area and are clipped to this boundary.
          </div>
        </div>
      </div>
    </div>
  );
};

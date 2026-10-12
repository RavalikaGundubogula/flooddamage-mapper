import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Move,
  Eye,
  EyeOff,
} from 'lucide-react';

export interface LayerItem {
  key: string;
  labelEn: string;
  labelNe: string;
  color: string;
  type: 'fill' | 'line' | 'point';
  count?: number | string;
  category: 'hazard' | 'infrastructure' | 'connectivity';
}

export const LAYER_DEFINITIONS: LayerItem[] = [
  { key: 'aoi', labelEn: 'Area of Interest (AOI Boundary)', labelNe: 'विश्लेषण क्षेत्र (AOI सीमाना)', color: '#38bdf8', type: 'line', category: 'hazard' },
  { key: 'flood', labelEn: 'Flood Extent (AI Inundation)', labelNe: 'बाढी ग्रस्त क्षेत्र (AI वर्गीकरण)', color: '#0284c7', type: 'fill', category: 'hazard' },
  { key: 'blocked_roads', labelEn: 'Blocked Roads (Impasse)', labelNe: 'अवरुद्ध सडक खण्डहरू', color: '#e11d48', type: 'line', category: 'connectivity' },
  { key: 'isolated_settlements', labelEn: 'Isolated Settlements', labelNe: 'सम्पर्कविहीन बस्तीहरू', color: '#a855f7', type: 'point', category: 'connectivity' },
  { key: 'hospitals', labelEn: 'Hospitals with Lost Reachability', labelNe: 'पहुँच गुमाएका अस्पतालहरू', color: '#10b981', type: 'point', category: 'connectivity' },
  { key: 'bridges', labelEn: 'Critical Bridges in Impact Zone', labelNe: 'जोखिमयुक्त पुलहरू', color: '#06b6d4', type: 'point', category: 'infrastructure' },
  { key: 'buildings', labelEn: 'Exposed Buildings (OSM)', labelNe: 'प्रभावित भवनहरू (OSM)', color: '#ea580c', type: 'fill', category: 'infrastructure' },
  { key: 'roads', labelEn: 'Baseline Road Network', labelNe: 'सडक सञ्जाल (सञ्चालित)', color: '#64748b', type: 'line', category: 'infrastructure' },
  { key: 'change_zones', labelEn: 'Terrain Debris / Change Zones', labelNe: 'भू-परिवर्तन / पहिरो क्षेत्र', color: '#eab308', type: 'fill', category: 'hazard' },
];

interface DraggableLegendProps {
  layerVisibility: Record<string, boolean>;
  availableLayers: string[];
  onToggleLayer: (key: string) => void;
  language: 'en' | 'ne';
  counts?: Record<string, any>;
}

export const DraggableLegend: React.FC<DraggableLegendProps> = ({
  layerVisibility,
  availableLayers,
  onToggleLayer,
  language,
  counts = {},
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 20, y: 80 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragOffset({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const newX = Math.max(10, Math.min(window.innerWidth - 300, e.clientX - dragOffset.x));
    const newY = Math.max(70, Math.min(window.innerHeight - 200, e.clientY - dragOffset.y));
    setPosition({ x: newX, y: newY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Filter layers that actually exist for this job (or default hazard/aoi)
  const visibleItems = LAYER_DEFINITIONS.filter(
    (item) => availableLayers.length === 0 || availableLayers.includes(item.key) || item.key === 'aoi'
  );

  return (
    <div
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      className="absolute z-[1000] select-none shadow-2xl transition-shadow"
    >
      <div className="w-72 bg-slate-900/95 backdrop-blur-md border border-slate-700/80 rounded-xl overflow-hidden shadow-2xl text-xs">
        {/* Drag Handle & Header */}
        <div
          onMouseDown={handleMouseDown}
          className="flex items-center justify-between px-3 py-2 bg-slate-800/80 border-b border-slate-700/60 cursor-move text-slate-200"
        >
          <div className="flex items-center gap-2">
            <Move className="w-3.5 h-3.5 text-slate-400" />
            <span className="font-semibold text-[11px] uppercase tracking-wider font-heading">
              {language === 'ne' ? 'नक्सा तहहरू (Legend)' : 'Map Layers & Legend'}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="p-1 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-200 transition"
              title={collapsed ? 'Expand legend' : 'Collapse legend'}
            >
              {collapsed ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Content */}
        {!collapsed && (
          <div className="p-3 space-y-2 max-h-80 overflow-y-auto">
            <div className="text-[10px] text-slate-400 flex items-center justify-between font-mono pb-1 border-b border-slate-800">
              <span>{visibleItems.length} active layers</span>
              <span>toggle to inspect</span>
            </div>

            <div className="space-y-1">
              {visibleItems.map((item) => {
                const isVisible = !!layerVisibility[item.key];
                const countVal = counts[item.key];

                return (
                  <div
                    key={item.key}
                    onClick={() => onToggleLayer(item.key)}
                    className={`flex items-center justify-between px-2 py-1.5 rounded-lg cursor-pointer transition ${
                      isVisible
                        ? 'bg-slate-800/70 hover:bg-slate-800 text-slate-200'
                        : 'bg-slate-950/40 hover:bg-slate-950/80 text-slate-500'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0 pr-1">
                      {/* Swatch */}
                      {item.type === 'fill' && (
                        <span
                          className="w-3 h-3 rounded shrink-0 border border-white/20"
                          style={{ backgroundColor: item.color }}
                        />
                      )}
                      {item.type === 'line' && (
                        <span
                          className="w-3.5 h-1 rounded shrink-0"
                          style={{ backgroundColor: item.color }}
                        />
                      )}
                      {item.type === 'point' && (
                        <span
                          className="w-2.5 h-2.5 rounded-full shrink-0 ring-2 ring-white/20"
                          style={{ backgroundColor: item.color }}
                        />
                      )}

                      <span className="truncate text-[11px] font-medium">
                        {language === 'ne' ? item.labelNe : item.labelEn}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {countVal !== undefined && (
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-950 text-slate-300">
                          {countVal}
                        </span>
                      )}
                      {isVisible ? (
                        <Eye className="w-3.5 h-3.5 text-sky-400" />
                      ) : (
                        <EyeOff className="w-3.5 h-3.5 text-slate-600" />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

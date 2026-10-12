import React, { useState, useEffect } from 'react';
import {
  PipelineStatus,
  SituationReportResponse,
  AnalysisJobItem,
} from './types';
import { AnalysisSetupPage } from './components/AnalysisSetupPage';
import { ResultsViewPage } from './components/ResultsViewPage';
import { PIPELINE_STAGES } from './constants';
import {
  Globe,
  Compass,
  Clock,
  Layers,
} from 'lucide-react';

export const App: React.FC = () => {
  // Navigation: 'setup' | 'progress' | 'results' | 'jobs'
  const [currentPage, setCurrentPage] = useState<'setup' | 'progress' | 'results' | 'jobs'>('setup');
  const [language, setLanguage] = useState<'en' | 'ne'>('en');

  // Active Job State
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [reportData, setReportData] = useState<SituationReportResponse | null>(null);
  const [savedJobs, setSavedJobs] = useState<AnalysisJobItem[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Poll available jobs list on mount
  useEffect(() => {
    fetchSavedJobs();
  }, []);

  const fetchSavedJobs = async () => {
    try {
      const res = await fetch('/analysis');
      if (res.ok) {
        const data = await res.json();
        setSavedJobs(data.jobs || []);
      }
    } catch (e) {
      console.warn('Jobs list fetch note:', e);
    }
  };

  // Launch new analysis job with custom AOI, dates, and windows
  const handleStartAnalysis = async (payload: {
    aoi: any;
    floodDate: string;
    comparisonDate?: string;
    referenceDate?: string;
    preWindowDays: number;
    postWindowDays: number;
    language: 'en' | 'ne';
    demoMode: boolean;
  }) => {
    setIsSubmitting(true);
    setLanguage(payload.language);

    try {
      const res = await fetch('/analysis/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          aoi: payload.aoi,
          flood_date: payload.floodDate,
          comparison_date: payload.comparisonDate,
          reference_date: payload.referenceDate,
          pre_event_window_days: payload.preWindowDays,
          post_event_window_days: payload.postWindowDays,
          report_language: payload.language,
          demo_mode: payload.demoMode,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`API error (${res.status}): ${errText}`);
      }

      const initStatus: PipelineStatus = await res.json();
      setActiveJobId(initStatus.job_id);
      setPipelineStatus(initStatus);
      setCurrentPage('progress');
      startPolling(initStatus.job_id, payload.language);
    } catch (err: any) {
      alert(`Could not start analysis pipeline: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Poll active pipeline progress
  const startPolling = (jobId: string, lang: 'en' | 'ne') => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/analysis/${jobId}/status`);
        if (res.ok) {
          const st: PipelineStatus = await res.json();
          setPipelineStatus(st);

          if (
            ['completed', 'completed_with_limitations', 'blocked', 'failed'].includes(
              st.status
            )
          ) {
            clearInterval(timer);
            // Fetch report
            fetchReport(jobId, lang);
            fetchSavedJobs();
            // Automatically transition to dedicated results view
            setCurrentPage('results');
          }
        }
      } catch (err) {
        console.warn('Polling error:', err);
      }
    }, 1500);
  };

  const fetchReport = async (jobId: string, lang: 'en' | 'ne') => {
    try {
      const res = await fetch(`/analysis/${jobId}/report/${lang}`);
      if (res.ok) {
        const rep = await res.json();
        setReportData(rep);
      }
    } catch (err) {
      console.warn('Report fetch note:', err);
    }
  };

  // Open existing saved job directly into results view
  const handleOpenJob = async (jobId: string) => {
    setActiveJobId(jobId);
    try {
      const res = await fetch(`/analysis/${jobId}/status`);
      if (res.ok) {
        const st = await res.json();
        setPipelineStatus(st);
      }
      fetchReport(jobId, language);
      setCurrentPage('results');
    } catch (err) {
      console.warn('Job open note:', err);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans select-none">
      {/* 1. TOP GLOBAL NAVIGATION HEADER */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-2.5 flex items-center justify-between z-20 shrink-0">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-sky-500 to-cyan-600 flex items-center justify-center text-white shadow-md shadow-sky-600/30">
            <Globe className="w-4 h-4" />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-tight flex items-center gap-1.5 font-heading">
              <span>FloodScope</span>
              <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-sky-950 border border-sky-800 text-sky-400">
                DISASTER AI
              </span>
            </div>
            <div className="text-[10px] text-slate-400">Track B: Mapping Flood Damage from Space</div>
          </div>
        </div>

        {/* Global Navigation Links */}
        <nav className="flex items-center gap-1.5 text-xs font-medium">
          <button
            onClick={() => setCurrentPage('setup')}
            className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 ${
              currentPage === 'setup'
                ? 'bg-sky-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Configure AOI</span>
          </button>

          {activeJobId && (
            <button
              onClick={() => setCurrentPage('results')}
              className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 ${
                currentPage === 'results'
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Job Results ({activeJobId.replace('job_', '')})</span>
            </button>
          )}

          <button
            onClick={() => {
              fetchSavedJobs();
              setCurrentPage('jobs');
            }}
            className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 ${
              currentPage === 'jobs'
                ? 'bg-sky-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>Saved Jobs ({savedJobs.length})</span>
          </button>
        </nav>

        {/* Right Status & Language Controls */}
        <div className="flex items-center gap-3 text-xs">
          {/* Language Selector */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 text-[11px] font-medium">
            <button
              onClick={() => {
                setLanguage('en');
                if (activeJobId) fetchReport(activeJobId, 'en');
              }}
              className={`px-2 py-0.5 rounded transition ${
                language === 'en' ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              EN
            </button>
            <button
              onClick={() => {
                setLanguage('ne');
                if (activeJobId) fetchReport(activeJobId, 'ne');
              }}
              className={`px-2 py-0.5 rounded transition ${
                language === 'ne' ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              नेपाली
            </button>
          </div>

          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="hidden sm:inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition font-mono text-[11px]"
          >
            <span>API Docs</span>
            <span className="text-[10px]">↗</span>
          </a>
        </div>
      </header>

      {/* 2. BODY CONTENT ROUTED BY STATE */}
      {/* PAGE 1: SETUP */}
      {currentPage === 'setup' && (
        <AnalysisSetupPage
          onStartAnalysis={handleStartAnalysis}
          language={language}
          isConnecting={isSubmitting}
        />
      )}

      {/* PAGE 2: PIPELINE PROGRESS STEPPER */}
      {currentPage === 'progress' && (
        <div className="flex-1 flex flex-col items-center justify-center p-6 bg-slate-950">
          <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono text-sky-400 font-bold block mb-1">
                  JOB: {activeJobId}
                </span>
                <h2 className="text-base font-bold text-white font-heading">
                  Executing Disaster Analysis Pipeline
                </h2>
              </div>
              <span className="text-sm font-mono font-bold text-sky-400 bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800">
                {pipelineStatus?.progress_percent || 0}%
              </span>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
              <div
                className="bg-gradient-to-r from-sky-500 to-cyan-400 h-full transition-all duration-500 rounded-full"
                style={{ width: `${pipelineStatus?.progress_percent || 0}%` }}
              />
            </div>

            {/* Current Stage Message */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-ping shrink-0" />
              <span className="truncate">{pipelineStatus?.message || 'Processing stage...'}</span>
            </div>

            {/* Stepper items */}
            <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
              {PIPELINE_STAGES.map((s) => {
                const currentStage = pipelineStatus?.current_stage;
                const isCurrent = currentStage === s.id;
                return (
                  <div
                    key={s.id}
                    className={`flex items-center justify-between p-2 rounded-lg text-xs transition ${
                      isCurrent
                        ? 'bg-sky-950/80 border border-sky-800 text-sky-200'
                        : 'bg-slate-950/40 border border-slate-800/40 text-slate-400'
                    }`}
                  >
                    <span>{language === 'ne' ? s.labelNe : s.labelEn}</span>
                    <span className="font-mono text-[10px]">
                      {isCurrent ? '⚡ In progress' : ''}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* PAGE 3: DEDICATED RESULTS VIEW */}
      {currentPage === 'results' && activeJobId && (
        <ResultsViewPage
          jobId={activeJobId}
          status={pipelineStatus}
          reportData={reportData}
          language={language}
          onNewAnalysis={() => setCurrentPage('setup')}
          onRefresh={() => {
            if (activeJobId) {
              fetch(`/analysis/${activeJobId}/status`)
                .then((r) => r.json())
                .then((st) => setPipelineStatus(st));
            }
          }}
        />
      )}

      {/* PAGE 4: SAVED JOBS BROWSER */}
      {currentPage === 'jobs' && (
        <div className="flex-1 overflow-y-auto p-6 max-w-4xl mx-auto w-full space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h1 className="text-lg font-bold text-white font-heading">
                Disaster Analysis Job History
              </h1>
              <p className="text-xs text-slate-400">
                Browse pre-computed case studies and custom user-submitted runs.
              </p>
            </div>
            <button
              onClick={() => setCurrentPage('setup')}
              className="px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-xl text-xs font-semibold transition"
            >
              ＋ New Analysis
            </button>
          </div>

          <div className="space-y-2.5">
            {savedJobs.map((j) => (
              <div
                key={j.job_id}
                onClick={() => handleOpenJob(j.job_id)}
                className="bg-slate-900 border border-slate-800 hover:border-sky-500/80 rounded-xl p-4 cursor-pointer transition flex items-center justify-between group shadow-sm"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <strong className="text-sm text-white group-hover:text-sky-300 font-heading">
                      {j.title || j.job_id}
                    </strong>
                    {j.is_real_case && (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-400">
                        REAL CASE
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400">{j.description}</p>
                  <div className="text-[10px] text-slate-500 font-mono">
                    Event Date: {j.flood_date} · ID: {j.job_id}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase ${
                      j.status === 'completed'
                        ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                        : 'bg-slate-950 text-slate-400 border-slate-800'
                    }`}
                  >
                    {j.status}
                  </span>
                  <span className="text-slate-500 group-hover:text-sky-400 transition text-sm">
                    →
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

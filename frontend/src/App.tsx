
import { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadNode } from './components/UploadNode';
import { LatticeVisualizer } from './components/LatticeVisualizer';
import { ComparisonSlider } from './components/ComparisonSlider';
import { Download, AlertCircle, Cpu, Eye, Sparkles, CheckCircle, Zap, Palette, Sliders, Target, Beaker, Send, Wand2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types

interface OperationLogEntry {
  stage: string;
  action: string;
  params: Record<string, any>;
}

interface DirectorParams {
  normalization: Record<string, number>;
  tone_curve: Record<string, any>;
  palette_identity: Record<string, any>;
  selective_corrections: Record<string, any>;
}

interface AnalysisResponse {
  session_id: string;
  original_points: number[][];
  warped_points: number[][];
  pins_source: number[][];
  pins_target: number[][];
  operations_log?: OperationLogEntry[];
  director_params?: DirectorParams;
}

interface RefinementRound {
  round: number;
  params: Record<string, number>;
  feedback: string;
  satisfied: boolean;
}

interface RefineResponse {
  session_id: string;
  rounds_completed: number;
  final_params: Record<string, number>;
  history: RefinementRound[];
  preview: string;
}

// AI Recipe types
interface ColorRecipe {
  exposure_mult: number;
  brightness_offset: number;
  a_channel_shift: number;
  b_channel_shift: number;
  saturation_mult: number;
  shadow_a: number;
  shadow_b: number;
  midtone_a: number;
  midtone_b: number;
  highlight_a: number;
  highlight_b: number;
  description: string;
}

interface AIRecipeResponse {
  session_id: string;
  recipe: ColorRecipe;
  workflow: string;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const _unusedTypeCheck: AIRecipeResponse | null = null;

// Vibe Replicator types
interface VibeAnalysis {
  vibe_description: string;
  mood_keywords: string[];
  color_grading_style: string;
  technical_observations: Record<string, any>;
}

interface GradingInstructions {
  exposure_adjustment: number;
  contrast_curve: string;
  contrast_strength: number;
  shadow_a: number;
  shadow_b: number;
  midtone_a: number;
  midtone_b: number;
  highlight_a: number;
  highlight_b: number;
  saturation_mult: number;
  special_notes: string[];
  description: string;
}

interface CritiqueResult {
  vibe_match_score: number;
  issues_found: string[];
  satisfied: boolean;
  feedback: string;
}

function App() {
  const [refFile, setRefFile] = useState<File | null>(null);
  const [srcFile, setSrcFile] = useState<File | null>(null);
  const [session, setSession] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Preview state
  const [srcPreviewUrl, setSrcPreviewUrl] = useState<string | null>(null);
  const [gradedPreviewUrl, setGradedPreviewUrl] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'lattice' | 'compare' | 'result'>('lattice');


  // AI Refinement state
  const [isRefining, setIsRefining] = useState(false);
  const [refinementHistory, setRefinementHistory] = useState<RefinementRound[]>([]);
  const [isRefined, setIsRefined] = useState(false);

  // AI Recipe workflow state
  const [workflowType, setWorkflowType] = useState<'tps' | 'ai_recipe' | 'vibe'>('tps');
  const [aiRecipe, setAiRecipe] = useState<ColorRecipe | null>(null);
  const [aiSessionId, setAiSessionId] = useState<string | null>(null);
  const [refinementInput, setRefinementInput] = useState('');
  const [isRefiningAi, setIsRefiningAi] = useState(false);

  // Vibe Replicator workflow state
  const [vibeAnalysis, setVibeAnalysis] = useState<VibeAnalysis | null>(null);
  const [vibeInstructions, setVibeInstructions] = useState<GradingInstructions | null>(null);
  const [vibeCritiqueHistory, setVibeCritiqueHistory] = useState<CritiqueResult[]>([]);
  const [vibeSessionId, setVibeSessionId] = useState<string | null>(null);

  // Create preview URL for source image when selected
  useEffect(() => {
    if (srcFile) {
      const url = URL.createObjectURL(srcFile);
      setSrcPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setSrcPreviewUrl(null);
    }
  }, [srcFile]);

  // Fetch graded preview when session is created
  useEffect(() => {
    if (session?.session_id) {
      axios.get(`http://localhost:8000/api/preview/${session.session_id}`)
        .then(res => {
          setGradedPreviewUrl(res.data.preview);
        })
        .catch(err => {
          console.error('Failed to fetch preview:', err);
        });
    }
  }, [session]);

  const handleAnalyze = async () => {
    if (!refFile || !srcFile) return;
    setIsLoading(true);
    setError(null);
    setGradedPreviewUrl(null);
    setRefinementHistory([]);
    setIsRefined(false);

    const formData = new FormData();
    formData.append('reference', refFile);
    formData.append('source', srcFile);

    try {
      const res = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSession(res.data);
      setViewMode('compare');
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.message || "Analysis failed. Ensure Backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefine = async () => {
    if (!session) return;
    setIsRefining(true);
    setError(null);

    try {
      const res = await axios.post<RefineResponse>(
        `http://localhost:8000/api/refine/${session.session_id}?max_rounds=4`
      );

      setRefinementHistory(res.data.history);
      setGradedPreviewUrl(res.data.preview);
      setIsRefined(true);
      setViewMode('compare');
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.detail || "AI refinement failed.");
    } finally {
      setIsRefining(false);
    }
  };

  const handleDownload = async () => {
    if (!session && !aiSessionId && !vibeSessionId) return;
    let sessionToUse: string | undefined;
    let endpoint: string;

    if (workflowType === 'vibe' && vibeSessionId) {
      sessionToUse = vibeSessionId;
      endpoint = 'download-vibe';
    } else if (workflowType === 'ai_recipe' && aiSessionId) {
      sessionToUse = aiSessionId;
      endpoint = 'download-ai';
    } else {
      sessionToUse = session?.session_id;
      endpoint = 'download';
    }

    if (sessionToUse) {
      window.location.href = `http://localhost:8000/api/${endpoint}/${sessionToUse}`;
    }
  };

  // AI Recipe Workflow handlers
  const handleAnalyzeAI = async () => {
    if (!refFile || !srcFile) return;
    setIsLoading(true);
    setError(null);
    setGradedPreviewUrl(null);
    setAiRecipe(null);
    setWorkflowType('ai_recipe');

    const formData = new FormData();
    formData.append('reference', refFile);
    formData.append('source', srcFile);

    try {
      const res = await axios.post('http://localhost:8000/api/analyze-ai', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setAiSessionId(res.data.session_id);
      setAiRecipe(res.data.recipe);

      // Fetch preview
      const previewRes = await axios.get(`http://localhost:8000/api/preview-ai/${res.data.session_id}`);
      setGradedPreviewUrl(previewRes.data.preview);
      setViewMode('compare');
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.message || "AI Recipe generation failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefineAI = async () => {
    if (!aiSessionId || !refinementInput.trim()) return;
    setIsRefiningAi(true);
    setError(null);

    try {
      const res = await axios.post(
        `http://localhost:8000/api/refine-ai/${aiSessionId}?feedback=${encodeURIComponent(refinementInput)}`
      );
      setAiRecipe(res.data.recipe);
      setGradedPreviewUrl(res.data.preview);
      setRefinementInput('');
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.detail || "AI refinement failed.");
    } finally {
      setIsRefiningAi(false);
    }
  };

  // Vibe Replicator Workflow handler
  const handleAnalyzeVibe = async () => {
    if (!refFile || !srcFile) return;
    setIsLoading(true);
    setError(null);
    setGradedPreviewUrl(null);
    setVibeAnalysis(null);
    setVibeInstructions(null);
    setVibeCritiqueHistory([]);
    setWorkflowType('vibe');

    const formData = new FormData();
    formData.append('reference', refFile);
    formData.append('source', srcFile);

    try {
      const res = await axios.post('http://localhost:8000/api/analyze-vibe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setVibeSessionId(res.data.session_id);
      setVibeAnalysis(res.data.vibe_analysis);
      setVibeInstructions(res.data.final_instructions);
      setVibeCritiqueHistory(res.data.critique_history);

      // Fetch preview
      const previewRes = await axios.get(`http://localhost:8000/api/preview-vibe/${res.data.session_id}`);
      setGradedPreviewUrl(previewRes.data.preview);
      setViewMode('compare');
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.message || "Vibe Replicator failed.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full relative overflow-hidden">
      {/* Noise Overlay */}
      <div className="noise" />

      {/* Content */}
      <div className="relative z-10 p-8 max-w-7xl mx-auto flex flex-col items-center gap-12">

        {/* Header */}
        <header className="w-full border-b-4 border-black pb-4 flex justify-between items-end">
          <h1 className="font-display text-6xl uppercase tracking-tighter">
            Lattice<span className="text-accent">.AI</span>
          </h1>
          <div className="text-right font-mono text-xs">
            <div>Hybrid Math-AI Engine</div>
            <div>v1.1.0 // BUILD_A23</div>
          </div>
        </header>

        {/* Trinity Nodes */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full items-center">
          <UploadNode label="REFERENCE (THE LOOK)" onFileSelect={setRefFile} isLoading={isLoading} />

          <div className="flex flex-col items-center justify-center h-full">
            <motion.button
              disabled={!refFile || !srcFile || isLoading}
              onClick={handleAnalyze}
              whileHover={(!refFile || !srcFile || isLoading) ? {} : { scale: 1.05 }}
              whileTap={(!refFile || !srcFile || isLoading) ? {} : { scale: 0.95 }}
              className={`w-28 h-28 rounded-full border-4 border-black flex flex-col items-center justify-center shadow-neo transition-all bg-white
                            ${(!refFile || !srcFile) ? 'opacity-50 cursor-not-allowed' : ''}
                            ${isLoading && workflowType === 'tps' ? 'animate-pulse bg-accent' : ''}`}
            >
              <Cpu className={`w-10 h-10 mb-1 ${isLoading && workflowType === 'tps' ? 'animate-spin' : ''}`} />
              <span className="font-display font-bold text-xs text-center">
                {isLoading && workflowType === 'tps' ? "EXTRACTING..." : "EXTRACT DNA"}
              </span>
            </motion.button>

            <motion.button
              disabled={!refFile || !srcFile || isLoading}
              onClick={handleAnalyzeAI}
              whileHover={(!refFile || !srcFile || isLoading) ? {} : { scale: 1.05 }}
              whileTap={(!refFile || !srcFile || isLoading) ? {} : { scale: 0.95 }}
              className={`w-28 h-28 rounded-full border-4 border-black flex flex-col items-center justify-center shadow-neo transition-all bg-gradient-to-br from-purple-100 to-blue-100
                            ${(!refFile || !srcFile) ? 'opacity-50 cursor-not-allowed' : ''}
                            ${isLoading && workflowType === 'ai_recipe' ? 'animate-pulse bg-purple-300' : ''}`}
            >
              <Beaker className={`w-10 h-10 mb-1 ${isLoading && workflowType === 'ai_recipe' ? 'animate-spin' : ''}`} />
              <span className="font-display font-bold text-xs text-center">
                {isLoading && workflowType === 'ai_recipe' ? "GENERATING..." : "AI RECIPE"}
              </span>
            </motion.button>

            <motion.button
              disabled={!refFile || !srcFile || isLoading}
              onClick={handleAnalyzeVibe}
              whileHover={(!refFile || !srcFile || isLoading) ? {} : { scale: 1.05 }}
              whileTap={(!refFile || !srcFile || isLoading) ? {} : { scale: 0.95 }}
              className={`w-28 h-28 rounded-full border-4 border-black flex flex-col items-center justify-center shadow-neo transition-all bg-gradient-to-br from-amber-100 to-orange-100
                            ${(!refFile || !srcFile) ? 'opacity-50 cursor-not-allowed' : ''}
                            ${isLoading && workflowType === 'vibe' ? 'animate-pulse bg-amber-300' : ''}`}
            >
              <Wand2 className={`w-10 h-10 mb-1 ${isLoading && workflowType === 'vibe' ? 'animate-spin' : ''}`} />
              <span className="font-display font-bold text-xs text-center">
                {isLoading && workflowType === 'vibe' ? "ANALYZING..." : "VIBE"}
              </span>
            </motion.button>
          </div>

          <UploadNode label="SOURCE (YOUR LOG)" onFileSelect={setSrcFile} isLoading={isLoading} />
        </div>

        {/* Error Display */}
        {error && (
          <div className="w-full bg-red-100 border-2 border-red-500 p-4 font-mono text-red-600 flex items-center gap-2">
            <AlertCircle />
            {error}
          </div>
        )}

        {/* Visualizer Area */}
        {(session || aiRecipe || vibeAnalysis) && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full flex flex-col gap-4"
          >
            {/* View Mode Toggle */}
            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('compare')}
                className={`px-4 py-2 font-mono text-sm border-2 border-black transition-all flex items-center gap-2
                  ${viewMode === 'compare' ? 'bg-accent shadow-brutal' : 'bg-white hover:bg-gray-100'}`}
              >
                <Eye className="w-4 h-4" />
                COMPARE
              </button>
              <button
                onClick={() => setViewMode('lattice')}
                className={`px-4 py-2 font-mono text-sm border-2 border-black transition-all flex items-center gap-2
                  ${viewMode === 'lattice' ? 'bg-accent shadow-brutal' : 'bg-white hover:bg-gray-100'}`}
              >
                <Cpu className="w-4 h-4" />
                3D LATTICE
              </button>
              {isRefined && (
                <button
                  onClick={() => setViewMode('result')}
                  className={`px-4 py-2 font-mono text-sm border-2 border-black transition-all flex items-center gap-2
                    ${viewMode === 'result' ? 'bg-purple-200 shadow-brutal' : 'bg-white hover:bg-gray-100'}`}
                >
                  <Sparkles className="w-4 h-4" />
                  AI RESULT
                </button>
              )}
            </div>

            {/* View Container */}
            <div className="flex gap-4 h-[500px]">
              {/* Main View */}
              <div className="flex-grow h-full">
                {viewMode === 'compare' && srcPreviewUrl && gradedPreviewUrl ? (
                  <ComparisonSlider
                    beforeImage={srcPreviewUrl}
                    afterImage={gradedPreviewUrl}
                    beforeLabel="ORIGINAL"
                    afterLabel={isRefined ? "AI REFINED" : workflowType === 'vibe' ? "VIBE GRADED" : "GRADED"}
                  />
                ) : viewMode === 'result' && gradedPreviewUrl ? (
                  <div className="w-full h-full border-4 border-black shadow-neo overflow-hidden bg-black">
                    <img
                      src={gradedPreviewUrl}
                      alt="AI Refined Result"
                      className="w-full h-full object-contain"
                    />
                  </div>
                ) : viewMode === 'compare' && !gradedPreviewUrl ? (
                  <div className="w-full h-full bg-gray-200 border-4 border-black flex items-center justify-center font-mono">
                    <div className="animate-pulse">Generating Preview...</div>
                  </div>
                ) : session ? (
                  <div className="shadow-neo h-full">
                    <LatticeVisualizer
                      originalPoints={session.original_points}
                      warpedPoints={session.warped_points}
                    />
                  </div>
                ) : (
                  <div className="w-full h-full bg-gray-200 border-4 border-black flex items-center justify-center font-mono">
                    <div>AI Recipe Mode - No lattice visualization</div>
                  </div>
                )}
              </div>

              {/* Controls / Impulse */}
              <div className="w-72 bg-white border-4 border-black shadow-neo p-4 flex flex-col gap-4 overflow-y-auto">
                <h3 className="font-display text-xl uppercase border-b-2 border-black pb-2">Control Deck</h3>

                <div className="font-mono text-xs space-y-1 text-gray-500">
                  {workflowType === 'tps' && session ? (
                    <>
                      <p>PINS: {session.pins_source.length}</p>
                      <p>TPS: 0.01</p>
                      <p>STATUS: {isRefined ? '✓ AI REFINED' : 'INITIAL'}</p>
                    </>
                  ) : workflowType === 'vibe' && vibeAnalysis ? (
                    <>
                      <p>MODE: VIBE REPLICATOR</p>
                      <p>ITERATIONS: {vibeCritiqueHistory.length}</p>
                      <p>SCORE: {vibeCritiqueHistory.length > 0 ? `${vibeCritiqueHistory[vibeCritiqueHistory.length - 1].vibe_match_score}/10` : 'N/A'}</p>
                    </>
                  ) : (
                    <>
                      <p>MODE: AI RECIPE</p>
                      <p>STATUS: {aiRecipe ? '✓ GENERATED' : 'PENDING'}</p>
                    </>
                  )}
                </div>

                {/* AI Recipe Display */}
                {workflowType === 'ai_recipe' && aiRecipe && (
                  <div className="bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-300 p-3 space-y-2 rounded">
                    <div className="font-display text-xs uppercase text-blue-700 flex items-center gap-1 border-b border-blue-200 pb-2">
                      <Beaker className="w-3 h-3" />
                      Color DNA Recipe
                    </div>
                    <p className="text-xs text-gray-600 italic">{aiRecipe.description}</p>
                    <div className="grid grid-cols-2 gap-1 text-xs font-mono">
                      <div className="text-green-600">a-shift: {aiRecipe.a_channel_shift?.toFixed(1)}</div>
                      <div className="text-blue-600">b-shift: {aiRecipe.b_channel_shift?.toFixed(1)}</div>
                      <div>exposure: {aiRecipe.exposure_mult?.toFixed(2)}</div>
                      <div>sat: {aiRecipe.saturation_mult?.toFixed(2)}</div>
                    </div>
                    <div className="text-xs font-mono space-y-0.5 border-t border-blue-200 pt-2 mt-2">
                      <div className="text-gray-500">Zone Tints:</div>
                      <div>Shadow: a={aiRecipe.shadow_a?.toFixed(1)}, b={aiRecipe.shadow_b?.toFixed(1)}</div>
                      <div>Mid: a={aiRecipe.midtone_a?.toFixed(1)}, b={aiRecipe.midtone_b?.toFixed(1)}</div>
                      <div>High: a={aiRecipe.highlight_a?.toFixed(1)}, b={aiRecipe.highlight_b?.toFixed(1)}</div>
                    </div>
                  </div>
                )}

                {/* AI Recipe Refinement Input */}
                {workflowType === 'ai_recipe' && aiSessionId && (
                  <div className="space-y-2">
                    <input
                      type="text"
                      placeholder="e.g. 'More green' or 'Warmer highlights'"
                      value={refinementInput}
                      onChange={(e) => setRefinementInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleRefineAI()}
                      className="w-full border-2 border-black px-3 py-2 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-purple-400"
                    />
                    <button
                      onClick={handleRefineAI}
                      disabled={isRefiningAi || !refinementInput.trim()}
                      className={`w-full py-2 font-display text-sm uppercase border-2 border-black transition-all flex items-center justify-center gap-2
                        ${isRefiningAi ? 'bg-purple-200 animate-pulse' : 'bg-purple-100 hover:bg-purple-200 hover:shadow-brutal'}
                        ${!refinementInput.trim() ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <Send className={`w-4 h-4 ${isRefiningAi ? 'animate-spin' : ''}`} />
                      {isRefiningAi ? 'REFINING...' : 'REFINE RECIPE'}
                    </button>
                  </div>
                )}

                {/* Vibe Replicator Display */}
                {workflowType === 'vibe' && vibeAnalysis && (
                  <div className="bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-300 p-3 space-y-2 rounded">
                    <div className="font-display text-xs uppercase text-amber-700 flex items-center gap-1 border-b border-amber-200 pb-2">
                      <Wand2 className="w-3 h-3" />
                      Vibe Analysis
                    </div>
                    <p className="text-xs text-gray-700 font-medium">"{vibeAnalysis.vibe_description}"</p>
                    <div className="flex flex-wrap gap-1">
                      {vibeAnalysis.mood_keywords.map((kw, i) => (
                        <span key={i} className="px-2 py-0.5 bg-amber-200 text-amber-800 text-xs rounded-full">{kw}</span>
                      ))}
                    </div>
                    <p className="text-xs text-gray-600 italic">{vibeAnalysis.color_grading_style}</p>

                    {/* Grading Instructions */}
                    {vibeInstructions && (
                      <div className="border-t border-amber-200 pt-2 mt-2">
                        <div className="text-xs font-mono space-y-0.5">
                          <div className="text-gray-500 mb-1">Instructions:</div>
                          <div>Contrast: {vibeInstructions.contrast_curve} @ {vibeInstructions.contrast_strength?.toFixed(1)}</div>
                          <div>Shadow: a={vibeInstructions.shadow_a?.toFixed(0)}, b={vibeInstructions.shadow_b?.toFixed(0)}</div>
                          <div>Mid: a={vibeInstructions.midtone_a?.toFixed(0)}, b={vibeInstructions.midtone_b?.toFixed(0)}</div>
                          <div>Sat: {vibeInstructions.saturation_mult?.toFixed(2)}</div>
                        </div>
                      </div>
                    )}

                    {/* Critique History */}
                    {vibeCritiqueHistory.length > 0 && (
                      <div className="border-t border-amber-200 pt-2 mt-2">
                        <div className="text-xs text-gray-500 mb-1">Critique Rounds:</div>
                        {vibeCritiqueHistory.map((c, i) => (
                          <div key={i} className="text-xs font-mono flex items-center gap-1 py-0.5">
                            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold ${c.satisfied ? 'bg-green-500' : 'bg-amber-500'}`}>
                              {c.vibe_match_score}
                            </span>
                            <span className="text-gray-600 truncate">{c.feedback}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* AI Colorist Log */}
                {session?.operations_log && session.operations_log.length > 0 && (
                  <div className="bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-300 p-3 space-y-2 rounded">
                    <div className="font-display text-xs uppercase text-purple-700 flex items-center gap-1 border-b border-purple-200 pb-2">
                      <Zap className="w-3 h-3" />
                      AI Colorist Log
                    </div>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {session.operations_log.map((entry, idx) => (
                        <div key={idx} className="text-xs font-mono border-l-2 pl-2" style={{
                          borderColor: entry.stage === 'normalization' ? '#f59e0b' :
                            entry.stage === 'tone' ? '#3b82f6' :
                              entry.stage === 'palette' ? '#8b5cf6' :
                                entry.stage === 'selective' ? '#10b981' : '#6b7280'
                        }}>
                          <div className="flex items-center gap-1 text-gray-500 uppercase" style={{ fontSize: '10px' }}>
                            {entry.stage === 'normalization' && <Sliders className="w-3 h-3 text-amber-500" />}
                            {entry.stage === 'tone' && <Sliders className="w-3 h-3 text-blue-500" />}
                            {entry.stage === 'palette' && <Palette className="w-3 h-3 text-purple-500" />}
                            {entry.stage === 'selective' && <Target className="w-3 h-3 text-green-500" />}
                            {entry.stage}
                          </div>
                          <p className="text-gray-700 mt-0.5">{entry.action}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* AI Refine Button */}
                <button
                  onClick={handleRefine}
                  disabled={isRefining}
                  className={`w-full py-3 font-display text-sm uppercase border-2 border-black transition-all flex items-center justify-center gap-2
                    ${isRefining ? 'bg-purple-200 animate-pulse' : 'bg-purple-100 hover:bg-purple-200 hover:shadow-brutal'}`}
                >
                  <Sparkles className={`w-4 h-4 ${isRefining ? 'animate-spin' : ''}`} />
                  {isRefining ? 'REFINING...' : 'REFINE WITH AI'}
                </button>

                {/* AI Feedback Display */}
                <AnimatePresence>
                  {refinementHistory.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="bg-gray-50 border-2 border-gray-300 p-3 space-y-2"
                    >
                      <div className="font-display text-xs uppercase text-gray-700 flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        AI Feedback ({refinementHistory.length} rounds)
                      </div>
                      {refinementHistory.map((round, idx) => (
                        <div key={idx} className="text-xs font-mono border-l-2 border-purple-400 pl-2">
                          <div className="flex items-center gap-1 text-purple-600">
                            {round.satisfied ? <CheckCircle className="w-3 h-3" /> : null}
                            Round {round.round}
                          </div>
                          <p className="text-gray-600 mt-1">{round.feedback}</p>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="flex-grow" />

                <button
                  onClick={handleDownload}
                  className="w-full bg-black text-white py-4 font-display text-xl uppercase hover:bg-accent hover:text-black transition-colors flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  .CUBE
                </button>
              </div>
            </div>
          </motion.div>
        )}


      </div>
    </div>
  );
}

export default App;

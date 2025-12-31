
import { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadNode } from './components/UploadNode';
import { LatticeVisualizer } from './components/LatticeVisualizer';
import { ComparisonSlider } from './components/ComparisonSlider';
import { Download, AlertCircle, Cpu, Eye, Sparkles, CheckCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types

interface AnalysisResponse {
  session_id: string;
  original_points: number[][];
  warped_points: number[][];
  pins_source: number[][];
  pins_target: number[][];
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
    if (!session) return;
    window.location.href = `http://localhost:8000/api/download/${session.session_id}`;
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
              className={`w-32 h-32 rounded-full border-4 border-black flex flex-col items-center justify-center shadow-neo transition-all bg-white
                            ${(!refFile || !srcFile) ? 'opacity-50 cursor-not-allowed' : ''}
                            ${isLoading ? 'animate-pulse bg-accent' : ''}`}
            >
              <Cpu className={`w-12 h-12 mb-2 ${isLoading ? 'animate-spin' : ''}`} />
              <span className="font-display font-bold text-sm">
                {isLoading ? "EXTRACTING..." : "EXTRACT DNA"}
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
        {session && (
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
                    afterLabel={isRefined ? "AI REFINED" : "GRADED"}
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
                ) : (
                  <div className="shadow-neo h-full">
                    <LatticeVisualizer
                      originalPoints={session.original_points}
                      warpedPoints={session.warped_points}
                    />
                  </div>
                )}
              </div>

              {/* Controls / Impulse */}
              <div className="w-72 bg-white border-4 border-black shadow-neo p-4 flex flex-col gap-4 overflow-y-auto">
                <h3 className="font-display text-xl uppercase border-b-2 border-black pb-2">Control Deck</h3>

                <div className="font-mono text-xs space-y-1 text-gray-500">
                  <p>PINS: {session.pins_source.length}</p>
                  <p>TPS: 0.01</p>
                  <p>STATUS: {isRefined ? '✓ AI REFINED' : 'INITIAL'}</p>
                </div>

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

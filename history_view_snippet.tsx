                ) : viewMode === 'result' && (srcPreviewUrl || gradedPreviewUrl) ? (
    <div className="w-full h-full border-4 border-black shadow-neo overflow-y-auto bg-gray-50 p-4">
        <h3 className="font-display text-2xl uppercase mb-4">Pipeline History</h3>
        <div className="grid grid-cols-1 gap-4">
            {/* Original */}
            {srcPreviewUrl && (
                <div className="border-2 border-black shadow-brutal bg-white">
                    <div className="bg-gray-200 border-b-2 border-black px-3 py-2 font-mono text-sm font-bold">
                        1. ORIGINAL SOURCE
                    </div>
                    <div className="p-2">
                        <img src={srcPreviewUrl} alt="Original Source" className="w-full h-auto" />
                    </div>
                </div>
            )}

            {/* V1 Baseline or AI Refined */}
            {gradedPreviewUrl && (
                <div className="border-2 border-black shadow-brutal bg-white">
                    <div className={`border-b-2 border-black px-3 py-2 font-mono text-sm font-bold flex items-center gap-2 ${isRefined ? 'bg-purple-200' : 'bg-yellow-200'
                        }`}>
                        {isRefined && <Sparkles className="w-4 h-4" />}
                        {isRefined ? '2. AI REFINED RESULT' : '2. V1 BASELINE GRADE'}
                    </div>
                    <div className="p-2">
                        <img src={gradedPreviewUrl} alt={isRefined ? "AI Refined" : "V1 Baseline"} className="w-full h-auto" />
                    </div>
                    <div className="bg-gray-100 border-t-2 border-black px-3 py-1 font-mono text-xs">
                        {isRefined ? 'Optimized by Gemini 3 Pro Critic' : 'Luma Histogram + TPS Chroma Warp'}
                    </div>
                </div>
            )}
        </div>
    </div>

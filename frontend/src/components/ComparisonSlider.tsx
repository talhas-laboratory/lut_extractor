import React, { useState, useRef, useCallback } from 'react';

interface ComparisonSliderProps {
    beforeImage: string;  // URL or base64 data URI
    afterImage: string;   // URL or base64 data URI
    beforeLabel?: string;
    afterLabel?: string;
}

export const ComparisonSlider: React.FC<ComparisonSliderProps> = ({
    beforeImage,
    afterImage,
    beforeLabel = 'BEFORE',
    afterLabel = 'AFTER'
}) => {
    const [sliderPosition, setSliderPosition] = useState(50);
    const containerRef = useRef<HTMLDivElement>(null);
    const isDragging = useRef(false);

    const handleMove = useCallback((clientX: number) => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
        setSliderPosition(percentage);
    }, []);

    const handleMouseDown = () => {
        isDragging.current = true;
    };

    const handleMouseUp = () => {
        isDragging.current = false;
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging.current) {
            handleMove(e.clientX);
        }
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        if (e.touches.length === 1) {
            handleMove(e.touches[0].clientX);
        }
    };

    const handleClick = (e: React.MouseEvent) => {
        handleMove(e.clientX);
    };

    return (
        <div
            ref={containerRef}
            className="relative w-full h-full overflow-hidden cursor-col-resize border-4 border-black shadow-brutal select-none"
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
            onClick={handleClick}
        >
            {/* After Image (Full) */}
            <img
                src={afterImage}
                alt="After"
                className="absolute inset-0 w-full h-full object-cover"
                draggable={false}
            />

            {/* Before Image (Clipped) */}
            <div
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${sliderPosition}%` }}
            >
                <img
                    src={beforeImage}
                    alt="Before"
                    className="absolute inset-0 w-full h-full object-cover"
                    style={{
                        width: containerRef.current ? `${containerRef.current.offsetWidth}px` : '100%',
                        maxWidth: 'none'
                    }}
                    draggable={false}
                />
            </div>

            {/* Slider Line */}
            <div
                className="absolute top-0 bottom-0 w-1 bg-white shadow-lg"
                style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
            >
                {/* Slider Handle */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 bg-white border-4 border-black rounded-full flex items-center justify-center shadow-brutal">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="3">
                        <path d="M18 8L22 12L18 16" />
                        <path d="M6 8L2 12L6 16" />
                    </svg>
                </div>
            </div>

            {/* Labels */}
            <div className="absolute top-3 left-3 bg-black text-white px-2 py-1 font-mono text-xs font-bold">
                {beforeLabel}
            </div>
            <div className="absolute top-3 right-3 bg-accent text-black px-2 py-1 font-mono text-xs font-bold">
                {afterLabel}
            </div>
        </div>
    );
};

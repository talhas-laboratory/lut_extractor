
import React, { useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Check, File as FileIcon } from 'lucide-react';
import clsx from 'clsx';

interface UploadNodeProps {
    label: string;
    onFileSelect: (file: File) => void;
    isLoading?: boolean;
}

export const UploadNode: React.FC<UploadNodeProps> = ({ label, onFileSelect, isLoading }) => {
    const [file, setFile] = useState<File | null>(null);
    const [isHovering, setIsHovering] = useState(false);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsHovering(false);
        if (e.dataTransfer.files?.[0]) {
            const f = e.dataTransfer.files[0];
            setFile(f);
            onFileSelect(f);
        }
    }, [onFileSelect]);

    return (
        <div className="flex flex-col items-center">
            <motion.div
                whileHover={{ x: -2, y: -2, boxShadow: "8px 8px 0px 0px #000" }}
                whileTap={{ x: 2, y: 2, boxShadow: "2px 2px 0px 0px #000" }}
                onDragOver={(e) => { e.preventDefault(); setIsHovering(true); }}
                onDragLeave={() => setIsHovering(false)}
                onDrop={handleDrop}
                className={clsx(
                    "relative w-48 h-48 border-4 border-black bg-white flex flex-col items-center justify-center cursor-pointer transition-colors overflow-hidden",
                    "shadow-neo",
                    isHovering ? "bg-gray-100" : ""
                )}
            >
                <input
                    type="file"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    onChange={(e) => {
                        if (e.target.files?.[0]) {
                            setFile(e.target.files[0]);
                            onFileSelect(e.target.files[0]);
                        }
                    }}
                />

                <AnimatePresence mode='wait'>
                    {file ? (
                        <motion.div
                            key="file"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center p-2 text-center"
                        >
                            <Check className="w-12 h-12 text-black mb-2" strokeWidth={3} />
                            <span className="font-mono text-xs break-all px-2 bg-black text-white p-1">
                                {file.name.slice(0, 15)}{file.name.length > 15 && "..."}
                            </span>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="empty"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center"
                        >
                            <Upload className="w-12 h-12 mb-2 stroke-1" />
                            <span className="font-display uppercase tracking-widest text-lg">{label}</span>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Power Line Animation if active */}
                {file && !isLoading && (
                    <motion.div
                        layoutId={`line-${label}`}
                        className="absolute bottom-0 left-1/2 w-1 bg-black h-0"
                        animate={{ height: "100%" }} // This logic needs external layout awareness for the "Power Line" connecting to engine
                    // For now, just a visual indicator bar
                    />
                )}
            </motion.div>

            <div className="mt-4 font-mono text-xs uppercase opacity-50 tracking-widest">[NODE: {label}]</div>
        </div>
    );
};

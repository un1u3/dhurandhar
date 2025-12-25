"use client";

import { useState, useRef } from "react";
import { Upload, File, Loader2, CheckCircle2, AlertCircle, Scan } from "lucide-react";

interface UploadZoneProps {
    onFileSelect: (file: File) => void;
    status: "idle" | "uploading" | "analyzing" | "success" | "error";
    error?: string;
}

export default function UploadZone({ onFileSelect, status, error }: UploadZoneProps) {
    const [isDragOver, setIsDragOver] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = () => {
        setIsDragOver(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
            onFileSelect(file);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            onFileSelect(file);
        }
    };

    const isInteractive = status === "idle" || status === "error";

    return (
        <div className="relative group w-full max-w-3xl mx-auto fade-in delay-100">
            {/* Glow Effect Background */}
            <div
                className={`absolute -inset-1 bg-gradient-to-r from-white/10 to-white/5 rounded-3xl blur-xl opacity-0 transition duration-1000 group-hover:opacity-100
        ${isDragOver ? "opacity-100 duration-200" : ""}`}
            />

            <div
                className={`
          relative w-full aspect-[2/1] min-h-[300px] rounded-2xl overflow-hidden
          glass-panel transition-all duration-300 ease-out
          flex flex-col items-center justify-center p-12 text-center
          ${isDragOver ? "border-white/40 scale-[1.01] bg-white/5 shadow-2xl" : "hover:border-white/20"}
          ${!isInteractive && "cursor-not-allowed opacity-90"}
          ${isInteractive ? "cursor-pointer" : ""}
        `}
                onDragOver={isInteractive ? handleDragOver : undefined}
                onDragLeave={isInteractive ? handleDragLeave : undefined}
                onDrop={isInteractive ? handleDrop : undefined}
                onClick={() => isInteractive && fileInputRef.current?.click()}
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="image/jpeg, image/png"
                />

                {/* Animated Grid on Hover/Active */}
                <div className={`absolute inset-0 bg-[url('/grid.svg')] opacity-[0.03] transition-opacity duration-500 ${isDragOver ? "opacity-15" : "group-hover:opacity-10"}`} />

                <div className="relative z-10 flex flex-col items-center gap-8">
                    {status === "idle" && (
                        <>
                            <div className={`p-6 rounded-full bg-white/5 border border-white/10 transition-transform duration-300 ${isDragOver ? "scale-110" : "group-hover:scale-105"}`}>
                                <Upload className="w-12 h-12 text-white/90" strokeWidth={1} />
                            </div>
                            <div className="space-y-3">
                                <p className="text-3xl font-light tracking-tight text-white">Upload Chest X-ray</p>
                                <p className="text-base text-muted uppercase tracking-widest text-[10px]">Drag & drop or click to select</p>
                            </div>
                            <div className="flex gap-3 mt-2 justify-center">
                                <span className="px-2 py-1 rounded text-[10px] uppercase font-medium bg-white/5 border border-white/5 text-muted">JPG</span>
                                <span className="px-2 py-1 rounded text-[10px] uppercase font-medium bg-white/5 border border-white/5 text-muted">PNG</span>
                            </div>
                        </>
                    )}

                    {status === "uploading" && (
                        <>
                            <div className="relative">
                                <div className="absolute inset-0 animate-ping rounded-full bg-white/10 opacity-75"></div>
                                <div className="relative p-4 rounded-full bg-white/5 border border-white/10">
                                    <Loader2 className="w-8 h-8 animate-spin text-white" strokeWidth={1} />
                                </div>
                            </div>
                            <div className="space-y-1">
                                <p className="text-sm font-medium tracking-wide">Uploading secure connection...</p>
                            </div>
                        </>
                    )}

                    {status === "analyzing" && (
                        <>
                            <div className="relative p-4 rounded-full bg-white/5 border border-white/10">
                                <Scan className="w-8 h-8 animate-pulse text-white" strokeWidth={1} />
                            </div>
                            <div className="space-y-2">
                                <p className="text-sm font-medium uppercase tracking-widest animate-pulse">Running Diagnosis Model</p>
                                <div className="w-48 h-1 bg-white/10 rounded-full overflow-hidden">
                                    <div className="h-full bg-white/80 animate-[loading_1s_ease-in-out_infinite] w-1/3 rounded-full"></div>
                                </div>
                            </div>
                        </>
                    )}

                    {status === "success" && (
                        <>
                            <div className="p-4 rounded-full bg-green-500/10 border border-green-500/20">
                                <CheckCircle2 className="w-10 h-10 text-green-400" strokeWidth={1.5} />
                            </div>
                            <div className="space-y-1">
                                <p className="text-sm font-medium text-green-400">Analysis Complete</p>
                            </div>
                        </>
                    )}

                    {status === "error" && (
                        <>
                            <div className="p-4 rounded-full bg-red-500/10 border border-red-500/20">
                                <AlertCircle className="w-10 h-10 text-red-400" strokeWidth={1.5} />
                            </div>
                            <div className="space-y-1">
                                <p className="text-sm font-medium text-red-400">Analysis Failed</p>
                                <p className="text-xs text-muted max-w-[200px] mx-auto">{error}</p>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

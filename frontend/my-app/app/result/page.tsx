"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { RefreshCw, ArrowLeft, Camera, Activity, Info } from "lucide-react";

// Mock data for the results
const MOCK_DISEASES = [
    { name: "Pneumonia", probability: 0.84, severity: "High" },
    { name: "Tb", probability: 0.12, severity: "Low" },
    { name: "Pneumothorax", probability: 0.05, severity: "Very Low" },
    { name: "Atelectasis", probability: 0.28, severity: "Moderate" },
    { name: "Pleural Effusion", probability: 0.42, severity: "Moderate" },
];

export default function Result() {
    const router = useRouter();
    const [analysisResult, setAnalysisResult] = useState<any>(null);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);

    useEffect(() => {
        const result = sessionStorage.getItem("analysisResult");
        const storedImage = sessionStorage.getItem("uploadedImage");

        if (storedImage) {
            setUploadedImage(storedImage);
        }

        if (result) {
            setAnalysisResult(JSON.parse(result));
        } else {
            // For demo purposes, we'll use mock data if nothing is in session
            setAnalysisResult({
                prediction: "Pneumonia",
                confidence: 0.84,
                diseases: MOCK_DISEASES
            });
        }
    }, [router]);

    if (!analysisResult) return null;

    return (
        <main className="h-screen w-full flex flex-col relative overflow-hidden bg-black text-white selection:bg-white selection:text-black">

            {/* Ambient Background - Monochrome only */}
            <div className="absolute top-0 left-0 w-full h-full bg-[url('/grid.svg')] opacity-[0.03] pointer-events-none"></div>
            <div className="absolute -top-[20%] -right-[20%] w-[60%] h-[60%] bg-white/[0.03] blur-[150px] rounded-full pointer-events-none"></div>
            <div className="absolute -bottom-[20%] -left-[20%] w-[60%] h-[60%] bg-white/[0.02] blur-[150px] rounded-full pointer-events-none"></div>

            {/* Header Navigation */}
            <div className="absolute top-0 left-0 w-full z-50 p-12 flex justify-between items-start pointer-events-none">
                <button
                    onClick={() => router.push("/upload")}
                    className="group pointer-events-auto flex items-center gap-3 text-white hover:text-white transition-all px-8 py-4 rounded-full bg-white/5 hover:bg-white/10 backdrop-blur-md border border-white/10 shadow-lg"
                >
                    <ArrowLeft className="w-5 h-5 transition-transform group-hover:-translate-x-1" />
                    <span className="text-xs font-bold tracking-[0.2em] uppercase">Return</span>
                </button>

                <button
                    onClick={() => router.push("/upload")}
                    className="pointer-events-auto flex items-center gap-3 text-white hover:text-white transition-all px-8 py-4 rounded-full bg-white/5 hover:bg-white/10 backdrop-blur-md border border-white/10 shadow-lg"
                >
                    <RefreshCw className="w-4 h-4" />
                    <span className="text-xs font-bold tracking-[0.2em] uppercase">New Analysis</span>
                </button>
            </div>

            {/* Page Header Title - Centered across entire page */}
            <div className="absolute top-12 left-0 w-full z-50 pointer-events-none flex justify-center">
                <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-white/40 uppercase text-center drop-shadow-2xl">
                    Your Results.
                </h1>
            </div>

            <div className="flex-1 w-full h-full flex flex-row">

                {/* LEFT: Massive Image Grid Area */}
                <div className="w-[60%] h-full relative flex flex-col p-8 z-10">
                    <div className="w-full h-full flex flex-col justify-center relative">
                        {/* Images Container - Centered Vertically with whitespace */}
                        <div className="w-full max-h-[60%] flex flex-col justify-center relative z-10">
                            <div className="grid grid-cols-2 gap-8 w-full px-12">
                                {/* Original X-ray */}
                                <div className="group relative aspect-square rounded-[2rem] overflow-hidden border border-white/10 bg-white/[0.02] shadow-2xl">
                                    <div className="absolute top-6 left-6 z-20">
                                        <span className="px-4 py-2 rounded-full bg-black/40 backdrop-blur-xl border border-white/10 text-[10px] uppercase tracking-widest font-bold text-white flex items-center gap-2 shadow-lg">
                                            <Camera className="w-3.5 h-3.5" />
                                            Original Input
                                        </span>
                                    </div>
                                    {/* Image Display */}
                                    <div className="w-full h-full flex items-center justify-center">
                                        {uploadedImage ? (
                                            <img src={uploadedImage} alt="Uploaded X-Ray" className="w-full h-full object-cover" />
                                        ) : (
                                            <img src="/grid.svg" alt="X-Ray" className="w-full h-full object-contain opacity-20" />
                                        )}
                                    </div>
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60"></div>
                                </div>

                                {/* Heatmap */}
                                <div className="group relative aspect-square rounded-[2rem] overflow-hidden border border-white/10 bg-white/[0.02] shadow-2xl">
                                    <div className="absolute top-6 left-6 z-20">
                                        <span className="px-4 py-2 rounded-full bg-white text-black text-[10px] uppercase tracking-widest font-bold flex items-center gap-2 shadow-lg">
                                            <Activity className="w-3.5 h-3.5" />
                                            CAM Analysis
                                        </span>
                                    </div>
                                    {/* Heatmap Placeholder - Monochrome Pulse */}
                                    <div className="w-full h-full flex items-center justify-center relative">
                                        <div className="absolute w-[60%] h-[60%] bg-white/10 rounded-full blur-[80px] animate-pulse"></div>
                                        <div className="absolute w-[40%] h-[40%] bg-white/5 rounded-full blur-[60px] animate-pulse delay-75"></div>
                                    </div>
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* RIGHT: Diagnosis Results - Monochrome & Vertically Centered */}
                <div className="w-[40%] h-full border-l border-white/5 bg-white/[0.01] backdrop-blur-sm flex flex-col justify-center relative z-20">
                    <div className="w-full max-w-xl mx-auto flex flex-col p-12">

                        {/* Primary Finding Block */}
                        <div className="mb-20">
                            <div className="flex items-center gap-4 mb-8">
                                <span className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_20px_rgba(255,255,255,0.5)] animate-pulse"></span>
                                <h2 className="text-xs uppercase tracking-[0.4em] font-bold text-white/50">Primary Diagnosis</h2>
                            </div>

                            <div className="space-y-6">
                                <h3 className="text-8xl font-bold text-white tracking-tighter leading-none">
                                    {analysisResult.prediction}
                                </h3>

                                <div className="flex items-center gap-8">
                                    <div className="flex flex-col">
                                        <span className="text-6xl font-light tracking-tight text-white">
                                            {(analysisResult.confidence * 100).toFixed(0)}<span className="text-3xl text-white/40 ml-1">%</span>
                                        </span>
                                        <span className="text-[10px] uppercase tracking-widest text-white/30 mt-2 font-bold">Confidence Score</span>
                                    </div>

                                    <div className="h-0.5 flex-1 bg-white/10 relative rounded-full overflow-hidden">
                                        <div className="absolute top-0 left-0 h-full bg-white transition-all duration-1000 ease-out" style={{ width: `${analysisResult.confidence * 100}%` }}></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Detailed Analysis List */}
                        <div className="mb-12">
                            <h2 className="text-xs uppercase tracking-[0.4em] text-white/30 font-bold mb-8 pl-1">Detailed Findings</h2>

                            <div className="space-y-4">
                                {analysisResult.diseases ? analysisResult.diseases.map((disease: any, index: number) => (
                                    <div key={disease.name} className="group flex items-center justify-between p-4 rounded-xl hover:bg-white/[0.03] transition-colors -mx-4 cursor-default">
                                        <div className="flex items-center gap-4">
                                            <div className="w-1.5 h-1.5 rounded-full bg-white/10 group-hover:bg-white/50 transition-colors"></div>
                                            <span className="text-lg text-white/60 group-hover:text-white transition-colors font-light">
                                                {disease.name}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <div className="w-24 h-1 bg-white/5 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full transition-all duration-1000 ease-out ${disease.probability > 0.5 ? 'bg-white/80' : 'bg-white/20'}`}
                                                    style={{ width: `${disease.probability * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="font-mono text-lg text-white/30 group-hover:text-white/80 transition-colors w-12 text-right">
                                                {(disease.probability * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                )) : MOCK_DISEASES.map((disease, index) => (
                                    <div key={disease.name} className="group flex items-center justify-between p-4 rounded-xl hover:bg-white/[0.03] transition-colors -mx-4 cursor-default">
                                        <div className="flex items-center gap-4">
                                            <div className="w-1.5 h-1.5 rounded-full bg-white/10 group-hover:bg-white/50 transition-colors"></div>
                                            <span className="text-lg text-white/60 group-hover:text-white transition-colors font-light">
                                                {disease.name}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <div className="w-24 h-1 bg-white/5 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full transition-all duration-1000 ease-out ${disease.probability > 0.5 ? 'bg-white/80' : 'bg-white/20'}`}
                                                    style={{ width: `${disease.probability * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="font-mono text-lg text-white/30 group-hover:text-white/80 transition-colors w-12 text-right">
                                                {(disease.probability * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Footer */}
                        <div className="flex items-center gap-4 opacity-40">
                            <Info className="w-4 h-4" />
                            <p className="text-[10px] uppercase tracking-widest font-medium">
                                AI Generated Analysis • Consult Medical Professional
                            </p>
                        </div>

                    </div>
                </div>

            </div>
        </main>
    );
}

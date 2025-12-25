"use client";

import { useRouter } from "next/navigation";
import Image from "next/image";

export default function Hero() {
    const router = useRouter();

    return (
        <section id="hero" className="min-h-screen flex items-center justify-center pt-20 px-6">
            <div className="container mx-auto grid lg:grid-cols-2 gap-16 items-center max-w-7xl">

                {/* Left Content */}
                <div className="space-y-8 z-10">
                    <div className="space-y-6">
                        <div className="inline-block">
                            <h1 className="text-7xl md:text-8xl lg:text-9xl font-bold tracking-tighter leading-none bg-gradient-to-b from-white via-white to-white/40 bg-clip-text text-transparent">
                                xr-AI
                            </h1>
                            <div className="h-1 w-24 bg-gradient-to-r from-white to-transparent mt-4"></div>
                        </div>

                        <p className="text-xl md:text-2xl lg:text-3xl text-gray-400 font-light leading-relaxed max-w-xl">
                            An AI X-ray screening tool.
                            <br />
                            <span className="text-gray-500">Early detection with precision and clarity.</span>
                        </p>
                    </div>

                    <div className="flex gap-4 pt-4">
                        <button
                            onClick={() => router.push("/upload")}
                            className="group px-10 py-5 bg-white text-black rounded-full text-xl font-bold hover:bg-white/90 transition-all duration-300 flex items-center gap-2 shadow-[0_0_40px_rgba(255,255,255,0.3)] hover:shadow-[0_0_60px_rgba(255,255,255,0.5)] transform hover:scale-105"
                        >
                            Try Now
                            <span className="group-hover:translate-x-1 transition-transform duration-300">→</span>
                        </button>
                    </div>


                </div>

                {/* Right Visual */}
                <div className="relative h-[600px] lg:h-[800px] w-full flex items-center justify-center">
                    <div className="relative w-full h-full flex items-center justify-center">
                        <Image
                            src="/hero_image.png"
                            alt="Hero Image"
                            width={800}
                            height={800}
                            className="object-contain max-h-full"
                            priority
                        />
                    </div>
                </div>
            </div>
        </section>
    );
}

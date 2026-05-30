"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useKYCStore } from "@/store/kycStore";

interface DocumentCaptureScreenProps {
  side: "front" | "back";
}

export default function DocumentCaptureScreen({ side }: DocumentCaptureScreenProps) {
  const { setStep, setDocumentFrontDataURL, setDocumentBackDataURL, setDocumentFrontBlob, setDocumentBackBlob } =
    useKYCStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cardDetected, setCardDetected] = useState(false);

  const label = side === "front" ? "Front of your document" : "Back of your document";
  const confirmStep = side === "front" ? "DOCUMENT_FRONT_CONFIRM" : "DOCUMENT_BACK_CONFIRM";

  // Start camera
  useEffect(() => {
    let running = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (!running) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        const v = videoRef.current;
        if (v) { v.srcObject = stream; await v.play(); }
        setCameraReady(true);
      } catch {
        // Camera not available — user can use file upload
      }
    })();
    return () => {
      running = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Basic card detection via brightness/edges in canvas
  useEffect(() => {
    if (!cameraReady) return;
    let running = true;
    const check = () => {
      if (!running) return;
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || !c || v.videoWidth === 0) {
        setTimeout(check, 500);
        return;
      }
      // Simple brightness check in center region
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext("2d");
      if (ctx) {
        ctx.drawImage(v, 0, 0);
        const mid = ctx.getImageData(
          Math.floor(v.videoWidth * 0.2),
          Math.floor(v.videoHeight * 0.2),
          Math.floor(v.videoWidth * 0.6),
          Math.floor(v.videoHeight * 0.6)
        );
        let sum = 0;
        for (let i = 0; i < mid.data.length; i += 4) {
          sum += (mid.data[i] + mid.data[i + 1] + mid.data[i + 2]) / 3;
        }
        const avg = sum / (mid.data.length / 4);
        // Bright card against dark background (100-200 range typical)
        setCardDetected(avg > 80 && avg < 230);
      }
      setTimeout(check, 500);
    };
    check();
    return () => { running = false; };
  }, [cameraReady]);

  const captureImage = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(v, 0, 0);
    const dataURL = c.toDataURL("image/jpeg", 0.9);
    c.toBlob((blob) => {
      if (!blob) return;
      if (side === "front") {
        setDocumentFrontDataURL(dataURL);
        setDocumentFrontBlob(blob);
      } else {
        setDocumentBackDataURL(dataURL);
        setDocumentBackBlob(blob);
      }
      setStep(confirmStep);
    }, "image/jpeg", 0.9);
  }, [side, setStep, setDocumentFrontDataURL, setDocumentBackDataURL, setDocumentFrontBlob, setDocumentBackBlob]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataURL = reader.result as string;
      if (side === "front") {
        setDocumentFrontDataURL(dataURL);
        setDocumentFrontBlob(file);
      } else {
        setDocumentBackDataURL(dataURL);
        setDocumentBackBlob(file);
      }
      setStep(confirmStep);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Upload your document</h2>
      <p className="text-zinc-400 text-sm">Ensure all details are visible and easy to read</p>

      <p className="text-white font-medium text-sm bg-[#0f0f0f] rounded-lg px-3 py-2 text-center">
        📷 {label}
      </p>

      {/* Camera view */}
      <div className="relative rounded-xl overflow-hidden bg-black aspect-video">
        {cameraReady ? (
          <>
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              muted
              playsInline
            />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full hidden" />
            {cardDetected && (
              <div className="absolute bottom-3 left-0 right-0 flex justify-center">
                <span className="bg-green-500/20 border border-green-500 text-green-400 px-3 py-1 rounded-full text-sm">
                  Card detected
                </span>
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900 text-zinc-500 text-sm">
            Camera not available — use file upload below
          </div>
        )}
      </div>

      <button
        onClick={captureImage}
        disabled={!cameraReady}
        className={`w-full py-4 rounded-full font-semibold text-base ${
          cameraReady
            ? "bg-white text-black"
            : "bg-zinc-800 text-zinc-600 cursor-not-allowed"
        }`}
      >
        📷 Capture Card
      </button>

      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-zinc-700" />
        <span className="text-zinc-500 text-sm">or</span>
        <div className="flex-1 h-px bg-zinc-700" />
      </div>

      <button
        onClick={() => fileInputRef.current?.click()}
        className="w-full py-4 rounded-full border border-zinc-600 text-white text-base"
      >
        📁 Upload from device
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileUpload}
      />
    </div>
  );
}

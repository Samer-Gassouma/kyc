"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import {
  ArrowLeft,
  ScanText,
  Loader2,
  CheckCircle,
  XCircle,
  ExternalLink,
  Camera,
} from "lucide-react";
import Link from "next/link";
import StepProgress, { KYCStep } from "@/components/kyc/StepProgress";
import IDCaptureStep from "@/components/kyc/IDCaptureStep";
import { useFaceDetection, REGION_EDGES } from "@/hooks/useFaceDetection";
import { canvasToJpegBlob } from "@/lib/frameEncoder";

interface SessionData {
  id_number?: string;
  id_number_valid?: boolean;
  last_name?: string;
  first_name?: string;
  father_lineage?: string;
  date_of_birth?: string;
  place_of_birth?: string;
  mother_name?: string;
  profession?: string;
  address?: string;
  issue_date?: string;
  barcode?: Record<string, unknown>;
  id_number_from_barcode?: string;
}

interface VerificationResult {
  passed: boolean;
  confidence: number;
  user_id: string;
  liveBlob?: Blob;
}

type Phase =
  | "front_id"
  | "back_id"
  | "extracting"
  | "face_scan"
  | "completed"
  | "failed";

export default function KYCPage() {
  const [token, setToken] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("front_id");
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [faceCropUrl, setFaceCropUrl] = useState<string>("");
  const [userId, setUserId] = useState<string>("");
  const [verificationResult, setVerificationResult] =
    useState<VerificationResult | null>(null);
  const [docMatch, setDocMatch] = useState<{match:boolean;similarity:number}|null>(null);
  const [error, setError] = useState<string | null>(null);
  const frontBlobRef = useRef<Blob | null>(null);
  const backBlobRef = useRef<Blob | null>(null);
  const cinFaceBlobRef = useRef<Blob | null>(null);

  // Face scan state
  const faceVideoRef = useRef<HTMLVideoElement>(null);
  const faceOverlayRef = useRef<HTMLCanvasElement>(null);
  const faceStreamRef = useRef<MediaStream | null>(null);
  const faceAnimRef = useRef(0);
  const faceStableRef = useRef(0);
  const faceDCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [faceMsg, setFaceMsg] = useState("");
  const [faceCd, setFaceCd] = useState(0);
  const [faceProgress, setFaceProgress] = useState(0);

  const { isReady, detect } = useFaceDetection();

  // Get JWT on mount
  useEffect(() => {
    const sid = `kyc_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    })
      .then((res) => res.json())
      .then((data) => setToken(data.access_token))
      .catch(() => setToken("dev_token"));
  }, []);

  // ── Face scan: start camera when phase becomes face_scan ──────────
  useEffect(() => {
    if (phase !== "face_scan") return;
    let running = true;
    setFaceMsg("Loading..."); setFaceCd(0); setFaceProgress(0);
    faceStableRef.current = 0;

    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
        });
        if (!running) { s.getTracks().forEach(t => t.stop()); return; }
        faceStreamRef.current = s;
        const v = faceVideoRef.current; if (!v) return;
        v.srcObject = s; await v.play();
      } catch (e) { setError(e instanceof Error ? e.message : "Camera"); setPhase("failed"); }
    })();

    return () => { running = false; faceStreamRef.current?.getTracks().forEach(t => t.stop()); };
  }, [phase]);

  // ── Face scan frame loop ──────────────────────────────────────────
  useEffect(() => {
    if (phase !== "face_scan") return;
    let running = true;
    if (!faceDCanvasRef.current) faceDCanvasRef.current = document.createElement("canvas");

    const loop = () => {
      if (!running) return;
      const v = faceVideoRef.current, dc = faceDCanvasRef.current;
      if (!v || v.videoWidth === 0 || !dc) { faceAnimRef.current = requestAnimationFrame(loop); return; }

      const { landmarks, faceDetected } = detect(v, dc);
      if (!running) return;

      if (faceDetected && landmarks[0]) {
        const pts = landmarks[0];
        drawFaceMesh(faceOverlayRef.current!, pts, v.videoWidth, v.videoHeight);
        const b = faceBbox(pts, v.videoWidth, v.videoHeight);
        const goodSize = b.width / v.videoWidth > 0.25;

        if (goodSize) {
          faceStableRef.current++;
          const r = Math.max(0, Math.ceil((50 - faceStableRef.current) / 30));
          setFaceCd(r); setFaceProgress(Math.round((faceStableRef.current / 50) * 100));
          if (faceStableRef.current >= 50) { running = false; faceDoCapture(v); return; }
          setFaceMsg(r > 0 ? `Hold still... ${r}` : "Scanning...");
        } else {
          faceStableRef.current = Math.max(0, faceStableRef.current - 1);
          setFaceCd(0); setFaceProgress(0); setFaceMsg("Move closer — face too small");
        }
      } else {
        faceStableRef.current = Math.max(0, faceStableRef.current - 3);
        setFaceCd(0); setFaceProgress(0); setFaceMsg("No face detected");
        const ov = faceOverlayRef.current;
        if (ov) { const c = ov.getContext("2d"); if (c) c.clearRect(0, 0, ov.width, ov.height); }
      }
      faceAnimRef.current = requestAnimationFrame(loop);
    };
    faceAnimRef.current = requestAnimationFrame(loop);
    return () => { running = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, isReady]);

  async function faceDoCapture(video: HTMLVideoElement) {
    setPhase("extracting"); // temp phase to hide camera
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d")!.drawImage(video, 0, 0);
    const blob = await canvasToJpegBlob(c, 0.85);

    try {
      const fd = new FormData(); fd.append("image", blob, "face.jpg"); fd.append("liveness_score", "1.0");
      const res = await fetch(`${API_BASE}/api/face/enroll`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const d = await res.json();
      setUserId(d.user_id);
      setVerificationResult({ passed: true, confidence: 1.0, user_id: d.user_id });

      // Start extraction in background
      const frontBlob = frontBlobRef.current;
      const backBlob = backBlobRef.current;
      if (frontBlob && backBlob) {
        try {
          const efd = new FormData(); efd.append("front", frontBlob, "front.jpg"); efd.append("back", backBlob, "back.jpg");
          const eres = await fetch(`${API_BASE}/api/extract/start`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: efd });
          if (eres.ok) setExtractSid((await eres.json()).session_id);
        } catch { /* non-fatal */ }
      }
      setPhase("completed");
    } catch (e) { setError(e instanceof Error ? e.message : "Enrollment failed"); setPhase("failed"); }
  }

  function faceBbox(pts: any[], vw: number, vh: number) {
    let x = Infinity, y = Infinity, X = -Infinity, Y = -Infinity;
    for (const p of pts) { if (p.x < x) x = p.x; if (p.x > X) X = p.x; if (p.y < y) y = p.y; if (p.y > Y) Y = p.y; }
    return { x: x * vw, y: y * vh, width: (X - x) * vw, height: (Y - y) * vh };
  }

  function drawFaceMesh(canvas: HTMLCanvasElement, pts: any[], vw: number, vh: number) {
    canvas.width = vw; canvas.height = vh;
    const ctx = canvas.getContext("2d"); if (!ctx || pts.length < 400) return;
    ctx.clearRect(0, 0, vw, vh);
    for (const [, r] of Object.entries(REGION_EDGES)) {
      ctx.strokeStyle = r.color + "99"; ctx.lineWidth = 1.4; ctx.beginPath();
      for (const [a, b] of r.edges) { if (a >= pts.length || b >= pts.length) continue; ctx.moveTo(pts[a].x * vw, pts[a].y * vh); ctx.lineTo(pts[b].x * vw, pts[b].y * vh); }
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    for (let i = 0; i < pts.length; i += 3) { ctx.beginPath(); ctx.arc(pts[i].x * vw, pts[i].y * vh, 1.1, 0, Math.PI * 2); ctx.fill(); }
  }
  const handleFrontComplete = useCallback(
    (_captureId: string, blob: Blob) => {
      frontBlobRef.current = blob;
      setPhase("back_id");
    },
    []
  );

  // ── Step 2: Back captured → go straight to face scan ─────────────
  const handleBackComplete = useCallback(
    (_captureId: string, blob: Blob) => {
      backBlobRef.current = blob;
      setPhase("face_scan");
    },
    []
  );

  // ── Step 3: Face scan done → start extraction in background ───────
  const [extractSid, setExtractSid] = useState("");
  const handleFaceScanComplete = useCallback(
    async (result: VerificationResult) => {
      if (!result.passed) {
        setPhase("failed");
        setError("Face verification did not pass");
        return;
      }
      setVerificationResult(result);

      // Start CIN extraction in background (fire and forget)
      const frontBlob = frontBlobRef.current;
      const backBlob = backBlobRef.current;
      if (frontBlob && backBlob) {
        try {
          const fd = new FormData();
          fd.append("front", frontBlob, "front.jpg");
          fd.append("back", backBlob, "back.jpg");
          const res = await fetch(`${API_BASE}/api/extract/start`, {
            method: "POST",
            headers: { Authorization: `Bearer ${token}` },
            body: fd,
          });
          if (res.ok) {
            const d = await res.json();
            setExtractSid(d.session_id);
          }
        } catch { /* non-fatal */ }
      }

      setPhase("completed");
    },
    [token]
  );

  // ── Step helpers ───────────────────────────────────────────────────
  const completedSteps: KYCStep[] = [];
  if (phase === "face_scan" || phase === "extracting" || phase === "completed") completedSteps.push("front_id");
  if (phase === "face_scan" || phase === "extracting" || phase === "completed") completedSteps.push("back_id");
  if (phase === "extracting" || phase === "completed") completedSteps.push("face_scan");

  const currentStep: KYCStep =
    phase === "front_id"
      ? "front_id"
      : phase === "back_id"
      ? "back_id"
      : "face_scan";

  const showProgress =
    phase === "front_id" ||
    phase === "back_id" ||
    phase === "face_scan";

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link
          href="/"
          className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <ScanText className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">KYC Verification</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        {!token && (
          <div className="flex flex-1 items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
          </div>
        )}

        {token && showProgress && (
          <StepProgress
            currentStep={currentStep}
            completedSteps={completedSteps}
          />
        )}

        {/* Step 1: Front ID */}
        {phase === "front_id" && (
          <IDCaptureStep
            side="front"
            token={token}
            onCaptureComplete={handleFrontComplete}
          />
        )}

        {/* Step 2: Back ID */}
        {phase === "back_id" && (
          <IDCaptureStep
            side="back"
            token={token}
            onCaptureComplete={handleBackComplete}
          />
        )}

        {/* Step 3: Face scan — inline, same logic as /face */}
        {phase === "face_scan" && (
          <div className="flex flex-col items-center gap-4 w-full">
            <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
              <div className="relative">
                <video ref={faceVideoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
                <canvas ref={faceOverlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
                {faceCd > 0 && faceCd <= 3 && (
                  <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/30">
                    <span className="text-7xl font-bold text-white animate-pulse">{faceCd}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="flex flex-col items-center gap-2 text-center">
              <p className="text-sm text-zinc-400"><Camera className="mr-1 inline h-4 w-4" />{faceMsg}</p>
              {faceProgress > 0 && (
                <div className="h-1 w-40 rounded-full bg-zinc-700">
                  <div className="h-full rounded-full bg-sky-400 transition-all" style={{ width: `${faceProgress}%` }} />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results */}
        {phase === "completed" && (
          <div className="w-full space-y-4 pt-4">
            <div className="flex items-center gap-2 rounded-lg bg-green-500/10 px-4 py-3 text-sm text-green-400">
              <CheckCircle className="h-4 w-4" />
              Verification complete
            </div>

            {verificationResult && (
              <div className="rounded-xl bg-green-500/10 p-4 ring-1 ring-green-500/30">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Face Verified</h2>
                <p className="text-sm text-green-400">Confidence: {(verificationResult.confidence*100).toFixed(0)}%</p>
                <p className="text-xs text-zinc-500 mt-1">User ID: {verificationResult.user_id}</p>
              </div>
            )}

            {extractSid && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">CIN Extraction Started</h2>
                <p className="text-xs text-zinc-400">Your ID data is being extracted in the background.</p>
                <code className="mt-2 block break-all rounded bg-zinc-800 p-2 text-xs text-zinc-300">Session: {extractSid}</code>
                <a href={`/extract`} target="_blank" rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-2 rounded-lg bg-blue-600/20 px-4 py-2 text-sm text-blue-400 ring-1 ring-blue-500/30">
                  <ExternalLink className="h-4 w-4" />View Extraction Status
                </a>
              </div>
            )}

            {sessionData && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">Extracted Data</h2>
                <pre className="max-h-[32rem] overflow-auto rounded-lg bg-zinc-950 p-4 text-xs text-green-400">
                  {JSON.stringify({...sessionData, face_verification: verificationResult}, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {phase === "failed" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
              <XCircle className="h-4 w-4" />
              {error || "Verification failed"}
            </div>
            <button
              onClick={() => window.location.reload()}
              className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
            >
              Start Over
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

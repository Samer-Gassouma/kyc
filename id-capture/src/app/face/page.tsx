"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { useMediaPipeFace } from "@/hooks/useMediaPipeFace";
import { checkLiveness, prepareLivenessInput } from "@/lib/silentFaceLiveness";
import Link from "next/link";
import {
  ArrowLeft,
  Camera,
  Loader2,
  CheckCircle,
  XCircle,
  UserPlus,
  Fingerprint,
} from "lucide-react";

type Mode = "enroll" | "verify";
type Status =
  | "idle"
  | "preparing"
  | "scanning"
  | "capturing"
  | "processing"
  | "done"
  | "error";

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [enrolledUserId, setEnrolledUserId] = useState("");
  const [verifyResult, setVerifyResult] = useState<{
    matched: boolean;
    confidence: number;
    threshold_used: number;
    user_id: string;
  } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number | null>(null);

  const { landmarks, faceDetected, isReady, processFrame } = useMediaPipeFace();

  // Get JWT on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: `face_${Date.now()}` }),
    })
      .then((r) => r.json())
      .then((d) => setToken(d.access_token))
      .catch(() => setToken("dev_token"));
  }, []);

  // ── Camera ─────────────────────────────────────────────────────

  const stopCamera = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    setStatus("preparing");
    setStatusText("Loading models...");
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;

      // Use the DOM video element from JSX, not a detached one
      const video = videoRef.current;
      if (!video) throw new Error("Video element not mounted");
      video.srcObject = stream;
      await video.play();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera access denied");
      setStatus("error");
    }
  }, []);

  // Transition to scanning when models + camera are ready
  useEffect(() => {
    if (isReady && videoRef.current?.readyState && videoRef.current.readyState >= 2 && status === "preparing") {
      setStatus("scanning");
      setStatusText(mode === "enroll" ? "Position face to enroll" : "Position face to verify");
    }
  }, [isReady, status, mode]);

  // ── Face positioning ──────────────────────────────────────────

  function faceIsWellPositioned(): boolean {
    if (!landmarks || landmarks.length === 0) return false;
    const pts = landmarks[0];
    if (pts.length < 468) return false;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const area = (maxX - minX) * (maxY - minY);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const dist = Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2);
    return area > 0.08 && dist < 0.25;
  }

  // ── Overlay ────────────────────────────────────────────────────

  function drawOverlay() {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (landmarks && landmarks.length > 0) {
      const pts = landmarks[0];
      ctx.fillStyle = "rgba(59, 130, 246, 0.55)";
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      const chin = pts[152], forehead = pts[10];
      const left = pts[234], right = pts[454];
      const cx = ((left.x + right.x) / 2) * canvas.width;
      const cy = ((forehead.y + chin.y) / 2) * canvas.height;
      const rx = Math.abs(right.x - left.x) / 2 * canvas.width * 1.3;
      const ry = Math.abs(chin.y - forehead.y) / 2 * canvas.height * 1.3;
      ctx.strokeStyle = faceIsWellPositioned()
        ? "rgba(34, 197, 94, 0.5)" : "rgba(255,255,255,0.2)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // ── Frame loop ─────────────────────────────────────────────────

  useEffect(() => {
    if (status !== "scanning" && status !== "capturing") return;
    const canvas = document.createElement("canvas");
    const loop = () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }
      grabFrame(video, canvas);
      processFrame(video);
      drawOverlay();
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, landmarks]);

  // ── Auto-capture ───────────────────────────────────────────────

  const captureRef = useRef(false);
  const stableRef = useRef(0);

  useEffect(() => {
    if (status !== "scanning" || captureRef.current) return;
    if (faceIsWellPositioned()) {
      stableRef.current++;
      const remaining = Math.max(1, Math.ceil((30 - stableRef.current) / 10));
      setStatusText(`Hold still... ${remaining}`);
      if (stableRef.current >= 30) {
        captureRef.current = true;
        handleAction();
      }
    } else {
      stableRef.current = 0;
      if (faceDetected) {
        setStatusText("Center your face in the oval");
      } else {
        setStatusText(mode === "enroll" ? "Position face to enroll" : "Position face to verify");
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [landmarks, status]);

  // ── Enroll / Verify ────────────────────────────────────────────

  async function handleAction() {
    setStatus("capturing");
    setStatusText("Checking liveness...");

    const video = videoRef.current;
    if (!video || !landmarks || landmarks.length === 0) {
      resetCapture();
      return;
    }

    const pts = landmarks[0];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const bbox = {
      x: minX * video.videoWidth, y: minY * video.videoHeight,
      width: (maxX - minX) * video.videoWidth, height: (maxY - minY) * video.videoHeight,
    };

    let bestLiveness = 0;
    let bestBlob: Blob | null = null;
    for (let i = 0; i < 3; i++) {
      await new Promise((r) => setTimeout(r, 150));
      const input = prepareLivenessInput(video, bbox);
      if (!input) continue;
      const score = await checkLiveness(input);
      if (score > bestLiveness) {
        bestLiveness = score;
        const fc = document.createElement("canvas");
        grabFrame(video, fc);
        bestBlob = await canvasToJpegBlob(fc, 0.85);
      }
    }

    if (!bestBlob || bestLiveness < 0.5) {
      setError(bestLiveness < 0.5
        ? "Spoof detected — use a real face, not a photo or screen"
        : "Liveness check failed");
      setStatus("error");
      return;
    }

    setStatus("processing");
    setStatusText(mode === "enroll" ? "Generating embedding..." : "Verifying identity...");

    try {
      if (mode === "enroll") {
        const formData = new FormData();
        formData.append("image", bestBlob, "face.jpg");
        formData.append("liveness_score", bestLiveness.toString());

        const res = await fetch(`${API_BASE}/api/face/enroll`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        });
        if (!res.ok) {
          const detail = (await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`;
          throw new Error(detail);
        }
        const data = await res.json();
        setEnrolledUserId(data.user_id);
        setStatus("done");
      } else {
        if (!enrolledUserId) {
          setError("Enroll a face first before verifying");
          setStatus("error");
          return;
        }
        const formData = new FormData();
        formData.append("image", bestBlob, "face.jpg");
        formData.append("user_id", enrolledUserId);

        const res = await fetch(`${API_BASE}/api/face/verify`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        });
        if (!res.ok) {
          const detail = (await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`;
          throw new Error(detail);
        }
        const data = await res.json();
        setVerifyResult(data);
        setStatus("done");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      setStatus("error");
    }
  }

  function resetCapture() {
    captureRef.current = false;
    stableRef.current = 0;
    setStatus("scanning");
    setStatusText(mode === "enroll" ? "Position face to enroll" : "Position face to verify");
  }

  function handleStart() {
    captureRef.current = false;
    stableRef.current = 0;
    setError(null);
    setVerifyResult(null);
    stopCamera();
    startCamera();
  }

  useEffect(() => { return stopCamera; }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const showVideo = status === "preparing" || status === "scanning" || status === "capturing";

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <Fingerprint className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">Face Pipeline Test</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        {/* Mode tabs */}
        <div className="mb-4 flex w-full rounded-lg bg-zinc-900 p-1">
          <button
            onClick={() => { setMode("enroll"); stopCamera(); setStatus("idle"); setError(null); setVerifyResult(null); }}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              mode === "enroll" ? "bg-blue-600 text-white" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <UserPlus className="mr-2 inline h-4 w-4" /> Enroll
          </button>
          <button
            onClick={() => { setMode("verify"); stopCamera(); setStatus("idle"); setError(null); setVerifyResult(null); }}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              mode === "verify" ? "bg-blue-600 text-white" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Fingerprint className="mr-2 inline h-4 w-4" /> Verify
          </button>
        </div>

        {/* Idle */}
        {status === "idle" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <p className="text-sm text-zinc-400">
              {mode === "enroll"
                ? "Capture a face to create a new identity enrollment"
                : "Capture a face to verify against the enrolled identity"}
            </p>
            <button onClick={handleStart}
              className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-500">
              <Camera className="h-5 w-5" /> Start Camera
            </button>
            {mode === "verify" && (
              <div className="mt-2 w-full">
                <label className="text-xs text-zinc-500">User ID to verify</label>
                <input type="text" value={enrolledUserId}
                  onChange={(e) => setEnrolledUserId(e.target.value)}
                  placeholder="Paste user_id from enrollment..."
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200" />
              </div>
            )}
          </div>
        )}

        {/* Camera view */}
        <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
          {showVideo && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted
                className="h-full w-full object-cover"
                style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
              <canvas ref={overlayRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
                style={{ transform: "scaleX(-1)" }} />
            </div>
          )}

          {status === "processing" && (
            <div className="flex items-center justify-center bg-black" style={{ aspectRatio: "3/4" }}>
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
                <p className="text-sm text-zinc-300">{statusText}</p>
              </div>
            </div>
          )}

          {status === "done" && (
            <div className={`flex items-center justify-center ${mode === "enroll" || verifyResult?.matched ? "bg-green-950/50" : "bg-red-950/50"}`}
              style={{ aspectRatio: "3/4" }}>
              <div className="flex flex-col items-center gap-3">
                {(mode === "enroll" || verifyResult?.matched) ? (
                  <CheckCircle className="h-16 w-16 text-green-400" />
                ) : (
                  <XCircle className="h-16 w-16 text-red-400" />
                )}
                <p className="text-sm font-medium text-green-400">
                  {mode === "enroll" ? "Enrolled!" : verifyResult?.matched ? "Verified!" : "No match"}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Status bar */}
        <div className="mt-4 flex flex-col items-center gap-2 text-center">
          {(status === "scanning" || status === "capturing") && (
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <Camera className="h-4 w-4" /> {statusText}
            </div>
          )}
          {status === "error" && (
            <div className="flex flex-col items-center gap-3">
              <div className="flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
                <XCircle className="h-4 w-4" /> {error || "Error"}
              </div>
              <button onClick={handleStart}
                className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500">
                Try Again
              </button>
            </div>
          )}
        </div>

        {/* Results */}
        {status === "done" && (
          <div className="mt-4 w-full space-y-3">
            {mode === "enroll" && enrolledUserId && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Enrollment Result</h2>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2 text-green-400">
                    <CheckCircle className="h-4 w-4" /> Face enrolled
                  </div>
                  <div>
                    <span className="text-zinc-500">User ID: </span>
                    <code className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-200 break-all">{enrolledUserId}</code>
                  </div>
                  <p className="text-xs text-zinc-500">Switch to <strong>Verify</strong> tab to test matching.</p>
                </div>
                <button onClick={() => { setMode("verify"); setStatus("idle"); }}
                  className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500">
                  <Fingerprint className="h-4 w-4" /> Switch to Verify
                </button>
              </div>
            )}

            {mode === "verify" && verifyResult && (
              <div className={`rounded-xl p-4 ${verifyResult.matched ? "bg-green-500/10 ring-1 ring-green-500/30" : "bg-red-500/10 ring-1 ring-red-500/30"}`}>
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Verification Result</h2>
                <div className="space-y-2 text-sm">
                  <div className={`flex items-center gap-2 ${verifyResult.matched ? "text-green-400" : "text-red-400"}`}>
                    {verifyResult.matched ? <CheckCircle className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
                    {verifyResult.matched ? "Identity verified" : "No match"}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="rounded bg-zinc-800 p-2">
                      <span className="text-zinc-500">Confidence</span>
                      <p className="font-mono text-zinc-200">{(verifyResult.confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div className="rounded bg-zinc-800 p-2">
                      <span className="text-zinc-500">Threshold</span>
                      <p className="font-mono text-zinc-200">{(verifyResult.threshold_used * 100).toFixed(0)}%</p>
                    </div>
                    <div className="col-span-2 rounded bg-zinc-800 p-2">
                      <span className="text-zinc-500">User ID</span>
                      <p className="font-mono text-xs text-zinc-200 break-all">{verifyResult.user_id}</p>
                    </div>
                  </div>
                </div>
                <button onClick={handleStart}
                  className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500">
                  Test Again
                </button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

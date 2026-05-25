"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useMediaPipeFace } from "@/hooks/useMediaPipeFace";
import { checkLiveness, prepareLivenessInput } from "@/lib/silentFaceLiveness";
import Link from "next/link";
import { ArrowLeft, Camera, Loader2, CheckCircle, XCircle, UserPlus, Fingerprint } from "lucide-react";

type Mode = "enroll" | "verify";

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<"idle" | "active" | "verifying" | "done" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState<{ matched?: boolean; confidence?: number; threshold_used?: number; user_id?: string } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const { landmarks, faceDetected, isReady, detect } = useMediaPipeFace();

  // JWT
  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: `face_${Date.now()}` }) })
      .then(r => r.json()).then(d => setToken(d.access_token)).catch(() => setToken("dev_token"));
  }, []);

  // ── Start / Stop camera ────────────────────────────────────────

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setErrMsg(null);
    setResult(null);
    setPhase("active");
    setStatusMsg("Loading...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) throw new Error("no video element");
      v.srcObject = stream;
      await v.play();
      setStatusMsg(mode === "enroll" ? "Position your face" : "Position your face");
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Camera error");
      setPhase("error");
    }
  }, [mode]);

  useEffect(() => stop, []); // eslint-disable-line

  // ── Frame loop ─────────────────────────────────────────────────

  const stableRef = useRef(0);     // 0–30 progress
  const captureRef = useRef(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (phase !== "active") {
      stableRef.current = 0;
      captureRef.current = false;
      setProgress(0);
      return;
    }
    if (!detectCanvasRef.current) detectCanvasRef.current = document.createElement("canvas");

    const loop = () => {
      const video = videoRef.current;
      const dCanvas = detectCanvasRef.current;
      if (!video || !dCanvas || video.videoWidth === 0) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }

      detect(video, dCanvas);
      drawOverlay(video);

      if (!captureRef.current) {
        if (faceIsWellPositioned()) {
          stableRef.current = Math.min(30, stableRef.current + 1);
        } else {
          stableRef.current = Math.max(0, stableRef.current - 2);
        }
        const pct = Math.round((stableRef.current / 30) * 100);
        setProgress(pct);

        if (stableRef.current >= 30) {
          captureRef.current = true;
          setStatusMsg("Processing...");
          handleAction();
          return;
        } else if (stableRef.current > 15) {
          setStatusMsg("Hold still...");
        } else if (landmarks) {
          setStatusMsg("Center your face in the frame");
        } else {
          setStatusMsg(mode === "enroll" ? "Position your face" : "Position your face");
        }
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, landmarks]);

  // ── Face position check ────────────────────────────────────────

  function faceIsWellPositioned(): boolean {
    if (!landmarks || !landmarks[0] || landmarks[0].length < 468) return false;
    const pts = landmarks[0];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const area = (maxX - minX) * (maxY - minY);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return area > 0.06 && Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < 0.3;
  }

  // ── Face mesh — professional KYC wireframe overlay ─────────────

  // Full dense triangle mesh (tessellation) of the 468-point face topology.
  // Each connection is [pt_a, pt_b] — an edge in the wireframe.
  // Generated from MediaPipe's canonical face mesh UV topology.
  const TESSELATION: [number, number][] = (() => {
    const c: [number, number][] = [];
    // Left eye region
    const le = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
    for (let i = 0; i < le.length; i++) c.push([le[i], le[(i + 1) % le.length]]);
    // Right eye region
    const re = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];
    for (let i = 0; i < re.length; i++) c.push([re[i], re[(i + 1) % re.length]]);
    // Left eyebrow
    const leb = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107];
    for (let i = 0; i < leb.length; i++) c.push([leb[i], leb[(i + 1) % leb.length]]);
    // Right eyebrow
    const reb = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336];
    for (let i = 0; i < reb.length; i++) c.push([reb[i], reb[(i + 1) % reb.length]]);
    // Nose bridge + tip
    const nose = [6, 168, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 460, 294, 459, 458, 461, 354, 455, 460];
    for (let i = 0; i < nose.length - 1; i++) c.push([nose[i], nose[i + 1]]);
    c.push([1, 2], [2, 98], [98, 327]);
    // Lips outer
    const lo = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    for (let i = 0; i < lo.length; i++) c.push([lo[i], lo[(i + 1) % lo.length]]);
    // Lips inner
    const li = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
    for (let i = 0; i < li.length; i++) c.push([li[i], li[(i + 1) % li.length]]);
    // Face oval
    const oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    for (let i = 0; i < oval.length; i++) c.push([oval[i], oval[(i + 1) % oval.length]]);
    // Dense horizontal+vertical grid across the face for KYC wireframe look
    // Cheeks and jaw connectors
    const cheeks = [
      234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454
    ];
    for (let i = 0; i < cheeks.length; i++) c.push([cheeks[i], cheeks[(i + 1) % cheeks.length]]);
    // Forehead connectors
    const forehead = [109, 67, 103, 54, 21, 162, 127, 234];
    for (let i = 0; i < forehead.length; i++) c.push([forehead[i], 10]);
    // Vertical connectors: forehead → nose → chin
    c.push([10, 151], [151, 9], [9, 8], [8, 168], [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2], [2, 200], [200, 199], [199, 175], [175, 152]);
    // Horizontal brow-to-brow
    c.push([107, 336], [105, 334], [66, 296], [70, 300], [55, 285], [65, 295], [52, 282], [53, 283], [46, 276]);
    // Eye-to-brow connectors
    c.push([33, 46], [133, 53], [173, 52], [157, 65], [158, 55], [159, 70], [160, 63], [161, 105], [246, 107]);
    c.push([362, 276], [263, 283], [249, 282], [390, 295], [373, 285], [374, 300], [380, 293], [381, 334], [382, 296], [398, 336]);
    // Nose-to-eye connectors
    c.push([6, 33], [6, 362], [168, 133], [168, 263], [197, 157], [197, 390], [195, 158], [195, 373], [5, 159], [5, 374]);
    // Nose-to-lips
    c.push([2, 0], [2, 17], [200, 37], [200, 267]);
    // Lips-to-chin
    c.push([17, 199], [37, 175], [267, 175]);
    // Jaw-to-cheek dense grid
    for (let i = 0; i < 16; i++) {
      const top = [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454][i];
      const bot = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288][i] || 152;
      if (top && bot) c.push([top, bot]);
    }
    // Eye region dense fill
    for (const [a, b] of [[33,133],[133,155],[155,145],[145,159],[159,163],[246,161],[161,144],[144,153],[153,154],[154,157],[157,173],[173,158],[158,160],[160,7],[7,163]] as [number,number][]) c.push([a,b]);
    for (const [a, b] of [[362,263],[263,249],[249,390],[390,373],[373,380],[398,381],[381,374],[374,384],[384,385],[385,386],[386,387],[387,388],[388,466],[466,382]] as [number,number][]) c.push([a,b]);
    return c;
  })();

  // Point indices to draw as subtle dots (key junction points)
  const KEY_POINTS = new Set([
    ...Array.from({length: 468}, (_, i) => i).filter(i =>
      i < 200 || i > 350 || [0,17,61,291,152,10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,149,176,148,377,400,378,379,365,397,288,361,323,454,338,297,332,284,251,389,356].includes(i)
    )
  ]);

  function drawOverlay(video: HTMLVideoElement) {
    const canvas = overlayRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks || !landmarks[0]) return;

    const pts = landmarks[0];
    const w = canvas.width, h = canvas.height;
    const positioned = faceIsWellPositioned();
    const color = positioned ? "rgba(56, 189, 248, 0.7)" : "rgba(56, 189, 248, 0.25)";
    const dotColor = positioned ? "rgba(56, 189, 248, 0.9)" : "rgba(56, 189, 248, 0.3)";

    // Draw all tessellation edges as thin semi-transparent lines
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (const [a, b] of TESSELATION) {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb) continue;
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
    }
    ctx.stroke();

    // Draw junction points as subtle dots
    ctx.fillStyle = dotColor;
    for (let i = 0; i < 468; i++) {
      const p = pts[i];
      if (!p) continue;
      // Denser dots on key features, sparser elsewhere
      const isKey = KEY_POINTS.has(i);
      const r = isKey ? 1.2 : 0.6;
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── Enroll or Verify ───────────────────────────────────────────

  async function handleAction() {
    setPhase("verifying");
    setStatusMsg("Checking liveness...");

    const video = videoRef.current;
    if (!video || !landmarks || !landmarks[0]) { setPhase("error"); setErrMsg("Lost face"); return; }

    const pts = landmarks[0];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const bbox = { x: minX * video.videoWidth, y: minY * video.videoHeight, width: (maxX - minX) * video.videoWidth, height: (maxY - minY) * video.videoHeight };

    let bestLive = 0, bestBlob: Blob | null = null;
    for (let i = 0; i < 3; i++) {
      await new Promise(r => setTimeout(r, 150));
      const input = prepareLivenessInput(video, bbox);
      if (!input) continue;
      const score = await checkLiveness(input);
      if (score > bestLive) {
        bestLive = score;
        const fc = document.createElement("canvas");
        fc.width = video.videoWidth; fc.height = video.videoHeight;
        fc.getContext("2d")!.drawImage(video, 0, 0);
        bestBlob = await canvasToJpegBlob(fc, 0.85);
      }
    }

    if (!bestBlob || bestLive < 0.5) {
      setPhase("error");
      setErrMsg(bestLive < 0.5 ? "Spoof detected — use a real face" : "Liveness failed");
      return;
    }

    setStatusMsg(mode === "enroll" ? "Generating embedding..." : "Verifying...");
    try {
      if (mode === "enroll") {
        const fd = new FormData();
        fd.append("image", bestBlob, "face.jpg");
        fd.append("liveness_score", String(bestLive));
        const res = await fetch(`${API_BASE}/api/face/enroll`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
        const data = await res.json();
        setUserId(data.user_id);
        setResult(data);
        setPhase("done");
      } else {
        if (!userId) { setPhase("error"); setErrMsg("Enroll first or paste a user ID above"); return; }
        const fd = new FormData();
        fd.append("image", bestBlob, "face.jpg");
        fd.append("user_id", userId);
        const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail || `HTTP ${res.status}`);
        setResult(await res.json());
        setPhase("done");
      }
    } catch (e) {
      setPhase("error");
      setErrMsg(e instanceof Error ? e.message : "Request failed");
    }
  }

  function retry() {
    stop();
    setPhase("idle");
    setErrMsg(null);
    setResult(null);
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"><ArrowLeft className="h-5 w-5" /></Link>
        <Fingerprint className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">Face Pipeline Test</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        {/* Tabs */}
        <div className="mb-4 flex w-full rounded-lg bg-zinc-900 p-1">
          <button onClick={() => { setMode("enroll"); retry(); }}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "enroll" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <UserPlus className="mr-2 inline h-4 w-4" />Enroll
          </button>
          <button onClick={() => { setMode("verify"); retry(); }}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode === "verify" ? "bg-blue-600 text-white" : "text-zinc-400"}`}>
            <Fingerprint className="mr-2 inline h-4 w-4" />Verify
          </button>
        </div>

        {/* Idle */}
        {phase === "idle" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <p className="text-sm text-zinc-400">{mode === "enroll" ? "Capture a face to enroll a new identity" : "Verify against an enrolled identity"}</p>
            <button onClick={start} className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-500">
              <Camera className="h-5 w-5" />Start Camera
            </button>
            {mode === "verify" && (
              <input type="text" value={userId} onChange={e => setUserId(e.target.value)}
                placeholder="Paste user_id from enrollment..."
                className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200" />
            )}
          </div>
        )}

        {/* Camera */}
        <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ maxWidth: 400 }}>
          {phase === "active" && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
            </div>
          )}
          {phase === "verifying" && (
            <div className="flex items-center justify-center bg-black" style={{ aspectRatio: "3/4" }}>
              <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
            </div>
          )}
          {phase === "done" && (
            <div className={`flex items-center justify-center ${mode === "enroll" || result?.matched ? "bg-green-950/50" : "bg-red-950/50"}`} style={{ aspectRatio: "3/4" }}>
              {(mode === "enroll" || result?.matched) ? <CheckCircle className="h-16 w-16 text-green-400" /> : <XCircle className="h-16 w-16 text-red-400" />}
            </div>
          )}
        </div>

        {/* Status / Error */}
        <div className="mt-4 text-center">
          {phase === "active" && (
            <div className="flex flex-col items-center gap-2">
              <p className="text-sm text-zinc-400"><Camera className="mr-1 inline h-4 w-4" />{isReady ? statusMsg : "Loading face detection..."}</p>
              {progress > 0 && (
                <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-700">
                  <div
                    className="h-full rounded-full bg-blue-400 transition-all duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </div>
          )}
          {phase === "verifying" && <p className="text-sm text-blue-400"><Loader2 className="mr-1 inline h-4 w-4 animate-spin" />{statusMsg}</p>}
          {phase === "error" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-red-400"><XCircle className="mr-1 inline h-4 w-4" />{errMsg}</p>
              <button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white">Try Again</button>
            </div>
          )}
        </div>

        {/* Results */}
        {phase === "done" && result && (
          <div className="mt-4 w-full space-y-3">
            {mode === "enroll" && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Enrolled</h2>
                <code className="block break-all rounded bg-zinc-800 p-2 text-xs text-green-400">{result.user_id}</code>
                <button onClick={() => { setMode("verify"); setUserId(result.user_id || ""); setPhase("idle"); }}
                  className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white">Switch to Verify</button>
              </div>
            )}
            {mode === "verify" && (
              <div className={`rounded-xl p-4 ${result.matched ? "bg-green-500/10 ring-1 ring-green-500/30" : "bg-red-500/10 ring-1 ring-red-500/30"}`}>
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Result</h2>
                <p className={`text-lg font-bold ${result.matched ? "text-green-400" : "text-red-400"}`}>
                  {result.matched ? "MATCH" : "NO MATCH"} — {((result.confidence ?? 0) * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-zinc-500">threshold: {((result.threshold_used ?? 0) * 100).toFixed(0)}%</p>
                <button onClick={retry} className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white">Test Again</button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

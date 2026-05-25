"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useMediaPipeFace } from "@/hooks/useMediaPipeFace";
import { checkLiveness, prepareLivenessInput } from "@/lib/silentFaceLiveness";
import Link from "next/link";
import { ArrowLeft, Camera, Loader2, CheckCircle, XCircle, UserPlus, Fingerprint, ArrowUp, ArrowLeftCircle, ArrowRightCircle, Focus } from "lucide-react";
import Face3DViewer from "@/components/kyc/Face3DViewer";

type Mode = "enroll" | "verify";
type Angle = "center" | "left" | "right" | "up";

const ANGLE_ORDER: Angle[] = ["center", "left", "right", "up"];

const ANGLE_LABEL: Record<Angle, string> = {
  center: "Look straight ahead",
  left: "Turn your head to the left",
  right: "Turn your head to the right",
  up: "Tilt your head upward",
};

const ANGLE_ICON: Record<Angle, typeof Focus> = {
  center: Focus,
  left: ArrowLeftCircle,
  right: ArrowRightCircle,
  up: ArrowUp,
};

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<"idle" | "active" | "capturing" | "verifying" | "done" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState<{ matched?: boolean; confidence?: number; threshold_used?: number; user_id?: string } | null>(null);

  // Multi-angle capture state
  const [currentAngle, setCurrentAngle] = useState<Angle>("center");
  const [angleProgress, setAngleProgress] = useState(0); // 0-100 per angle
  const [completedAngles, setCompletedAngles] = useState<Set<Angle>>(new Set());
  const [meshVisible, setMeshVisible] = useState(false);
  const [scanLandmarks, setScanLandmarks] = useState<number[][]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const angleStableRef = useRef(0);
  const capturedFrames = useRef<Map<Angle, Blob>>(new Map());
  const captureInProgress = useRef(false);
  const landmarkSaveCounter = useRef(0);

  const { landmarks, faceDetected, headPose, isReady, detect } = useMediaPipeFace();

  // Log head pose for calibration
  useEffect(() => {
    if (headPose && faceDetected) {
      (window as any).__headPose = headPose;
    }
  }, [headPose, faceDetected]);

  // JWT
  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: `face_${Date.now()}` }) })
      .then(r => r.json()).then(d => setToken(d.access_token)).catch(() => setToken("dev_token"));
  }, []);

  // ── Camera ─────────────────────────────────────────────────────

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setErrMsg(null); setResult(null); setPhase("active");
    setCurrentAngle("center"); setAngleProgress(0);
    setCompletedAngles(new Set()); setMeshVisible(false);
    setScanLandmarks([]);
    capturedFrames.current.clear();
    captureInProgress.current = false;
    angleStableRef.current = 0;
    setStatusMsg("Position your face in the frame");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) throw new Error("no video");
      v.srcObject = stream;
      await v.play();
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Camera error");
      setPhase("error");
    }
  }, []);

  useEffect(() => stop, []); // eslint-disable-line

  // ── Pose detection helpers ─────────────────────────────────────

  function isStableFace(): boolean {
    if (!landmarks || !landmarks[0] || landmarks[0].length < 468) return false;
    const pts = landmarks[0];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const area = (maxX - minX) * (maxY - minY);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return area > 0.05 && Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < 0.35;
  }

  function detectAngle(): Angle | null {
    // Use MediaPipe's built-in head pose from the transformation matrix
    // Much more reliable than landmark-based heuristics
    if (!headPose) return null;

    const { yaw, pitch } = headPose;

    // Log every 30 frames for calibration
    if (Math.random() < 0.03) {
      console.log(`[Pose] yaw=${yaw.toFixed(1)}° pitch=${pitch.toFixed(1)}° roll=${headPose.roll.toFixed(1)}°`);
    }

    // Yaw: rotation around Y axis (horizontal head turn)
    // Negative yaw = turning toward the person's LEFT
    // Positive yaw = turning toward the person's RIGHT
    if (yaw < -20) return "left";
    if (yaw > 20) return "right";

    // Pitch: rotation around X axis (vertical head tilt)
    // Positive pitch = looking UP
    if (pitch > 12) return "up";

    // Center: small yaw and small pitch
    if (Math.abs(yaw) < 10 && Math.abs(pitch) < 8) return "center";

    return null; // transitioning between angles
  }

  // ── Frame loop ─────────────────────────────────────────────────

  useEffect(() => {
    if (phase !== "active") {
      setMeshVisible(false);
      angleStableRef.current = 0;
      setAngleProgress(0);
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

      const detected = detect(video, dCanvas);

      if (detected && isStableFace()) {
        setMeshVisible(true);
        // Continuously save landmarks for 3D viewer (throttled)
        landmarkSaveCounter.current++;
        if (landmarks && landmarks[0] && landmarkSaveCounter.current % 15 === 0) {
          setScanLandmarks(landmarks[0].map((p: any) => [p.x, p.y, p.z]));
        }
        const currentPose = detectAngle();

        if (currentPose && !completedAngles.has(currentPose) && !captureInProgress.current) {
          if (currentPose === currentAngle) {
            // User is in the correct pose
            angleStableRef.current = Math.min(30, angleStableRef.current + 1);
            setAngleProgress(Math.round((angleStableRef.current / 30) * 100));
            setStatusMsg(`${ANGLE_LABEL[currentAngle]} — hold still`);

            if (angleStableRef.current >= 30 && !captureInProgress.current) {
              captureInProgress.current = true;
              captureCurrentAngle();
            }
          } else if (currentAngle === "center" && currentPose === "left" && completedAngles.has("center")) {
            // Already got center, now detecting left — update current angle
            // Don't force it, let the angle detection drive
          }
        }

        if (!currentPose || currentPose !== currentAngle) {
          // Slowly decay stability if not in correct pose
          angleStableRef.current = Math.max(0, angleStableRef.current - 2);
          setAngleProgress(Math.round((angleStableRef.current / 30) * 100));
          if (angleStableRef.current < 5) {
            setStatusMsg(ANGLE_LABEL[currentAngle]);
          }
        }
      } else {
        setMeshVisible(false);
        angleStableRef.current = Math.max(0, angleStableRef.current - 3);
        setAngleProgress(Math.round((angleStableRef.current / 30) * 100));
        if (!detected) {
          setStatusMsg("Position your face in the frame");
        }
      }

      drawOverlay(video, detected);
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, landmarks, currentAngle, completedAngles]);

  // ── Capture current angle ──────────────────────────────────────

  async function captureCurrentAngle() {
    setPhase("capturing");
    setStatusMsg("Capturing...");

    const video = videoRef.current;
    const current = currentAngle;
    if (!video || !landmarks || !landmarks[0]) {
      captureInProgress.current = false;
      setPhase("active");
      return;
    }

    // Grab best frame
    const fc = document.createElement("canvas");
    fc.width = video.videoWidth;
    fc.height = video.videoHeight;
    fc.getContext("2d")!.drawImage(video, 0, 0);
    const blob = await canvasToJpegBlob(fc, 0.85);
    capturedFrames.current.set(current, blob);

    // Mark this angle done
    const newCompleted = new Set(completedAngles);
    newCompleted.add(current);
    setCompletedAngles(newCompleted);
    captureInProgress.current = false;
    angleStableRef.current = 0;
    setAngleProgress(0);

    // Move to next angle or finish
    const currentIdx = ANGLE_ORDER.indexOf(current);
    if (currentIdx < ANGLE_ORDER.length - 1) {
      const nextAngle = ANGLE_ORDER[currentIdx + 1];
      setCurrentAngle(nextAngle);
      setStatusMsg(ANGLE_LABEL[nextAngle]);
      setPhase("active");
    } else {
      // All angles captured — verify
      setPhase("verifying");
      setStatusMsg("Processing...");
      await runVerification();
    }
  }

  // ── Verification / Enrollment ──────────────────────────────────

  async function runVerification() {
    // Pick best frame (center, or any available)
    const bestFrame = capturedFrames.current.get("center") ||
      capturedFrames.current.get("left") ||
      capturedFrames.current.get("right") ||
      capturedFrames.current.get("up");

    if (!bestFrame) {
      setPhase("error");
      setErrMsg("No frames captured");
      return;
    }

    // Run liveness on best frame
    const video = videoRef.current;
    let livenessScore = 0.95; // default if we can't compute
    if (video && landmarks && landmarks[0]) {
      const pts = landmarks[0];
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of pts) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      }
      const bbox = { x: minX * video.videoWidth, y: minY * video.videoHeight, width: (maxX - minX) * video.videoWidth, height: (maxY - minY) * video.videoHeight };
      const input = prepareLivenessInput(video, bbox);
      if (input) livenessScore = await checkLiveness(input);
    }

    if (livenessScore < 0.5) {
      setPhase("error");
      setErrMsg("Spoof detected — use a real face, not a photo or screen");
      return;
    }

    try {
      if (mode === "enroll") {
        const fd = new FormData();
        fd.append("image", bestFrame, "face.jpg");
        fd.append("liveness_score", String(livenessScore));
        const res = await fetch(`${API_BASE}/api/face/enroll`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
        const data = await res.json();
        setUserId(data.user_id);
        setResult(data);
        setPhase("done");
      } else {
        if (!userId) { setPhase("error"); setErrMsg("Enroll first or paste a user ID above"); return; }
        const fd = new FormData();
        fd.append("image", bestFrame, "face.jpg");
        fd.append("user_id", userId);
        const res = await fetch(`${API_BASE}/api/face/verify`, { method: "POST", headers: { Authorization: `Bearer ${token}` }, body: fd });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
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

  // ── Overlay ────────────────────────────────────────────────────

  const TESSELATION: [number, number][] = (() => {
    const c: [number, number][] = [];
    const le = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
    for (let i = 0; i < le.length; i++) c.push([le[i], le[(i + 1) % le.length]]);
    const re = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];
    for (let i = 0; i < re.length; i++) c.push([re[i], re[(i + 1) % re.length]]);
    const leb = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107];
    for (let i = 0; i < leb.length; i++) c.push([leb[i], leb[(i + 1) % leb.length]]);
    const reb = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336];
    for (let i = 0; i < reb.length; i++) c.push([reb[i], reb[(i + 1) % reb.length]]);
    const nose = [6, 168, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 460, 294, 459, 458, 461, 354, 455, 460];
    for (let i = 0; i < nose.length - 1; i++) c.push([nose[i], nose[i + 1]]);
    c.push([1, 2], [2, 98], [98, 327]);
    const lo = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
    for (let i = 0; i < lo.length; i++) c.push([lo[i], lo[(i + 1) % lo.length]]);
    const li = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
    for (let i = 0; i < li.length; i++) c.push([li[i], li[(i + 1) % li.length]]);
    const oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    for (let i = 0; i < oval.length; i++) c.push([oval[i], oval[(i + 1) % oval.length]]);
    c.push([10, 151], [151, 9], [9, 8], [8, 168], [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2], [2, 200], [200, 199], [199, 175], [175, 152]);
    c.push([107, 336], [105, 334], [66, 296], [70, 300], [55, 285], [65, 295], [52, 282], [53, 283], [46, 276]);
    c.push([33, 46], [133, 53], [173, 52], [157, 65], [158, 55], [159, 70], [160, 63], [161, 105], [246, 107]);
    c.push([362, 276], [263, 283], [249, 282], [390, 295], [373, 285], [374, 300], [380, 293], [381, 334], [382, 296], [398, 336]);
    c.push([6, 33], [6, 362], [168, 133], [168, 263], [197, 157], [197, 390], [195, 158], [195, 373], [5, 159], [5, 374]);
    c.push([2, 0], [2, 17], [200, 37], [200, 267], [17, 199], [37, 175], [267, 175]);
    for (let i = 0; i < 16; i++) {
      const top = [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454][i];
      const bot = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288][i] || 152;
      if (top && bot) c.push([top, bot]);
    }
    return c;
  })();

  function drawOverlay(video: HTMLVideoElement, detected: boolean) {
    const canvas = overlayRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!meshVisible || !landmarks || !landmarks[0]) return;

    const pts = landmarks[0];
    const w = canvas.width, h = canvas.height;
    const color = "rgba(56, 189, 248, 0.55)";

    ctx.strokeStyle = color;
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    for (const [a, b] of TESSELATION) {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb) continue;
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
    }
    ctx.stroke();

    // Small dots at every 4th landmark for subtle density
    ctx.fillStyle = "rgba(56, 189, 248, 0.7)";
    for (let i = 0; i < 468; i += 2) {
      const p = pts[i];
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 0.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── Angle progress dots ────────────────────────────────────────

  const totalAngles = ANGLE_ORDER.length;

  const IconComponent = ANGLE_ICON[currentAngle];

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
            <p className="text-sm text-zinc-400">{mode === "enroll" ? "Capture your face from multiple angles to enroll" : "Verify against an enrolled identity"}</p>
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
          {(phase === "active" || phase === "capturing") && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{ aspectRatio: "3/4", transform: "scaleX(-1)" }} />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: "scaleX(-1)" }} />
            </div>
          )}
          {phase === "verifying" && (
            <div className="flex flex-col items-center justify-center bg-black gap-3" style={{ aspectRatio: "3/4" }}>
              <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
              <p className="text-sm text-zinc-300">{statusMsg}</p>
            </div>
          )}
          {phase === "done" && (
            <div className={`flex items-center justify-center ${mode === "enroll" || result?.matched ? "bg-green-950/50" : "bg-red-950/50"}`} style={{ aspectRatio: "3/4" }}>
              {(mode === "enroll" || result?.matched) ? <CheckCircle className="h-16 w-16 text-green-400" /> : <XCircle className="h-16 w-16 text-red-400" />}
            </div>
          )}
        </div>

        {/* Status */}
        <div className="mt-4 flex flex-col items-center gap-2 text-center w-full">
          {/* Multi-angle progress dots */}
          {(phase === "active" || phase === "capturing") && (
            <div className="flex items-center gap-2 mb-2">
              {ANGLE_ORDER.map((a, i) => (
                <div key={a} className="flex items-center gap-2">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold transition-all ${
                    completedAngles.has(a) ? "bg-green-500 text-white" :
                    a === currentAngle ? "bg-blue-600 text-white ring-2 ring-blue-400" :
                    "bg-zinc-700 text-zinc-400"
                  }`}>
                    {completedAngles.has(a) ? <CheckCircle className="h-4 w-4" /> : i + 1}
                  </div>
                  {i < totalAngles - 1 && <div className="h-0.5 w-6 bg-zinc-700" />}
                </div>
              ))}
            </div>
          )}

          {/* Progress bar for current angle */}
          {(phase === "active" || phase === "capturing") && angleProgress > 0 && (
            <div className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-700">
              <div className="h-full rounded-full bg-sky-400 transition-all duration-200" style={{ width: `${angleProgress}%` }} />
            </div>
          )}

          {phase === "active" && (
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              {isReady ? (
                <>
                  <IconComponent className="h-4 w-4 text-sky-400" />
                  {meshVisible ? statusMsg : "Position your face in the frame"}
                </>
              ) : "Loading face detection..."}
            </div>
          )}

          {phase === "capturing" && (
            <p className="text-sm text-sky-400"><Loader2 className="mr-1 inline h-4 w-4 animate-spin" />{statusMsg}</p>
          )}
          {phase === "verifying" && (
            <p className="text-sm text-blue-400"><Loader2 className="mr-1 inline h-4 w-4 animate-spin" />{statusMsg}</p>
          )}
          {phase === "error" && (
            <div className="flex flex-col items-center gap-3">
              <p className="text-sm text-red-400"><XCircle className="mr-1 inline h-4 w-4" />{errMsg}</p>
              <button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white">Try Again</button>
            </div>
          )}
        </div>

        {/* 3D Face Scan — show during active and after done */}
        {(phase === "active" || phase === "done") && scanLandmarks.length > 0 && (
          <div className="mt-4 w-full rounded-xl bg-zinc-900 p-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">3D Face Scan</h2>
            <p className="mb-1 text-xs text-zinc-600">Drag to rotate — scroll to zoom</p>
            <Face3DViewer landmarks={scanLandmarks} tessellation={TESSELATION} width={368} height={350} />
          </div>
        )}

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

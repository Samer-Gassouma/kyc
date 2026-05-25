"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useMediaPipeFace, getFaceTessellation } from "@/hooks/useMediaPipeFace";
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
    // Landmark-based head pose — reliable, no matrix decomposition ambiguity.
    // MediaPipe processes the RAW (un-mirrored) camera frame:
    //   RAW image right side  = person's LEFT side
    //   RAW image left side   = person's RIGHT side
    //
    // When turning head LEFT physically:  nose goes RIGHT in the raw image
    // When turning head RIGHT physically: nose goes LEFT in the raw image
    if (!landmarks || !landmarks[0] || landmarks[0].length < 468) return null;
    const pts = landmarks[0];

    const nose = pts[1];       // nose tip
    const chin = pts[152];     // chin
    const leftEar = pts[234];  // left ear / cheek
    const rightEar = pts[454]; // right ear / cheek
    const glabella = pts[168]; // between eyebrows
    const leftEyeOuter = pts[33];
    const rightEyeOuter = pts[263];

    const faceCenterX = (leftEar.x + rightEar.x) / 2;
    const faceW = Math.max(Math.abs(rightEar.x - leftEar.x), 0.01);
    const noseOffX = (nose.x - faceCenterX) / faceW;

    // Pitch: compare nose-to-eyebrow distance vs chin-to-eyebrow distance.
    // When looking UP, nose comes up (closer to glabella in image).
    // y=0 is top of image. Smaller y = higher.
    const eyeY = (glabella.y + leftEyeOuter.y + rightEyeOuter.y) / 3;
    const faceH = Math.max(Math.abs(chin.y - eyeY), 0.01);
    const noseRelEyeY = (nose.y - eyeY) / faceH;

    // Log for calibration (once per 2 seconds)
    if (Math.random() < 0.015) {
      console.log(`[Pose] noseOffX=${noseOffX.toFixed(3)} (neg=right, pos=left)  noseRelEyeY=${noseRelEyeY.toFixed(3)} (small=up)`);
    }

    // noseOffX > 0  = nose on right side of raw image = turned LEFT
    if (noseOffX > 0.10) return "left";
    // noseOffX < 0  = nose on left side of raw image = turned RIGHT
    if (noseOffX < -0.10) return "right";
    // noseRelEyeY < 0.20 = nose close to eye level = looking UP
    if (noseRelEyeY < 0.20) return "up";
    // centered: small offset in both axes
    if (Math.abs(noseOffX) < 0.05 && noseRelEyeY > 0.28) return "center";

    return null;
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

        // Compute and send quality score
        const quality = computeQuality();
        fd.append("quality_score", String(quality));

        // Send landmarks_3d for storage
        if (scanLandmarks.length > 0) {
          fd.append("landmarks_3d", JSON.stringify(scanLandmarks));
        }

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

  function computeQuality(): number {
    if (!headPose || !landmarks || !landmarks[0]) return 0.3;
    const isCentered = Math.abs(headPose.yaw) < 10 && Math.abs(headPose.pitch) < 8;
    const hasAll = landmarks[0].length >= 468;
    const zVals = landmarks[0].map((p: any) => p.z);
    const zSpread = Math.max(...zVals) - Math.min(...zVals);
    const depthScore = Math.min(zSpread / 0.1, 1.0);
    return isCentered && hasAll ? Math.round((0.6 + depthScore * 0.4) * 100) / 100 : 0.3;
  }

  function retry() {
    stop();
    setPhase("idle");
    setErrMsg(null);
    setResult(null);
  }

  // ── Overlay ────────────────────────────────────────────────────


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
    for (const [a, b] of getFaceTessellation()) {
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
            <Face3DViewer landmarks={scanLandmarks} width={368} height={400} />
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

"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import { canvasToJpegBlob } from "@/lib/frameEncoder";
import { useFaceDetection, REGION_EDGES } from "@/hooks/useFaceDetection";
import Link from "next/link";
import { ArrowLeft, Camera, Loader2, CheckCircle, XCircle, UserPlus, Fingerprint } from "lucide-react";
import Face3DViewer from "@/components/kyc/Face3DViewer";

type Mode = "enroll" | "verify";

export default function FacePage() {
  const [mode, setMode] = useState<Mode>("enroll");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<"idle"|"active"|"countdown"|"capturing"|"done"|"error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errMsg, setErrMsg] = useState<string|null>(null);
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState<any>(null);
  const [countdown, setCountdown] = useState(0);
  const [scanPoints, setScanPoints] = useState<{x:number,y:number,z:number}[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream|null>(null);
  const animRef = useRef(0);
  const stableRef = useRef(0);
  const detectCanvasRef = useRef<HTMLCanvasElement|null>(null);

  const { isReady, detect } = useFaceDetection();

  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({session_id:`face_${Date.now()}`}) })
      .then(r=>r.json()).then(d=>setToken(d.access_token)).catch(()=>setToken("dev_token"));
  }, []);

  const stop = useCallback(() => { cancelAnimationFrame(animRef.current); streamRef.current?.getTracks().forEach(t=>t.stop()); streamRef.current=null; }, []);

  const start = useCallback(async () => {
    setErrMsg(null); setResult(null); setPhase("active"); setScanPoints([]);
    stableRef.current=0; setCountdown(0); setStatusMsg("Position your face");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video:{facingMode:"user",width:{ideal:640},height:{ideal:480}}, audio:false });
      streamRef.current=stream;
      const v=videoRef.current; if(!v) throw new Error("no video");
      v.srcObject=stream; await v.play();
    } catch(e) { setErrMsg(e instanceof Error?e.message:"Camera error"); setPhase("error"); }
  }, []);

  useEffect(()=>stop, []); // eslint-disable-line

  // ── Frame loop ─────────────────────────────────────────────────

  useEffect(() => {
    if (phase!=="active") return;
    let running=true;
    if (!detectCanvasRef.current) detectCanvasRef.current=document.createElement("canvas");

    const loop = () => {
      if (!running) return;
      const video=videoRef.current;
      const dCanvas=detectCanvasRef.current;
      if (!video||video.videoWidth===0||!dCanvas) { animRef.current=requestAnimationFrame(loop); return; }

      const { landmarks, faceDetected } = detect(video, dCanvas);
      if (!running) return;

      if (faceDetected && landmarks[0]) {
        const pts = landmarks[0];
        const box = bbox(pts, video.videoWidth, video.videoHeight);

        // Draw colored mesh
        drawMesh(overlayRef.current!, pts, video.videoWidth, video.videoHeight);

        // Quality: face must be > 25% of frame width
        const goodSize = box.width / video.videoWidth > 0.25;

        if (goodSize) {
          stableRef.current++;
          const remaining = Math.max(0, Math.ceil((60 - stableRef.current) / 30));
          setCountdown(remaining);
          if (stableRef.current >= 60) {
            running = false;
            setScanPoints([...pts.map(p=>({x:p.x,y:p.y,z:p.z}))]);
            handleCapture(video, pts);
            return;
          }
          setStatusMsg(remaining > 0 ? `Hold still... ${remaining}` : "Scanning...");
        } else {
          stableRef.current = Math.max(0, stableRef.current - 1);
          setCountdown(0);
          setStatusMsg("Move closer — face too small");
        }
      } else {
        stableRef.current = Math.max(0, stableRef.current - 3);
        setCountdown(0);
        setStatusMsg("No face detected");
        // Clear overlay
        const ov = overlayRef.current;
        if (ov) { const c=ov.getContext("2d"); if(c) c.clearRect(0,0,ov.width,ov.height); }
      }

      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { running=false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // ── Capture ────────────────────────────────────────────────────

  async function handleCapture(video: HTMLVideoElement, pts: any[]) {
    setPhase("capturing"); setStatusMsg("Capturing...");

    // Grab frame
    const c = document.createElement("canvas");
    c.width=video.videoWidth; c.height=video.videoHeight;
    c.getContext("2d")!.drawImage(video,0,0);
    const blob = await canvasToJpegBlob(c, 0.92);

    try {
      if (mode==="enroll") {
        const fd = new FormData();
        fd.append("image", blob, "face.jpg");
        fd.append("liveness_score", "1.0");
        fd.append("landmarks_3d", JSON.stringify({
          points: pts.map(p=>({x:p.x,y:p.y,z:p.z})),
          capturedAt: Date.now(),
          frameWidth: video.videoWidth,
          frameHeight: video.videoHeight,
        }));
        const q = quality(pts);
        fd.append("quality_score", q.toFixed(2));

        const res = await fetch(`${API_BASE}/api/face/enroll`, { method:"POST", headers:{Authorization:`Bearer ${token}`}, body:fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail||`HTTP ${res.status}`);
        const data = await res.json();
        setUserId(data.user_id); setResult(data);
      } else {
        if (!userId) { setPhase("error"); setErrMsg("Enroll first or paste a user ID"); return; }
        const fd = new FormData(); fd.append("image", blob, "face.jpg"); fd.append("user_id", userId);
        const res = await fetch(`${API_BASE}/api/face/verify`, { method:"POST", headers:{Authorization:`Bearer ${token}`}, body:fd });
        if (!res.ok) throw new Error((await res.json().catch(()=>({}))).detail||`HTTP ${res.status}`);
        setResult(await res.json());
      }
      setPhase("done");
    } catch(e) { setPhase("error"); setErrMsg(e instanceof Error?e.message:"Request failed"); }
  }

  function retry() { stop(); setPhase("idle"); setErrMsg(null); setResult(null); }

  // ── Helpers ────────────────────────────────────────────────────

  function bbox(pts: any[], vw:number, vh:number) {
    let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for (const p of pts) { if(p.x<minX)minX=p.x; if(p.x>maxX)maxX=p.x; if(p.y<minY)minY=p.y; if(p.y>maxY)maxY=p.y; }
    return { x:minX*vw, y:minY*vh, width:(maxX-minX)*vw, height:(maxY-minY)*vh };
  }

  function quality(pts: any[]): number {
    if (pts.length<400) return 0.3;
    const zs = pts.map(p=>p.z); const zMin=Math.min(...zs), zMax=Math.max(...zs);
    return Math.round((0.6 + Math.min((zMax-zMin)/0.08, 1)*0.4)*100)/100;
  }

  function drawMesh(canvas: HTMLCanvasElement, pts: any[], vw:number, vh:number) {
    canvas.width=vw; canvas.height=vh;
    const ctx=canvas.getContext("2d"); if(!ctx) return;
    ctx.clearRect(0,0,vw,vh);
    if (pts.length<400) return;

    for (const [, region] of Object.entries(REGION_EDGES)) {
      ctx.strokeStyle = region.color + "99"; // 60% alpha
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (const [a,b] of region.edges) {
        if (a>=pts.length||b>=pts.length) continue;
        ctx.moveTo(pts[a].x*vw, pts[a].y*vh);
        ctx.lineTo(pts[b].x*vw, pts[b].y*vh);
      }
      ctx.stroke();
    }

    // Small dots at every 3rd point
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    for (let i=0; i<pts.length; i+=3) {
      ctx.beginPath(); ctx.arc(pts[i].x*vw, pts[i].y*vh, 1.1, 0, Math.PI*2); ctx.fill();
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"><ArrowLeft className="h-5 w-5"/></Link>
        <Fingerprint className="h-5 w-5 text-blue-400"/><h1 className="text-base font-semibold">Face Pipeline Test</h1>
      </header>
      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        <div className="mb-4 flex w-full rounded-lg bg-zinc-900 p-1">
          <button onClick={()=>{setMode("enroll");retry();}} className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode==="enroll"?"bg-blue-600 text-white":"text-zinc-400"}`}><UserPlus className="mr-2 inline h-4 w-4"/>Enroll</button>
          <button onClick={()=>{setMode("verify");retry();}} className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${mode==="verify"?"bg-blue-600 text-white":"text-zinc-400"}`}><Fingerprint className="mr-2 inline h-4 w-4"/>Verify</button>
        </div>

        {phase==="idle" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <p className="text-sm text-zinc-400">{mode==="enroll"?"Face scan — hold still, auto-captures":"Verify against enrolled identity"}</p>
            <button onClick={start} className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-500"><Camera className="h-5 w-5"/>Start Camera</button>
            {mode==="verify" && <input type="text" value={userId} onChange={e=>setUserId(e.target.value)} placeholder="Paste user_id..." className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200"/>}
          </div>
        )}

        <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{maxWidth:400}}>
          {(phase==="active"||phase==="countdown"||phase==="capturing") && (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{aspectRatio:"3/4",transform:"scaleX(-1)"}}/>
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{transform:"scaleX(-1)"}}/>
              {/* Countdown overlay */}
              {countdown > 0 && countdown <= 3 && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/30">
                  <span className="text-7xl font-bold text-white drop-shadow-lg animate-pulse">{countdown}</span>
                </div>
              )}
            </div>
          )}
          {phase==="done" && scanPoints.length>0 && (
            <Face3DViewer points={scanPoints.map(p=>[p.x,p.y,p.z])} width={368} height={460}/>
          )}
        </div>

        <div className="mt-4 flex flex-col items-center gap-2 text-center">
          {(phase==="active"||phase==="countdown"||phase==="capturing") && (
            <p className="text-sm text-zinc-400"><Camera className="mr-1 inline h-4 w-4"/>{isReady?statusMsg:"Loading..."}</p>
          )}
          {phase==="done" && result && (
            <div className="w-full space-y-2">
              <div className="flex items-center gap-2 rounded-full bg-green-600 px-4 py-1.5 text-sm font-medium text-white"><CheckCircle className="h-4 w-4"/>Scan complete</div>
              {result.confidence != null && (
                <p className="text-xs text-zinc-400">Match: {((result.confidence??0)*100).toFixed(1)}% (threshold {((result.threshold_used??0)*100).toFixed(0)}%)</p>
              )}
              {mode==="enroll" && userId && (
                <>
                  <code className="block break-all rounded bg-zinc-800 p-1.5 text-[10px] text-green-400">{userId}</code>
                  <button onClick={()=>{setMode("verify");setUserId(userId);setPhase("idle");}} className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-medium text-white">Switch to Verify</button>
                </>
              )}
              <button onClick={retry} className="rounded-lg bg-zinc-800 px-4 py-1.5 text-xs text-zinc-300">Scan Again</button>
            </div>
          )}
          {phase==="error" && (
            <div className="flex flex-col items-center gap-3"><p className="text-sm text-red-400"><XCircle className="mr-1 inline h-4 w-4"/>{errMsg}</p><button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm text-white">Try Again</button></div>
          )}
        </div>
      </main>
    </div>
  );
}

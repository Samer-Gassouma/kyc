"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { API_BASE } from "@/lib/apiBase";
import { useFaceDetection, REGION_EDGES } from "@/hooks/useFaceDetection";
import { CheckCircle, Loader2, XCircle, Camera } from "lucide-react";

interface FaceScanStepProps { token: string; userId: string; onComplete: (r: { passed: boolean; confidence: number; user_id: string; liveBlob?: Blob }) => void; }
type State = "idle"|"preparing"|"scanning"|"verifying"|"passed"|"failed";

export default function FaceScanStep({ token, userId, onComplete }: FaceScanStepProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream|null>(null);
  const animRef = useRef(0);
  const onC = useRef(onComplete); onC.current=onComplete;
  const stableRef = useRef(0);
  const dCanvasRef = useRef<HTMLCanvasElement|null>(null);

  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState<string|null>(null);
  const [msg, setMsg] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [cd, setCd] = useState(0);

  const { isReady, detect } = useFaceDetection();

  const cleanup = useCallback(() => { cancelAnimationFrame(animRef.current); streamRef.current?.getTracks().forEach(t=>t.stop()); streamRef.current=null; }, []);

  const startCam = useCallback(async () => {
    try { setState("preparing"); setMsg("Loading...");
      const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}},audio:false});
      streamRef.current=s; const v=videoRef.current; if(!v) throw new Error("no video"); v.srcObject=s; await v.play(); }
    catch(e) { setError(e instanceof Error?e.message:"Camera"); setState("failed"); }
  }, []);

  useEffect(()=>{startCam();return cleanup;},[]); // eslint-disable-line
  useEffect(()=>{ if(isReady&&videoRef.current&&videoRef.current.readyState>=2&&state==="preparing") {setState("scanning");setMsg("Position your face");} },[isReady,state]);

  useEffect(() => {
    if (state!=="scanning") return;
    let running=true;
    if (!dCanvasRef.current) dCanvasRef.current=document.createElement("canvas");

    const loop = () => {
      if (!running) return;
      const v=videoRef.current, dc=dCanvasRef.current;
      if (!v||v.videoWidth===0||!dc) { animRef.current=requestAnimationFrame(loop); return; }

      const { landmarks, faceDetected } = detect(v, dc);
      if (!running) return;

      if (faceDetected && landmarks[0]) {
        const pts=landmarks[0];
        const b=bbox(pts,v.videoWidth,v.videoHeight);
        drawMesh(overlayRef.current!,pts,v.videoWidth,v.videoHeight);

        if (b.width/v.videoWidth>0.25) {
          stableRef.current++;
          const r=Math.max(0,Math.ceil((50-stableRef.current)/30));
          setCd(r);
          if (stableRef.current>=50) { running=false; doCapture(v); return; }
          setMsg(r>0?`Hold still... ${r}`:"Scanning...");
        } else { stableRef.current=Math.max(0,stableRef.current-1); setCd(0); setMsg("Move closer"); }
      } else {
        stableRef.current=Math.max(0,stableRef.current-3); setCd(0); setMsg("No face detected");
        const ov=overlayRef.current; if(ov){const c=ov.getContext("2d");if(c)c.clearRect(0,0,ov.width,ov.height);}
      }
      animRef.current=requestAnimationFrame(loop);
    };
    animRef.current=requestAnimationFrame(loop);
    return ()=>{running=false;};
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  async function doCapture(video:HTMLVideoElement) {
    setState("verifying"); setMsg("Verifying...");
    const c=document.createElement("canvas"); grabFrame(video,c);
    const blob=await canvasToJpegBlob(c,0.85);
    try {
      const fd=new FormData(); fd.append("image",blob,"face.jpg"); fd.append("user_id",userId);
      const res=await fetch(`${API_BASE}/api/face/verify`,{method:"POST",headers:{Authorization:`Bearer ${token}`},body:fd});
      if(!res.ok) throw new Error((await res.json().catch(()=>({}))).detail||`HTTP ${res.status}`);
      const d=await res.json(); setConfidence(d.confidence);
      if(d.matched){setState("passed");onC.current({passed:true,confidence:d.confidence,user_id:d.user_id,liveBlob:blob});}
      else{setState("failed");setError(`No match (${(d.confidence*100).toFixed(0)}%)`);}
    } catch(e) { setState("failed"); setError(e instanceof Error?e.message:"Verification failed"); }
  }

  function bbox(pts:any[],vw:number,vh:number){let x=Infinity,y=Infinity,X=-Infinity,Y=-Infinity;for(const p of pts){if(p.x<x)x=p.x;if(p.x>X)X=p.x;if(p.y<y)y=p.y;if(p.y>Y)Y=p.y;}return{x:x*vw,y:y*vh,width:(X-x)*vw,height:(Y-y)*vh};}
  function drawMesh(canvas:HTMLCanvasElement,pts:any[],vw:number,vh:number){canvas.width=vw;canvas.height=vh;const ctx=canvas.getContext("2d");if(!ctx||pts.length<400)return;ctx.clearRect(0,0,vw,vh);for(const[,r] of Object.entries(REGION_EDGES)){ctx.strokeStyle=r.color+"99";ctx.lineWidth=1.4;ctx.beginPath();for(const[a,b]of r.edges){if(a>=pts.length||b>=pts.length)continue;ctx.moveTo(pts[a].x*vw,pts[a].y*vh);ctx.lineTo(pts[b].x*vw,pts[b].y*vh);}ctx.stroke();}ctx.fillStyle="rgba(255,255,255,0.5)";for(let i=0;i<pts.length;i+=3){ctx.beginPath();ctx.arc(pts[i].x*vw,pts[i].y*vh,1.1,0,Math.PI*2);ctx.fill();}}
  function retry(){stableRef.current=0;setCd(0);setError(null);cleanup();startCam();}

  return(
    <div className="flex flex-col items-center gap-4">
      {error&&<div className="flex flex-col items-center gap-3 p-4 text-center"><p className="text-sm text-red-400">{error}</p><button onClick={retry} className="rounded-full bg-blue-600 px-4 py-2 text-sm text-white">Retry</button></div>}
      <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{maxWidth:400}}>
        {(state==="preparing"||state==="scanning")&&(
          <div className="relative">
            <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" style={{aspectRatio:"3/4",transform:"scaleX(-1)"}}/>
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{transform:"scaleX(-1)"}}/>
            {cd>0&&cd<=3&&<div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/30"><span className="text-7xl font-bold text-white animate-pulse">{cd}</span></div>}
          </div>
        )}
        {state==="verifying"&&<div className="flex items-center justify-center bg-black" style={{aspectRatio:"3/4"}}><Loader2 className="h-10 w-10 animate-spin text-blue-400"/></div>}
        {state==="passed"&&<div className="flex items-center justify-center bg-green-950/50" style={{aspectRatio:"3/4"}}><CheckCircle className="h-16 w-16 text-green-400"/></div>}
      </div>
      <div className="flex flex-col items-center gap-2 text-center">
        {(state==="scanning")&&<div className="flex items-center gap-2 text-sm text-zinc-400"><Camera className="h-4 w-4"/>{msg}</div>}
        {state==="passed"&&<div className="flex items-center gap-2 rounded-full bg-green-600 px-5 py-2 text-sm font-medium text-white"><CheckCircle className="h-4 w-4"/>Verified ({(confidence*100).toFixed(0)}%)</div>}
        {state==="failed"&&!error&&<div className="flex flex-col items-center gap-3"><div className="flex items-center gap-2 rounded-full bg-red-600 px-5 py-2 text-sm text-white"><XCircle className="h-4 w-4"/>Failed</div><button onClick={retry} className="rounded-full bg-blue-600 px-6 py-2.5 text-sm text-white">Try Again</button></div>}
      </div>
    </div>
  );
}

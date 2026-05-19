"use client";

import { useRef, useEffect } from "react";

export type BorderState = "neutral" | "error" | "warning" | "success";

interface CameraOverlayProps {
  borderState: BorderState;
  holdProgress: number; // 0..1
  label?: string;
  bbox?: number[]; // [x1,y1,x2,y2] in video pixel coords
  videoSize?: { w: number; h: number }; // natural video dimensions
  children?: React.ReactNode;
}

const BRACKET_STROKE: Record<BorderState, string> = {
  neutral: "rgba(255,255,255,0.7)",
  error: "rgba(239,68,68,0.9)",
  warning: "rgba(250,204,21,0.9)",
  success: "rgba(34,197,94,0.9)",
};

const BBOX_STROKE: Record<BorderState, string> = {
  neutral: "rgba(59,130,246,0.0)",
  error: "rgba(239,68,68,0.6)",
  warning: "rgba(250,204,21,0.6)",
  success: "rgba(34,197,94,0.6)",
};

/**
 * Camera overlay matching the reference design:
 * - Dark translucent mask with a rectangular cutout
 * - Large corner brackets at the cutout edges
 * - Optional detection bbox rendered inside
 */
export default function CameraOverlay({
  borderState,
  holdProgress,
  label,
  bbox,
  videoSize,
  children,
}: CameraOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Draw the dark mask + corner brackets + detection bbox on a canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;

    // Card cutout dimensions — 85% width, ID aspect ratio 1.586:1, centered
    const cardW = W * 0.85;
    const cardH = cardW / 1.586;
    const cardX = (W - cardW) / 2;
    const cardY = (H - cardH) / 2 - 10; // slight upward offset for bottom UI
    const r = 12; // corner radius

    // ── 1. Dark overlay with rounded-rect cutout ──────────────────
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(0,0,0,0.50)";
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    // Cutout (counter-clockwise winding = hole)
    ctx.moveTo(cardX + r, cardY);
    ctx.lineTo(cardX + cardW - r, cardY);
    ctx.arcTo(cardX + cardW, cardY, cardX + cardW, cardY + r, r);
    ctx.lineTo(cardX + cardW, cardY + cardH - r);
    ctx.arcTo(cardX + cardW, cardY + cardH, cardX + cardW - r, cardY + cardH, r);
    ctx.lineTo(cardX + r, cardY + cardH);
    ctx.arcTo(cardX, cardY + cardH, cardX, cardY + cardH - r, r);
    ctx.lineTo(cardX, cardY + r);
    ctx.arcTo(cardX, cardY, cardX + r, cardY, r);
    ctx.closePath();
    ctx.fill("evenodd");

    // ── 2. Corner brackets ────────────────────────────────────────
    const bracketLen = 36;
    const lw = 3.5;
    ctx.strokeStyle = BRACKET_STROKE[borderState];
    ctx.lineWidth = lw;
    ctx.lineCap = "round";

    // Top-left
    ctx.beginPath();
    ctx.moveTo(cardX, cardY + bracketLen);
    ctx.lineTo(cardX, cardY + r);
    ctx.arcTo(cardX, cardY, cardX + r, cardY, r);
    ctx.lineTo(cardX + bracketLen, cardY);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(cardX + cardW - bracketLen, cardY);
    ctx.lineTo(cardX + cardW - r, cardY);
    ctx.arcTo(cardX + cardW, cardY, cardX + cardW, cardY + r, r);
    ctx.lineTo(cardX + cardW, cardY + bracketLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(cardX, cardY + cardH - bracketLen);
    ctx.lineTo(cardX, cardY + cardH - r);
    ctx.arcTo(cardX, cardY + cardH, cardX + r, cardY + cardH, r);
    ctx.lineTo(cardX + bracketLen, cardY + cardH);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(cardX + cardW - bracketLen, cardY + cardH);
    ctx.lineTo(cardX + cardW - r, cardY + cardH);
    ctx.arcTo(cardX + cardW, cardY + cardH, cardX + cardW, cardY + cardH - r, r);
    ctx.lineTo(cardX + cardW, cardY + cardH - bracketLen);
    ctx.stroke();

    // ── 3. Detection bbox (mapped from video coords → overlay coords) ──
    if (bbox && bbox.length === 4 && videoSize && videoSize.w > 0) {
      const sx = W / videoSize.w;
      const sy = H / videoSize.h;
      const bx1 = bbox[0] * sx;
      const by1 = bbox[1] * sy;
      const bx2 = bbox[2] * sx;
      const by2 = bbox[3] * sy;

      ctx.strokeStyle = BBOX_STROKE[borderState];
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
      ctx.setLineDash([]);
    }

    // ── 4. Hold progress arc ──────────────────────────────────────
    if (holdProgress > 0 && holdProgress < 1) {
      const cx = W / 2;
      const cy = cardY + cardH / 2;
      const radius = 28;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, radius - 3, -Math.PI / 2, -Math.PI / 2 + holdProgress * Math.PI * 2);
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Hold still", cx, cy);
    }
  }, [borderState, holdProgress, bbox, videoSize]);

  // Re-draw on resize
  useEffect(() => {
    const obs = new ResizeObserver(() => {
      canvasRef.current?.dispatchEvent(new Event("resize"));
      // force re-render
    });
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="pointer-events-none absolute inset-0">
      <canvas ref={canvasRef} className="absolute inset-0" />

      {/* Label above card cutout */}
      {label && (
        <div className="absolute left-1/2 -translate-x-1/2" style={{ top: "16%" }}>
          <span className="whitespace-nowrap rounded-full bg-black/50 px-4 py-1.5 text-sm font-medium text-white backdrop-blur-sm">
            {label}
          </span>
        </div>
      )}

      {children}
    </div>
  );
}

"use client";

import { useRef, useCallback } from "react";

export interface CardResult {
  corners: { x: number; y: number }[];  // 4 corners of the card quad
  rect: { x: number; y: number; width: number; height: number }; // bounding rect
  detected: boolean;
}

/**
 * Lightweight card detection using brightness-based blob finding.
 * No OpenCV, no ML — just finds the largest bright rectangular region
 * (the ID card) against a darker background.
 */
export function useCardDetection() {
  const lastResult = useRef<CardResult | null>(null);

  const detect = useCallback((video: HTMLVideoElement, canvas: HTMLCanvasElement): CardResult => {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return { corners: [], rect: { x: 0, y: 0, width: 0, height: 0 }, detected: false };

    // Downscale for speed
    const scale = 0.25;
    const sw = Math.floor(vw * scale), sh = Math.floor(vh * scale);
    canvas.width = sw; canvas.height = sh;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return { corners: [], rect: { x: 0, y: 0, width: 0, height: 0 }, detected: false };

    ctx.drawImage(video, 0, 0, sw, sh);
    const data = ctx.getImageData(0, 0, sw, sh).data;

    // Build grayscale + find average brightness
    const gray = new Uint8Array(sw * sh);
    let totalBright = 0;
    for (let i = 0; i < gray.length; i++) {
      const v = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
      gray[i] = v;
      totalBright += v;
    }
    const avgBright = totalBright / gray.length;

    // Threshold: bright pixels (card is usually brighter than background)
    const thresh = Math.max(avgBright * 1.1, 128);
    const mask = new Uint8Array(sw * sh);
    for (let i = 0; i < gray.length; i++) {
      mask[i] = gray[i] > thresh ? 1 : 0;
    }

    // Find largest connected bright region using simplified scan
    let bestBlob = { minX: 0, minY: 0, maxX: 0, maxY: 0, area: 0 };

    // Row-based run-length encoding to find bright region bounds
    let runStart = -1;
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const idx = y * sw + x;
        if (mask[idx] && runStart < 0) runStart = x;
        if (!mask[idx] && runStart >= 0) {
          // End of run — check if this run overlaps with our best blob
          const runEnd = x - 1;
          if (bestBlob.area === 0) {
            bestBlob = { minX: runStart, minY: y, maxX: runEnd, maxY: y + 1, area: (runEnd - runStart) };
          }
          runStart = -1;
        }
      }
    }

    // Scan all pixels to find the bounding box of the largest bright cluster
    // Simple approach: iterate to find bright region bounds
    let minX = sw, minY = sh, maxX = 0, maxY = 0;
    let brightCount = 0;

    // Use a center-weighted search — card is usually in the middle 60% of frame
    const cx0 = Math.floor(sw * 0.2), cx1 = Math.floor(sw * 0.8);
    const cy0 = Math.floor(sh * 0.2), cy1 = Math.floor(sh * 0.8);

    for (let y = cy0; y < cy1; y++) {
      for (let x = cx0; x < cx1; x++) {
        if (mask[y * sw + x]) {
          brightCount++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Validate: card should cover 10-60% of the center region
    const centerArea = (cx1 - cx0) * (cy1 - cy0);
    const cardArea = (maxX - minX) * (maxY - minY);
    const coverage = cardArea / centerArea;

    if (coverage < 0.08 || coverage > 0.7 || brightCount < 200) {
      return { corners: [], rect: { x: 0, y: 0, width: 0, height: 0 }, detected: false };
    }

    // Add padding and scale back up
    const padX = (maxX - minX) * 0.05;
    const padY = (maxY - minY) * 0.05;
    const rx = Math.max(0, (minX - padX) / scale);
    const ry = Math.max(0, (minY - padY) / scale);
    const rw = Math.min(vw - rx, (maxX - minX + padX * 2) / scale);
    const rh = Math.min(vh - ry, (maxY - minY + padY * 2) / scale);

    const rect = { x: rx, y: ry, width: rw, height: rh };
    const corners = [
      { x: rx, y: ry },
      { x: rx + rw, y: ry },
      { x: rx + rw, y: ry + rh },
      { x: rx, y: ry + rh },
    ];

    lastResult.current = { corners, rect, detected: true };
    return { corners, rect, detected: true };
  }, []);

  /** Draw card highlight on overlay canvas */
  const drawHighlight = useCallback((canvas: HTMLCanvasElement, result: CardResult) => {
    const ctx = canvas.getContext("2d");
    if (!ctx || !result.detected) return;

    const { rect, corners } = result;
    const bracketSize = Math.min(25, rect.width * 0.12, rect.height * 0.12);

    // Glowing green border
    ctx.save();
    ctx.shadowColor = "#22c55e";
    ctx.shadowBlur = 15;
    ctx.strokeStyle = "rgba(34, 197, 94, 0.8)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    // Semi-transparent green fill
    ctx.fillStyle = "rgba(34, 197, 94, 0.08)";
    ctx.fill();

    // Corner brackets
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 4;
    corners.forEach(c => {
      const bx = c.x < rect.x + rect.width / 2 ? 1 : -1;
      const by = c.y < rect.y + rect.height / 2 ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(c.x, c.y + by * bracketSize);
      ctx.lineTo(c.x, c.y);
      ctx.lineTo(c.x + bx * bracketSize, c.y);
      ctx.stroke();
    });
  }, []);

  return { detect, drawHighlight };
}

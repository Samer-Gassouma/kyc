"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { canvasToJpegBlob, grabFrame } from "@/lib/frameEncoder";
import { getWsUrl } from "@/lib/apiBase";

export interface DetectionResult {
  detected: boolean;
  model: string;
  confidence: number;
  bbox: number[];
  rotated_bbox: { angle: number; cx: number; cy: number };
  quality: {
    centered: boolean;
    straight: boolean;
    fully_visible: boolean;
    sharp: boolean;
    lighting_ok: boolean;
  };
  issues: string[];
  ready_to_capture: boolean;
}

export interface UseIDWebSocketReturn {
  detection: DetectionResult | null;
  isConnected: boolean;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  sendFrame: (video: HTMLVideoElement) => Promise<void>;
}

const WS_URL = getWsUrl("/ws/stream");
const TARGET_FPS = 10;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

/**
 * Token bucket rate limiter — caps at 10fps.
 */
class TokenBucket {
  private tokens: number;
  private lastRefill: number;

  constructor(
    private maxTokens: number = TARGET_FPS,
    private refillRate: number = TARGET_FPS
  ) {
    this.tokens = maxTokens;
    this.lastRefill = Date.now();
  }

  consume(): boolean {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRate);
    this.lastRefill = now;

    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }
}

export function useIDWebSocket(): UseIDWebSocketReturn {
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bucketRef = useRef(new TokenBucket());

  useEffect(() => {
    canvasRef.current = document.createElement("canvas");
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data: DetectionResult = JSON.parse(event.data);
          setDetection(data);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err instanceof Error ? err.message : "WebSocket failed");
    }
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setIsConnected(false);
    setDetection(null);
  }, []);

  const sendFrame = useCallback(async (video: HTMLVideoElement) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!bucketRef.current.consume()) return;
    if (!canvasRef.current) return;

    try {
      grabFrame(video, canvasRef.current);
      const blob = await canvasToJpegBlob(canvasRef.current, 0.80);
      const buffer = await blob.arrayBuffer();
      wsRef.current.send(buffer);
    } catch {
      // Frame send failed — non-fatal, skip
    }
  }, []);

  return { detection, isConnected, error, connect, disconnect, sendFrame };
}

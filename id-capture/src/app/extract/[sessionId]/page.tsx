"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/apiBase";
import { Loader2, CheckCircle, XCircle, ExternalLink, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";

export default function TrackSessionPage() {
  const params = useParams();
  const sessionId = params?.sessionId as string;
  const [status, setStatus] = useState<"loading" | "completed" | "failed">("loading");
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [faceCropUrl, setFaceCropUrl] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const fetchStatus = async () => {
      const tokenRes = await fetch(`${API_BASE}/api/auth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: `track_${Date.now()}` }),
      });
      const tokenData = await tokenRes.json().catch(() => ({ access_token: "dev_token" }));
      const token = tokenData.access_token;

      const maxAttempts = 60;
      for (let i = 0; i < maxAttempts; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        try {
          const res = await fetch(`${API_BASE}/api/extract/status/${sessionId}`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (!res.ok) continue;
          const json = await res.json();

          if (json.status === "completed") {
            setStatus("completed");
            setData(json.data ?? null);
            if (json.face_crop_url) {
              const url = `${API_BASE}${json.face_crop_url}?token=${encodeURIComponent(token)}`;
              setFaceCropUrl(url);
            }
            return;
          }
          if (json.status === "failed") {
            setStatus("failed");
            setError(json.error || "Extraction failed");
            return;
          }
        } catch {
          // continue
        }
      }
      setStatus("failed");
      setError("Timed out — session may still be processing");
    };

    fetchStatus();
  }, [sessionId]);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/extract" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <h1 className="text-base font-semibold">Track Extraction</h1>
      </header>

      <main className="mx-auto w-full max-w-4xl flex-1 p-4 space-y-6">
        <p className="text-sm text-zinc-500">Session ID: <code className="text-zinc-300">{sessionId}</code></p>

        {status === "loading" && (
          <div className="rounded-xl bg-zinc-900 p-6 text-center">
            <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-blue-400" />
            <p className="text-sm font-medium text-zinc-300">Checking session status…</p>
          </div>
        )}

        {status === "completed" && data && (
          <>
            <div className="flex items-center gap-2 rounded-lg bg-green-500/10 px-4 py-3 text-sm text-green-400">
              <CheckCircle className="h-4 w-4" />
              Extraction complete
            </div>
            <div className="rounded-xl bg-zinc-900 p-4">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">
                Extracted Data
              </h2>
              <pre className="max-h-[32rem] overflow-auto rounded-lg bg-zinc-950 p-4 text-xs text-green-400">
                {JSON.stringify({ ...data, face_crop_url: faceCropUrl || null }, null, 2)}
              </pre>
            </div>
            {faceCropUrl && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">Face Crop</h2>
                <a href={faceCropUrl} target="_blank" rel="noopener noreferrer"
                   className="inline-flex items-center gap-2 rounded-lg bg-blue-600/20 px-4 py-2 text-sm text-blue-400 ring-1 ring-blue-500/30 transition-colors hover:bg-blue-600/30">
                  <ExternalLink className="h-4 w-4" />
                  Open / Download Face Image
                </a>
                <p className="mt-2 text-xs text-zinc-500">{faceCropUrl}</p>
              </div>
            )}
          </>
        )}

        {status === "failed" && (
          <div className="flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
            <XCircle className="h-4 w-4" />
            {error || "Extraction failed"}
          </div>
        )}
      </main>
    </div>
  );
}

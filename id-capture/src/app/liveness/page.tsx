"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { API_BASE } from "@/lib/apiBase";
import { ArrowLeft, Shield, CheckCircle, XCircle } from "lucide-react";
import Link from "next/link";

const LivenessStep = dynamic(
  () => import("@/components/kyc/LivenessStep"),
  { ssr: false }
);

export default function LivenessPage() {
  const [token, setToken] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "passed" | "failed">("idle");

  useEffect(() => {
    const sid = `liveness_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    setSessionId(sid);

    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    })
      .then((res) => res.json())
      .then((data) => setToken(data.access_token))
      .catch(() => setToken("dev_token"));
  }, []);

  const handleComplete = (passed: boolean) => {
    setStatus(passed ? "passed" : "failed");
  };

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <Shield className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">Liveness Check</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        {!token && (
          <div className="flex flex-1 items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
          </div>
        )}

        {token && status === "idle" && (
          <div className="flex flex-1 flex-col items-center justify-center">
            <div className="mb-4 text-center">
              <p className="text-sm text-zinc-400">
                Position your face within the oval and hold still
              </p>
            </div>
            <LivenessStep
              token={token}
              sessionId={sessionId}
              onComplete={handleComplete}
            />
          </div>
        )}

        {status === "passed" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="rounded-full bg-green-500/10 p-6">
              <CheckCircle className="h-16 w-16 text-green-500" />
            </div>
            <div className="text-center">
              <h2 className="text-xl font-semibold text-zinc-100">
                Liveness Verified
              </h2>
              <p className="mt-2 text-sm text-zinc-400">
                Your live face was successfully confirmed.
              </p>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
            >
              Test Again
            </button>
          </div>
        )}

        {status === "failed" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="rounded-full bg-red-500/10 p-6">
              <XCircle className="h-16 w-16 text-red-500" />
            </div>
            <div className="text-center">
              <h2 className="text-xl font-semibold text-zinc-100">
                Liveness Failed
              </h2>
              <p className="mt-2 text-sm text-zinc-400">
                Could not verify a live face. Please try again with good lighting.
              </p>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
            >
              Retry
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

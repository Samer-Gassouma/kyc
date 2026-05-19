"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { CheckCircle, Loader2, Shield, X, Upload } from "lucide-react";
import { API_BASE } from "@/lib/apiBase";

const LivenessStep = dynamic(() => import("@/components/kyc/LivenessStep"), {
  ssr: false,
});

interface KYCResult {
  kyc_passed: boolean;
  face_match_score?: number | null;
  face_match_possible?: boolean;
  cin_fields?: Record<string, string>;
  reason?: string;
  status?: string;
  message?: string;
}

export default function KYCPage() {
  const [token, setToken] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [step, setStep] = useState<"upload" | "liveness" | "done">("upload");
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [backFile, setBackFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<KYCResult | null>(null);
  const [polling, setPolling] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: `kyc_${Date.now()}` }),
    })
      .then((r) => r.json())
      .then((d) => setToken(d.access_token))
      .catch(() => setToken("dev_token"));
  }, []);

  const handleUpload = async () => {
    if (!frontFile || !backFile) {
      setError("Please select both front and back CIN images");
      return;
    }
    setUploading(true);
    setError("");

    const fd = new FormData();
    fd.append("front", frontFile);
    fd.append("back", backFile);

    try {
      const res = await fetch(`${API_BASE}/api/kyc/start`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
      });
      const data = await res.json();
      if (data.session_id) {
        setSessionId(data.session_id);
        setStep("liveness");
      } else {
        setError(data.detail || "Failed to start KYC");
      }
    } catch {
      setError("Server connection failed");
    }
    setUploading(false);
  };

  const handleLivenessComplete = useCallback(
    async (passed: boolean) => {
      if (!passed || !sessionId) {
        setError("Liveness failed");
        return;
      }
      setPolling(true);
      try {
        const res = await fetch(`${API_BASE}/api/kyc/finalize/${sessionId}`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });
        const data: KYCResult = await res.json();
        if (data.status === "processing") {
          let attempts = 0;
          const poll = setInterval(async () => {
            const sr = await fetch(`${API_BASE}/api/kyc/status/${sessionId}`, {
              headers: { Authorization: `Bearer ${token}` },
            });
            const sd = await sr.json();
            attempts++;
            if (
              sd.cin_status === "completed" ||
              sd.cin_status === "failed" ||
              attempts > 30
            ) {
              clearInterval(poll);
              const fr = await fetch(
                `${API_BASE}/api/kyc/finalize/${sessionId}`,
                {
                  method: "POST",
                  headers: { Authorization: `Bearer ${token}` },
                },
              );
              setResult(await fr.json());
              setStep("done");
              setPolling(false);
            }
          }, 3000);
        } else {
          setResult(data);
          setStep("done");
          setPolling(false);
        }
      } catch {
        setError("Finalize failed");
        setPolling(false);
      }
    },
    [sessionId, token],
  );

  if (step === "done" && result) {
    return (
      <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
        <div className="flex flex-1 flex-col items-center justify-center gap-6 p-8">
          <div
            className={`animate-scaleIn rounded-full p-6 ${result.kyc_passed ? "bg-green-500/10" : "bg-red-500/10"}`}
          >
            {result.kyc_passed ? (
              <CheckCircle className="h-16 w-16 text-green-500" />
            ) : (
              <X className="h-16 w-16 text-red-500" />
            )}
          </div>
          <h2 className="text-xl font-semibold">
            {result.kyc_passed
              ? "Verification Complete"
              : "Verification Failed"}
          </h2>
          <p className="text-sm text-zinc-400">{result.reason}</p>

          {result.cin_fields && Object.keys(result.cin_fields).length > 0 && (
            <div className="w-full max-w-md rounded-xl bg-zinc-900 p-4">
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Extracted CIN Fields
              </h3>
              <dl className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(result.cin_fields).map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-xs text-zinc-500">{k}</dt>
                    <dd className="font-medium">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {result.face_match_score !== null &&
            result.face_match_score !== undefined && (
              <p className="text-sm text-zinc-400">
                Face match: {(result.face_match_score * 100).toFixed(0)}%
              </p>
            )}

          <button
            onClick={() => window.location.reload()}
            className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-medium text-white"
          >
            New Verification
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-blue-400" />
          <h1 className="text-base font-semibold">KYC Verification</h1>
        </div>
      </header>

      <div className="flex flex-1 flex-col items-center p-4">
        {step === "upload" && (
          <div className="flex w-full max-w-md flex-col gap-4 pt-8">
            <h2 className="text-center text-lg font-semibold">
              Upload CIN Card
            </h2>
            <p className="text-center text-sm text-zinc-400">
              Select the front and back images of your National ID card
            </p>

            <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-zinc-600 p-8 hover:border-blue-400">
              <Upload className="h-8 w-8 text-zinc-400" />
              <span className="text-sm text-zinc-400">
                {frontFile ? frontFile.name : "Front side (photo side)"}
              </span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setFrontFile(e.target.files?.[0] || null)}
              />
            </label>

            <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-zinc-600 p-8 hover:border-blue-400">
              <Upload className="h-8 w-8 text-zinc-400" />
              <span className="text-sm text-zinc-400">
                {backFile ? backFile.name : "Back side (address side)"}
              </span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setBackFile(e.target.files?.[0] || null)}
              />
            </label>

            {error && (
              <p className="text-center text-sm text-red-400">{error}</p>
            )}

            <button
              onClick={handleUpload}
              disabled={uploading || !frontFile || !backFile}
              className="rounded-xl bg-blue-600 py-3 text-sm font-medium text-white disabled:opacity-50"
            >
              {uploading ? "Starting..." : "Begin Verification"}
            </button>
          </div>
        )}

        {step === "liveness" && (
          <div className="flex flex-1 flex-col items-center justify-center">
            <div className="mb-4 text-center">
              <p className="text-sm text-zinc-400">
                Look at the camera to verify your identity
              </p>
              {polling && (
                <div className="mt-2 flex items-center justify-center gap-2 text-sm text-blue-400">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing CIN extraction...</span>
                </div>
              )}
            </div>
            <LivenessStep
              token={token}
              sessionId={sessionId}
              onComplete={handleLivenessComplete}
            />
          </div>
        )}
      </div>
    </div>
  );
}

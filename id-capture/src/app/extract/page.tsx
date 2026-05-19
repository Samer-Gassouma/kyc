"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/apiBase";
import { FileUp, Loader2, CheckCircle, XCircle, ScanText, ArrowLeft, ExternalLink } from "lucide-react";
import Link from "next/link";

interface SessionData {
  id_number?: string;
  id_number_valid?: boolean;
  last_name?: string;
  first_name?: string;
  father_lineage?: string;
  date_of_birth?: string;
  place_of_birth?: string;
  mother_name?: string;
  profession?: string;
  address?: string;
  issue_date?: string;
  barcode?: Record<string, unknown>;
  id_number_from_barcode?: string;
}

export default function ExtractPage() {
  const router = useRouter();
  const [token, setToken] = useState<string>("");
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [backFile, setBackFile] = useState<File | null>(null);
  const [frontPreview, setFrontPreview] = useState<string>("");
  const [backPreview, setBackPreview] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionStatus, setSessionStatus] = useState<"idle" | "processing" | "completed" | "failed">("idle");
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [faceCropUrl, setFaceCropUrl] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Get token on mount
  useEffect(() => {
    const sid = `extract_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    })
      .then((res) => res.json())
      .then((data) => setToken(data.access_token))
      .catch(() => setToken("dev_token"));
  }, []);

  const handleFileSelect = (
    e: React.ChangeEvent<HTMLInputElement>,
    side: "front" | "back"
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (side === "front") {
      setFrontFile(file);
      setFrontPreview(URL.createObjectURL(file));
    } else {
      setBackFile(file);
      setBackPreview(URL.createObjectURL(file));
    }
  };

  const pollSession = async (sid: string) => {
    const maxAttempts = 60;
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise((r) => setTimeout(r, 2000));

      try {
        const res = await fetch(`${API_BASE}/api/extract/status/${sid}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) continue;
        const statusJson = await res.json();

        if (statusJson.status === "completed") {
          setSessionStatus("completed");
          setSessionData(statusJson.data ?? null);
          if (statusJson.face_crop_url) {
            const url = `${API_BASE}${statusJson.face_crop_url}?token=${encodeURIComponent(token)}`;
            setFaceCropUrl(url);
          }
          return;
        }

        if (statusJson.status === "failed") {
          setSessionStatus("failed");
          setError(statusJson.error || "Extraction failed");
          return;
        }
      } catch {
        // continue polling
      }
    }
    setSessionStatus("failed");
    setError("Extraction timed out — try again");
  };

  const handleSubmit = async () => {
    if (!frontFile || !backFile || !token) {
      setError("Please select both front and back images");
      return;
    }
    setSessionStatus("processing");
    setError(null);
    setSessionData(null);
    setFaceCropUrl("");

    const formData = new FormData();
    formData.append("front", frontFile);
    formData.append("back", backFile);

    try {
      const res = await fetch(`${API_BASE}/api/extract/start`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Start failed: ${res.status}`);
      }
      const result = await res.json();
      if (result.session_id) {
        setSessionId(result.session_id);
        router.push(`/extract/${result.session_id}`);
      } else {
        throw new Error("No session ID returned");
      }
    } catch (err) {
      setSessionStatus("failed");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  const showResults = sessionStatus === "completed" || sessionStatus === "failed";

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link href="/" className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <ScanText className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">CIN Data Extraction</h1>
      </header>

      <main className="mx-auto w-full max-w-4xl flex-1 p-4">
        {/* Upload Section */}
        {!showResults && (
          <div className="mb-6 space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {/* Front upload */}
              <div
                className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition-colors ${
                  frontPreview ? "border-blue-500/50 bg-blue-500/5" : "border-zinc-700 bg-zinc-900"
                }`}
              >
                {frontPreview ? (
                  <img src={frontPreview} alt="Front preview" className="h-48 w-full object-contain" />
                ) : (
                  <>
                    <FileUp className="mb-3 h-8 w-8 text-zinc-500" />
                    <p className="text-sm font-medium text-zinc-300">Front of CIN</p>
                    <p className="mt-1 text-xs text-zinc-500">Click or drag to upload</p>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  className="absolute inset-0 cursor-pointer opacity-0"
                  onChange={(e) => handleFileSelect(e, "front")}
                />
              </div>

              {/* Back upload */}
              <div
                className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition-colors ${
                  backPreview ? "border-blue-500/50 bg-blue-500/5" : "border-zinc-700 bg-zinc-900"
                }`}
              >
                {backPreview ? (
                  <img src={backPreview} alt="Back preview" className="h-48 w-full object-contain" />
                ) : (
                  <>
                    <FileUp className="mb-3 h-8 w-8 text-zinc-500" />
                    <p className="text-sm font-medium text-zinc-300">Back of CIN</p>
                    <p className="mt-1 text-xs text-zinc-500">Click or drag to upload</p>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  className="absolute inset-0 cursor-pointer opacity-0"
                  onChange={(e) => handleFileSelect(e, "back")}
                />
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
                <XCircle className="h-4 w-4" />
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={!frontFile || !backFile || sessionStatus === "processing" || !token}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
            >
              {sessionStatus === "processing" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing…
                </>
              ) : (
                <>
                  <ScanText className="h-4 w-4" />
                  Extract Data
                </>
              )}
            </button>
          </div>
        )}

        {/* Session Status */}
        {sessionStatus === "processing" && (
          <div className="mb-6 rounded-xl bg-zinc-900 p-6 text-center">
            <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-blue-400" />
            <p className="text-sm font-medium text-zinc-300">Extracting data…</p>
            <p className="mt-1 text-xs text-zinc-500">Session: {sessionId}</p>
          </div>
        )}

        {/* Results */}
        {sessionStatus === "completed" && sessionData && (
          <div className="space-y-6">
            {/* Status badge */}
            <div className="flex items-center gap-2 rounded-lg bg-green-500/10 px-4 py-3 text-sm text-green-400">
              <CheckCircle className="h-4 w-4" />
              Extraction complete — Session {sessionId}
            </div>

            {/* Extracted JSON with face crop link */}
            <div className="rounded-xl bg-zinc-900 p-4">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">
                Extracted Data
              </h2>
              <pre className="max-h-[32rem] overflow-auto rounded-lg bg-zinc-950 p-4 text-xs text-green-400">
                {JSON.stringify(
                  {
                    ...sessionData,
                    face_crop_url: faceCropUrl || null,
                  },
                  null,
                  2
                )}
              </pre>
            </div>

            {/* Face crop download link */}
            {faceCropUrl && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">
                  Face Crop
                </h2>
                <a
                  href={faceCropUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600/20 px-4 py-2 text-sm text-blue-400 ring-1 ring-blue-500/30 transition-colors hover:bg-blue-600/30"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open / Download Face Image
                </a>
                <p className="mt-2 text-xs text-zinc-500">{faceCropUrl}</p>
              </div>
            )}
          </div>
        )}

        {sessionStatus === "failed" && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
              <XCircle className="h-4 w-4" />
              Extraction failed — {error || "Unknown error"}
            </div>
            <button
              onClick={() => {
                setSessionStatus("idle");
                setError(null);
                setSessionData(null);
                setFaceCropUrl("");
                setSessionId("");
              }}
              className="rounded-xl bg-zinc-800 px-4 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-700"
            >
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

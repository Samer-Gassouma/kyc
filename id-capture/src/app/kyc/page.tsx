"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/apiBase";
import {
  ArrowLeft,
  ScanText,
  Loader2,
  CheckCircle,
  XCircle,
  ExternalLink,
} from "lucide-react";
import Link from "next/link";
import StepProgress, { KYCStep } from "@/components/kyc/StepProgress";
import IDCaptureStep from "@/components/kyc/IDCaptureStep";
import FaceScanStep from "@/components/kyc/FaceScanStep";

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

interface VerificationResult {
  passed: boolean;
  confidence: number;
  user_id: string;
  liveBlob?: Blob;
}

type Phase =
  | "front_id"
  | "back_id"
  | "extracting"
  | "face_scan"
  | "completed"
  | "failed";

export default function KYCPage() {
  const [token, setToken] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("front_id");
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [faceCropUrl, setFaceCropUrl] = useState<string>("");
  const [userId, setUserId] = useState<string>("");
  const [verificationResult, setVerificationResult] =
    useState<VerificationResult | null>(null);
  const [docMatch, setDocMatch] = useState<{match:boolean;similarity:number}|null>(null);
  const [error, setError] = useState<string | null>(null);
  const frontBlobRef = useRef<Blob | null>(null);
  const backBlobRef = useRef<Blob | null>(null);
  const cinFaceBlobRef = useRef<Blob | null>(null);

  // Get JWT on mount
  useEffect(() => {
    const sid = `kyc_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    })
      .then((res) => res.json())
      .then((data) => setToken(data.access_token))
      .catch(() => setToken("dev_token"));
  }, []);

  // ── Step 1: Front captured ────────────────────────────────────────
  const handleFrontComplete = useCallback(
    (_captureId: string, blob: Blob) => {
      frontBlobRef.current = blob;
      setPhase("back_id");
    },
    []
  );

  // ── Step 2: Back captured → extract CIN ───────────────────────────
  const handleBackComplete = useCallback(
    async (_captureId: string, blob: Blob) => {
      backBlobRef.current = blob;
      setPhase("extracting");
      setError(null);

      const frontBlob = frontBlobRef.current;
      if (!frontBlob) {
        setPhase("failed");
        setError("Front image missing");
        return;
      }

      try {
        const formData = new FormData();
        formData.append("front", frontBlob, "front.jpg");
        formData.append("back", blob, "back.jpg");

        const res = await fetch(`${API_BASE}/api/extract/start`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        const sid = result.session_id;

        // Poll until extraction done
        for (let i = 0; i < 180; i++) {
          await new Promise((r) => setTimeout(r, 2000));

          const statusRes = await fetch(
            `${API_BASE}/api/extract/status/${sid}`,
            { headers: { Authorization: `Bearer ${token}` } }
          );
          if (!statusRes.ok) continue;
          const statusJson = await statusRes.json();

          if (statusJson.status === "completed") {
            setSessionData(statusJson.data ?? null);
            if (statusJson.face_crop_url) {
              setFaceCropUrl(
                `${API_BASE}${statusJson.face_crop_url}?token=${encodeURIComponent(token)}`
              );
            }

            // Auto-enroll CIN face: fetch the face crop and POST to /api/face/enroll
            try {
              const faceCropRes = await fetch(
                `${API_BASE}${statusJson.face_crop_url}?token=${encodeURIComponent(token)}`
              );
              if (faceCropRes.ok) {
                const faceBlob = await faceCropRes.blob();
                cinFaceBlobRef.current = faceBlob; // save for cross-check
                const enrollForm = new FormData();
                enrollForm.append("image", faceBlob, "cin_face.jpg");
                const enrollRes = await fetch(`${API_BASE}/api/face/enroll`, {
                  method: "POST",
                  headers: { Authorization: `Bearer ${token}` },
                  body: enrollForm,
                });
                if (enrollRes.ok) {
                  const enrollData = await enrollRes.json();
                  setUserId(enrollData.user_id);
                }
              }
            } catch {
              // non-fatal: can still try face scan
            }

            setPhase("face_scan");
            return;
          }

          if (statusJson.status === "failed") {
            setError(statusJson.error || "Extraction failed");
            setPhase("failed");
            return;
          }
        }

        setError("Extraction timed out");
        setPhase("failed");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
        setPhase("failed");
      }
    },
    [token]
  );

  // ── Step 3: Face scan done → cross-check against CIN document ──────
  const handleFaceScanComplete = useCallback(
    async (result: VerificationResult) => {
      if (!result.passed) {
        setPhase("failed");
        setError("Face verification did not pass");
        return;
      }
      setVerificationResult(result);

      // Cross-check: live face vs CIN document photo
      const cinBlob = cinFaceBlobRef.current;
      const liveBlob = result.liveBlob;
      if (cinBlob && liveBlob) {
        try {
          const crossForm = new FormData();
          crossForm.append("document_image", cinBlob, "cin.jpg");
          crossForm.append("live_image", liveBlob, "live.jpg");

          const crossRes = await fetch(`${API_BASE}/api/face/verify-against-document`, {
            method: "POST",
            headers: { Authorization: `Bearer ${token}` },
            body: crossForm,
          });

          if (crossRes.ok) {
            setDocMatch(await crossRes.json());
          }
        } catch { /* non-fatal */ }
      }

      setPhase("completed");
    },
    [token]
  );

  // ── Step helpers ───────────────────────────────────────────────────
  const completedSteps: KYCStep[] = [];
  if (
    phase === "back_id" ||
    phase === "extracting" ||
    phase === "face_scan" ||
    phase === "completed"
  )
    completedSteps.push("front_id");
  if (
    phase === "extracting" ||
    phase === "face_scan" ||
    phase === "completed"
  )
    completedSteps.push("back_id");
  if (phase === "completed")
    completedSteps.push("face_scan");

  const currentStep: KYCStep =
    phase === "front_id"
      ? "front_id"
      : phase === "back_id"
      ? "back_id"
      : "face_scan";

  const showProgress =
    phase === "front_id" ||
    phase === "back_id" ||
    phase === "face_scan";

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <Link
          href="/"
          className="rounded-full p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <ScanText className="h-5 w-5 text-blue-400" />
        <h1 className="text-base font-semibold">KYC Verification</h1>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col items-center p-4">
        {!token && (
          <div className="flex flex-1 items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
          </div>
        )}

        {token && showProgress && (
          <StepProgress
            currentStep={currentStep}
            completedSteps={completedSteps}
          />
        )}

        {/* Step 1: Front ID */}
        {phase === "front_id" && (
          <IDCaptureStep
            side="front"
            token={token}
            onCaptureComplete={handleFrontComplete}
          />
        )}

        {/* Step 2: Back ID */}
        {phase === "back_id" && (
          <IDCaptureStep
            side="back"
            token={token}
            onCaptureComplete={handleBackComplete}
          />
        )}

        {/* Extracting spinner */}
        {phase === "extracting" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <Loader2 className="h-10 w-10 animate-spin text-blue-400" />
            <p className="text-sm font-medium text-zinc-300">
              Extracting CIN data...
            </p>
            <p className="text-xs text-zinc-500">Running OCR on both sides</p>
          </div>
        )}

        {/* Step 3: Face scan */}
        {phase === "face_scan" && (
          <FaceScanStep
            token={token}
            userId={userId}
            onComplete={handleFaceScanComplete}
          />
        )}

        {/* Results */}
        {phase === "completed" && (
          <div className="w-full space-y-4 pt-4">
            <div className="flex items-center gap-2 rounded-lg bg-green-500/10 px-4 py-3 text-sm text-green-400">
              <CheckCircle className="h-4 w-4" />
              Verification complete
            </div>

            {/* Document face match */}
            {docMatch && (
              <div className={`rounded-xl p-4 ${docMatch.match ? "bg-green-500/10 ring-1 ring-green-500/30" : "bg-red-500/10 ring-1 ring-red-500/30"}`}>
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">ID Document Match</h2>
                <div className="flex items-center gap-2">
                  {docMatch.match ? <CheckCircle className="h-5 w-5 text-green-400" /> : <XCircle className="h-5 w-5 text-red-400" />}
                  <span className={`text-sm font-medium ${docMatch.match ? "text-green-400" : "text-red-400"}`}>
                    {docMatch.match ? "ID Photo Matched" : "ID Photo Mismatch"} — {(docMatch.similarity*100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}

            {/* Face verification result */}
            {verificationResult && (
              <div
                className={`rounded-xl p-4 ${
                  verificationResult.passed
                    ? "bg-green-500/10 ring-1 ring-green-500/30"
                    : "bg-red-500/10 ring-1 ring-red-500/30"
                }`}
              >
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  Face Verification
                </h2>
                <div className="flex items-center gap-2">
                  {verificationResult.passed ? (
                    <CheckCircle className="h-5 w-5 text-green-400" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-400" />
                  )}
                  <span
                    className={`text-sm font-medium ${
                      verificationResult.passed ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {verificationResult.passed ? "Verified" : "Not verified"} —{" "}
                    {(verificationResult.confidence * 100).toFixed(1)}%
                    confidence
                  </span>
                </div>
                <p className="mt-1 text-xs text-zinc-500">
                  User ID: {verificationResult.user_id}
                </p>
              </div>
            )}

            {/* Extracted data */}
            {sessionData && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  Extracted Data
                </h2>
                <pre className="max-h-[32rem] overflow-auto rounded-lg bg-zinc-950 p-4 text-xs text-green-400">
                  {JSON.stringify(
                    {
                      ...sessionData,
                      face_verification: verificationResult,
                      face_crop_url: faceCropUrl || null,
                    },
                    null,
                    2
                  )}
                </pre>
              </div>
            )}

            {faceCropUrl && (
              <div className="rounded-xl bg-zinc-900 p-4">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  Face Crop
                </h2>
                <a
                  href={faceCropUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600/20 px-4 py-2 text-sm text-blue-400 ring-1 ring-blue-500/30 transition-colors hover:bg-blue-600/30"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open Face Image
                </a>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {phase === "failed" && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
              <XCircle className="h-4 w-4" />
              {error || "Verification failed"}
            </div>
            <button
              onClick={() => window.location.reload()}
              className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-blue-500"
            >
              Start Over
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import StepProgress, { KYCStep } from "@/components/kyc/StepProgress";
import { useCaptureStatus } from "@/hooks/useCaptureStatus";
import { CheckCircle, Loader2, Shield, X } from "lucide-react";

const IDCaptureStep = dynamic(
  () => import("@/components/kyc/IDCaptureStep"),
  { ssr: false }
);
const LivenessStep = dynamic(
  () => import("@/components/kyc/LivenessStep"),
  { ssr: false }
);

import { API_BASE } from "@/lib/apiBase";

type OverallStatus = "in_progress" | "completed" | "failed";

export default function KYCPage() {
  const [token, setToken] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [currentStep, setCurrentStep] = useState<KYCStep>("front_id");
  const [completedSteps, setCompletedSteps] = useState<KYCStep[]>([]);
  const [overallStatus, setOverallStatus] = useState<OverallStatus>("in_progress");
  const [frontCaptureId, setFrontCaptureId] = useState<string>("");
  const [backCaptureId, setBackCaptureId] = useState<string>("");

  const frontStatus = useCaptureStatus();
  const backStatus = useCaptureStatus();

  // Obtain JWT on mount
  useEffect(() => {
    const sid = `kyc_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    setSessionId(sid);

    fetch(`${API_BASE}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    })
      .then((res) => res.json())
      .then((data) => setToken(data.access_token))
      .catch(() => {
        // If backend is not available, use a placeholder token for dev
        setToken("dev_token");
      });
  }, []);

  // Handle front ID capture complete
  const handleFrontComplete = useCallback(
    (captureId: string) => {
      setFrontCaptureId(captureId);
      setCompletedSteps((prev) => [...prev, "front_id"]);
      setCurrentStep("back_id");
      if (token && captureId) {
        frontStatus.startPolling(captureId, token);
      }
    },
    [token, frontStatus]
  );

  // Handle back ID capture complete
  const handleBackComplete = useCallback(
    (captureId: string) => {
      setBackCaptureId(captureId);
      setCompletedSteps((prev) => [...prev, "back_id"]);
      setCurrentStep("liveness");
      if (token && captureId) {
        backStatus.startPolling(captureId, token);
      }
    },
    [token, backStatus]
  );

  // Handle liveness complete
  const handleLivenessComplete = useCallback(
    (passed: boolean) => {
      if (passed) {
        setCompletedSteps((prev) => [...prev, "liveness"]);
        setOverallStatus("completed");
      } else {
        setOverallStatus("failed");
      }
    },
    []
  );

  // ── Final success screen ─────────────────────────────────────────
  if (overallStatus === "completed") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-6 p-8">
        <div className="animate-scaleIn rounded-full bg-green-500/10 p-6">
          <CheckCircle className="h-16 w-16 text-green-500" />
        </div>
        <div className="text-center">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
            Verification Complete
          </h2>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
            Your identity has been verified successfully.
          </p>
        </div>

        {/* OCR results summary */}
        <div className="w-full max-w-md space-y-3">
          {frontStatus.data?.mrz_parsed && (
            <div className="rounded-xl bg-white p-4 shadow-sm dark:bg-zinc-900">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Extracted Information
              </h3>
              <dl className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(
                  frontStatus.data.mrz_parsed as Record<string, string>
                ).map(
                  ([key, value]) =>
                    value && (
                      <div key={key}>
                        <dt className="text-zinc-400 capitalize">
                          {key.replace(/_/g, " ")}
                        </dt>
                        <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                          {String(value)}
                        </dd>
                      </div>
                    )
                )}
              </dl>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Loading token ────────────────────────────────────────────────
  if (!token) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  // ── Main KYC flow ────────────────────────────────────────────────
  return (
    <div className="flex flex-1 flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-blue-600" />
          <h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
            Identity Verification
          </h1>
        </div>
        <button className="rounded-full p-1.5 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-800">
          <X className="h-5 w-5" />
        </button>
      </header>

      {/* Step banner */}
      <div className="bg-blue-600 py-2.5 text-center text-sm font-medium text-white">
        {currentStep === "front_id" && "Take an ID Photo"}
        {currentStep === "back_id" && "Take Back Side Photo"}
        {currentStep === "liveness" && "Liveness Verification"}
      </div>

      {/* Progress */}
      <StepProgress currentStep={currentStep} completedSteps={completedSteps} />

      {/* Active step */}
      <div className="flex flex-1 flex-col items-center px-4 pb-8">
        {currentStep === "front_id" && (
          <IDCaptureStep
            side="front"
            token={token}
            onCaptureComplete={handleFrontComplete}
          />
        )}

        {currentStep === "back_id" && (
          <IDCaptureStep
            side="back"
            token={token}
            onCaptureComplete={handleBackComplete}
          />
        )}

        {currentStep === "liveness" && (
          <LivenessStep token={token} sessionId={sessionId} frontCaptureId={frontCaptureId} onComplete={handleLivenessComplete} />
        )}
      </div>

      {/* Footer */}
      <footer className="py-3 text-center text-[11px] text-zinc-400">
        Secured with end-to-end encryption
      </footer>
    </div>
  );
}

"use client";

import clsx from "clsx";
import { CheckCircle, Loader2, RotateCcw, XCircle } from "lucide-react";

interface CaptureReviewProps {
  imageUrl: string;
  cropImageUrl?: string;
  status: "validating" | "success" | "failed";
  rejectionReason?: string | null;
  onRetry: () => void;
  onContinue: () => void;
}

export default function CaptureReview({
  imageUrl,
  cropImageUrl,
  status,
  rejectionReason,
  onRetry,
  onContinue,
}: CaptureReviewProps) {
  return (
    <div className="flex flex-col items-center gap-4 p-4">
      {/* Primary image: cropped card if available, else original */}
      <div
        className={clsx(
          "relative overflow-hidden rounded-xl border-2",
          "w-full max-w-[400px]",
          {
            "border-zinc-300": status === "validating",
            "border-green-500": status === "success",
            "border-red-500 animate-shake": status === "failed",
          }
        )}
        style={{ aspectRatio: "1.586 / 1" }}
      >
        <img
          src={cropImageUrl || imageUrl}
          alt="Captured ID"
          className="h-full w-full object-cover"
        />

        {/* Status overlay */}
        <div
          className={clsx(
            "absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/40 backdrop-blur-[2px]",
            { hidden: status === "success" }
          )}
        >
          {status === "validating" && (
            <>
              <Loader2 className="h-10 w-10 animate-spin text-white" />
              <span className="text-sm font-medium text-white">
                Validating capture...
              </span>
            </>
          )}

          {status === "failed" && (
            <>
              <XCircle className="h-10 w-10 text-red-400" />
              <span className="text-sm font-medium text-red-200">
                {rejectionReason || "Validation failed"}
              </span>
            </>
          )}
        </div>

        {/* Success checkmark */}
        {status === "success" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-scaleIn rounded-full bg-green-500/90 p-3">
              <CheckCircle className="h-8 w-8 text-white" />
            </div>
          </div>
        )}
      </div>

      {/* Original image thumbnail (when crop is shown) */}
      {cropImageUrl && status !== "validating" && (
        <div className="flex flex-col items-center gap-1.5">
          <span className="text-[11px] font-medium text-zinc-400">
            Original photo
          </span>
          <div className="relative h-[80px] w-[128px] overflow-hidden rounded-lg border border-zinc-600/30">
            <img
              src={imageUrl}
              alt="Original"
              className="h-full w-full object-cover opacity-70"
            />
          </div>
        </div>
      )}

      {/* Rejection reason card */}
      {status === "failed" && rejectionReason && (
        <div className="w-full max-w-[400px] rounded-xl border border-red-200/30 bg-red-500/10 px-4 py-3 text-center">
          <p className="text-sm font-medium text-red-300">{rejectionReason}</p>
          <p className="mt-1 text-xs text-red-200/70">
            Try positioning your card more clearly and retry.
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        {status === "failed" && (
          <button
            onClick={onRetry}
            className="flex items-center gap-2 rounded-full bg-zinc-800 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-zinc-700"
          >
            <RotateCcw className="h-4 w-4" />
            Retry
          </button>
        )}

        {status === "success" && (
          <button
            onClick={onContinue}
            className="flex items-center gap-2 rounded-full bg-blue-600 px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-500"
          >
            <CheckCircle className="h-4 w-4" />
            Continue
          </button>
        )}
      </div>
    </div>
  );
}

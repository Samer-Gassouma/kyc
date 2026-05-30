"use client";

import { useKYCStore } from "@/store/kycStore";

function formatRejectionReason(reason: string): string {
  const labels: Record<string, string> = {
    liveness_check_failed: "Liveness check failed — couldn't confirm you're a real person",
    phone_check_failed: "Phone verification not completed",
    document_check_failed: "Document was not uploaded or couldn't be read",
    face_check_failed: "Face didn't match the photo on your document",
    email_check_failed: "Email verification not completed",
    duplicate_identity: "This identity has already been verified",
  };
  return labels[reason] || reason.replace(/_/g, " ");
}

export default function RejectedScreen() {
  const { rejectionReasons, reset } = useKYCStore();

  return (
    <div className="flex flex-col items-center gap-6 py-12 text-center">
      <div className="w-20 h-20 rounded-full bg-red-500/20 flex items-center justify-center text-4xl">
        ❌
      </div>
      <div>
        <h2 className="text-2xl font-bold text-white">Verification Failed</h2>
        <div className="space-y-1 mt-2">
          {rejectionReasons.map((reason) => (
            <p key={reason} className="text-red-400 text-sm">
              {formatRejectionReason(reason)}
            </p>
          ))}
          {rejectionReasons.length === 0 && (
            <p className="text-zinc-400 text-sm">
              Your verification could not be completed. Please try again.
            </p>
          )}
        </div>
      </div>
      <div className="flex gap-3 w-full">
        <button
          onClick={reset}
          className="flex-1 py-4 rounded-full border border-zinc-600 text-white text-base"
        >
          Try Again
        </button>
        <button className="flex-1 py-4 rounded-full bg-white text-black font-semibold text-base">
          Contact Support
        </button>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useRef } from "react";
import { useKYCStore } from "@/store/kycStore";

export default function VerifyingScreen() {
  const { pollStatus, submitSession, step } = useKYCStore();
  const submittedRef = useRef(false);

  useEffect(() => {
    // Submit once
    if (!submittedRef.current) {
      submittedRef.current = true;
      submitSession();
    }

    // Poll every 3 seconds
    const interval = setInterval(() => {
      pollStatus();
    }, 3000);

    return () => clearInterval(interval);
  }, [submitSession, pollStatus]);

  // If we've already transitioned, stop showing this
  if (step === "APPROVED" || step === "REJECTED") return null;

  return (
    <div className="flex flex-col items-center gap-6 py-12">
      <div className="relative">
        <div className="w-20 h-20 bg-[#0f0f0f] rounded-2xl flex items-center justify-center text-4xl animate-pulse">
          🗂️
        </div>
        <div className="absolute -top-1 -right-1 flex gap-0.5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      </div>

      <div className="text-center">
        <h2 className="text-xl font-bold text-white">Verifying your ID</h2>
        <p className="text-zinc-400 text-sm mt-2">
          This may take a few minutes. Please wait while we complete the check.
        </p>
      </div>
    </div>
  );
}

"use client";

import { useKYCStore } from "@/store/kycStore";

export default function IntroScreen() {
  const { createSession, setStep } = useKYCStore();

  const handleStart = async () => {
    await createSession();
    setStep("LIVENESS");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">
          Let&apos;s get you verified
        </h1>
        <p className="text-zinc-400 mt-1">Follow the simple steps below</p>
      </div>

      <div className="space-y-0 divide-y divide-zinc-800">
        {[
          "Perform a liveness check",
          "Phone verification",
          "Provide identity document",
          "Email verification",
        ].map((label, i) => (
          <div key={i} className="flex items-center gap-4 py-4">
            <span className="w-8 h-8 rounded-full border border-zinc-600 flex items-center justify-center text-sm text-zinc-300">
              {i + 1}
            </span>
            <span className="text-white">{label}</span>
          </div>
        ))}
      </div>

      <div className="space-y-3">
        <button
          onClick={handleStart}
          className="w-full py-4 rounded-full bg-white text-black font-semibold text-base"
        >
          Start verification
        </button>
        <button className="w-full py-4 rounded-full border border-zinc-600 text-white text-base">
          Continue on phone
        </button>
      </div>
    </div>
  );
}

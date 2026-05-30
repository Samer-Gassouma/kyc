"use client";

import { useKYCStore } from "@/store/kycStore";

export default function ApprovedScreen() {
  const { reset } = useKYCStore();

  return (
    <div className="flex flex-col items-center gap-6 py-12 text-center">
      <div className="w-20 h-20 rounded-full bg-green-500/20 flex items-center justify-center text-4xl">
        ✅
      </div>
      <div>
        <h2 className="text-2xl font-bold text-white">Verification Complete</h2>
        <p className="text-zinc-400 mt-2">
          Your identity has been successfully verified.
        </p>
      </div>
      <button
        onClick={reset}
        className="w-full py-4 rounded-full bg-white text-black font-semibold text-base"
      >
        Done
      </button>
    </div>
  );
}

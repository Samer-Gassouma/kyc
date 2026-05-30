"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useKYCStore, KYCStep } from "@/store/kycStore";

interface OtpScreenProps {
  type: "phone" | "email";
  target: string;
  onVerify: (otp: string) => Promise<boolean>;
  onResend: () => Promise<boolean>;
  nextStep: KYCStep;
}

export default function OtpScreen({ type, target, onVerify, onResend, nextStep }: OtpScreenProps) {
  const { setStep } = useKYCStore();
  const [digits, setDigits] = useState<string[]>(["", "", "", "", "", ""]);
  const [cooldown, setCooldown] = useState(60);
  const [error, setError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Countdown timer
  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setInterval(() => setCooldown((c) => c - 1), 1000);
    return () => clearInterval(t);
  }, [cooldown]);

  const handleDigit = (i: number, value: string) => {
    if (!/^\d?$/.test(value)) return;
    const next = [...digits];
    next[i] = value;
    setDigits(next);
    setError(null);

    // Auto-advance to next input
    if (value && i < 5) {
      inputRefs.current[i + 1]?.focus();
    }

    // Auto-submit when all 6 filled
    if (value && next.every((d) => d !== "")) {
      handleSubmit(next.join(""));
    }
  };

  const handleKeyDown = (i: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && !digits[i] && i > 0) {
      inputRefs.current[i - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (pasted.length === 6) {
      const next = pasted.split("");
      setDigits(next);
      handleSubmit(pasted);
    }
  };

  const handleSubmit = async (code: string) => {
    setVerifying(true);
    setError(null);
    const ok = await onVerify(code);
    setVerifying(false);
    if (ok) {
      setStep(nextStep);
    } else {
      setError("Invalid code. Please try again.");
      setDigits(["", "", "", "", "", ""]);
      inputRefs.current[0]?.focus();
    }
  };

  const handleResend = async () => {
    await onResend();
    setCooldown(60);
    setDigits(["", "", "", "", "", ""]);
    setError(null);
  };

  const label = type === "phone" ? "phone number" : "email address";

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center gap-2">
        <div className={`w-12 h-12 ${type === "phone" ? "bg-blue-600/20" : "bg-purple-600/20"} rounded-xl flex items-center justify-center text-2xl`}>
          {type === "phone" ? "📱" : "✉️"}
        </div>
        <h2 className="text-xl font-bold text-white">Enter verification code</h2>
        <p className="text-zinc-400 text-sm text-center">
          Enter the 6-digit code sent to {target || `your ${label}`}
        </p>
      </div>

      {/* 6 digit boxes */}
      <div className="flex gap-2 justify-center" onPaste={handlePaste}>
        {digits.map((d, i) => (
          <input
            key={i}
            ref={(el) => { inputRefs.current[i] = el; }}
            maxLength={1}
            value={d}
            onChange={(e) => handleDigit(i, e.target.value)}
            onKeyDown={(e) => handleKeyDown(i, e)}
            autoFocus={i === 0}
            className="w-12 h-14 text-center text-xl font-bold bg-[#0f0f0f] border border-zinc-700 rounded-xl text-white focus:border-blue-500 outline-none"
          />
        ))}
      </div>

      {error && (
        <p className="text-red-400 text-sm text-center">{error}</p>
      )}

      {verifying && (
        <div className="flex justify-center">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
        </div>
      )}

      {/* Resend with cooldown */}
      <p className="text-center text-sm text-zinc-500">
        {cooldown > 0 ? (
          `Resend code in ${cooldown}s`
        ) : (
          <button onClick={handleResend} className="text-blue-400 hover:underline">
            Resend code
          </button>
        )}
      </p>
    </div>
  );
}

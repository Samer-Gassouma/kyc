"use client";

import { useState } from "react";
import { useKYCStore } from "@/store/kycStore";

export default function EmailInputScreen() {
  const { email, setEmail, sendEmailOTP, setStep } = useKYCStore();
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    setSending(true);
    const ok = await sendEmailOTP();
    setSending(false);
    if (ok) setStep("EMAIL_OTP");
  };

  const isValid = email.includes("@") && email.includes(".") && email.length > 5;

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center gap-2">
        <div className="w-12 h-12 bg-purple-600/20 rounded-xl flex items-center justify-center text-2xl">
          ✉️
        </div>
        <h2 className="text-xl font-bold text-white">Email verification</h2>
        <p className="text-zinc-400 text-sm text-center">
          Enter your email address to receive a verification code
        </p>
      </div>

      <input
        type="email"
        placeholder="you@example.com"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full bg-[#0f0f0f] border border-zinc-700 rounded-xl px-4 py-3 text-white placeholder:text-zinc-600 outline-none focus:border-blue-500"
        autoFocus
      />

      <div className="space-y-3">
        <button
          onClick={handleSend}
          disabled={!isValid || sending}
          className={`w-full py-4 rounded-full font-semibold text-base ${
            isValid && !sending
              ? "bg-white text-black"
              : "bg-zinc-800 text-zinc-600 cursor-not-allowed"
          }`}
        >
          {sending ? "Sending..." : "Send verification code"}
        </button>
      </div>
    </div>
  );
}

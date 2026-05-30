"use client";

import { useState } from "react";
import { useKYCStore } from "@/store/kycStore";

export default function PhoneInputScreen() {
  const { phoneNumber, setPhoneNumber, sendPhoneOTP, setStep } = useKYCStore();
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    setSending(true);
    const ok = await sendPhoneOTP();
    setSending(false);
    if (ok) setStep("PHONE_OTP");
  };

  const isValid = phoneNumber.replace(/\s/g, "").length >= 8;

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center gap-2">
        <div className="w-12 h-12 bg-blue-600/20 rounded-xl flex items-center justify-center text-2xl">
          📱
        </div>
        <h2 className="text-xl font-bold text-white">Phone verification</h2>
        <p className="text-zinc-400 text-sm text-center">
          Enter your phone number to receive a verification code
        </p>
      </div>

      <div className="flex gap-2">
        <select className="bg-[#0f0f0f] border border-zinc-700 rounded-xl px-3 py-3 text-white appearance-none">
          <option value="+216">🇹🇳 +216</option>
        </select>
        <input
          type="tel"
          placeholder="53 249 239"
          value={phoneNumber}
          onChange={(e) => setPhoneNumber(e.target.value)}
          className="flex-1 bg-[#0f0f0f] border border-zinc-700 rounded-xl px-4 py-3 text-white placeholder:text-zinc-600 outline-none focus:border-blue-500"
        />
      </div>

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
        <button className="w-full py-4 rounded-full border border-zinc-600 text-white text-base">
          Continue on phone
        </button>
      </div>
    </div>
  );
}

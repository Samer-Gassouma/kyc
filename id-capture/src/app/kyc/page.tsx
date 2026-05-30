"use client";

import { useKYCStore, KYCStep } from "@/store/kycStore";
import IntroScreen from "@/components/kyc/IntroScreen";
import LivenessScreen from "@/components/kyc/LivenessScreen";
import PhoneInputScreen from "@/components/kyc/PhoneInputScreen";
import OtpScreen from "@/components/kyc/OtpScreen";
import DocumentTypeScreen from "@/components/kyc/DocumentTypeScreen";
import DocumentCaptureScreen from "@/components/kyc/DocumentCaptureScreen";
import DocumentConfirmScreen from "@/components/kyc/DocumentConfirmScreen";
import FaceScanScreen from "@/components/kyc/FaceScanScreen";
import EmailInputScreen from "@/components/kyc/EmailInputScreen";
import VerifyingScreen from "@/components/kyc/VerifyingScreen";
import ApprovedScreen from "@/components/kyc/ApprovedScreen";
import RejectedScreen from "@/components/kyc/RejectedScreen";

// ── Step counter mapping ─────────────────────────────────────────────

const STEP_NUMBER: Partial<Record<KYCStep, number>> = {
  LIVENESS: 1,
  PHONE_INPUT: 2,
  PHONE_OTP: 2,
  DOCUMENT_TYPE: 3,
  DOCUMENT_FRONT: 4,
  DOCUMENT_FRONT_CONFIRM: 4,
  DOCUMENT_BACK: 4,
  DOCUMENT_BACK_CONFIRM: 4,
  FACE_SCAN: 5,
  EMAIL_INPUT: 6,
  EMAIL_OTP: 6,
};

const TOTAL_MAIN_STEPS = 6;

const HIDE_HEADER: KYCStep[] = [
  "INTRO",
  "VERIFYING",
  "APPROVED",
  "REJECTED",
];

export default function KYCPage() {
  const {
    step,
    goBack,
    phoneNumber,
    verifyPhoneOTP,
    sendPhoneOTP,
    email,
    verifyEmailOTP,
    sendEmailOTP,
  } = useKYCStore();

  const showHeader = !HIDE_HEADER.includes(step);
  const stepNum = STEP_NUMBER[step];

  const renderStep = () => {
    switch (step) {
      case "INTRO":
        return <IntroScreen />;

      case "LIVENESS":
        return <LivenessScreen />;

      case "PHONE_INPUT":
        return <PhoneInputScreen />;

      case "PHONE_OTP":
        return (
          <OtpScreen
            type="phone"
            target={phoneNumber}
            onVerify={verifyPhoneOTP}
            onResend={sendPhoneOTP}
            nextStep="DOCUMENT_TYPE"
          />
        );

      case "DOCUMENT_TYPE":
        return <DocumentTypeScreen />;

      case "DOCUMENT_FRONT":
        return <DocumentCaptureScreen side="front" />;

      case "DOCUMENT_FRONT_CONFIRM":
        return <DocumentConfirmScreen side="front" />;

      case "DOCUMENT_BACK":
        return <DocumentCaptureScreen side="back" />;

      case "DOCUMENT_BACK_CONFIRM":
        return <DocumentConfirmScreen side="back" />;

      case "FACE_SCAN":
        return <FaceScanScreen />;

      case "EMAIL_INPUT":
        return <EmailInputScreen />;

      case "EMAIL_OTP":
        return (
          <OtpScreen
            type="email"
            target={email}
            onVerify={verifyEmailOTP}
            onResend={sendEmailOTP}
            nextStep="VERIFYING"
          />
        );

      case "VERIFYING":
        return <VerifyingScreen />;

      case "APPROVED":
        return <ApprovedScreen />;

      case "REJECTED":
        return <RejectedScreen />;

      default:
        return <IntroScreen />;
    }
  };

  return (
    <div className="min-h-screen bg-[#0f0f0f] flex flex-col items-center justify-start">
      {/* Header — shown on all steps except INTRO, VERIFYING, APPROVED, REJECTED */}
      {showHeader && (
        <header className="w-full max-w-lg flex items-center justify-between px-4 py-4">
          <button
            onClick={goBack}
            className="w-9 h-9 rounded-full bg-[#1a1a1a] flex items-center justify-center text-zinc-400 hover:text-white transition-colors"
          >
            ←
          </button>
          <span className="font-semibold text-white text-sm">
            Step{" "}
            <span className="bg-[#1a1a1a] px-2 py-1 rounded-full">
              {stepNum}/{TOTAL_MAIN_STEPS}
            </span>
          </span>
          <button className="px-3 py-1 rounded-full border border-zinc-700 text-sm text-zinc-300">
            🌐 En
          </button>
        </header>
      )}

      {/* Step content */}
      <main className="w-full max-w-lg px-4 flex-1">
        <div className="bg-[#1a1a1a] rounded-2xl p-6">
          {renderStep()}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 text-zinc-600 text-sm">
        Powered by ◈ TerrainTel
      </footer>
    </div>
  );
}

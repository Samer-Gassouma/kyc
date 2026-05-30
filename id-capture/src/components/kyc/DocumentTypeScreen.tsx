"use client";

import { useKYCStore } from "@/store/kycStore";
import { API_BASE } from "@/lib/apiBase";

const DOC_TYPES = ["ID card", "Passport", "Driver's license", "Residence permit"];

export default function DocumentTypeScreen() {
  const {
    sessionId,
    documentCountry,
    documentType,
    setDocumentCountry,
    setDocumentType,
    setStep,
  } = useKYCStore();

  const handleContinue = async () => {
    // Update document type on backend
    const fd = new FormData();
    fd.append("country", documentCountry);
    fd.append("document_type", documentType);
    try {
      await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/document-type`,
        { method: "PATCH", body: fd }
      );
    } catch {}

    // Skip to face scan for passport (single side), otherwise capture front
    if (documentType === "Passport") {
      setStep("DOCUMENT_FRONT"); // Passport front only, then skip back
    }
    setStep("DOCUMENT_FRONT");
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-white">
        Select type and issuing country of your identity document
      </h2>

      <div className="space-y-2">
        <label className="text-sm text-zinc-400">Issuing country *</label>
        <select
          value={documentCountry}
          onChange={(e) => setDocumentCountry(e.target.value)}
          className="w-full bg-[#0f0f0f] border border-zinc-700 rounded-xl px-4 py-3 text-white outline-none focus:border-blue-500 appearance-none"
        >
          <option value="TN">🇹🇳 Tunisia</option>
        </select>
      </div>

      <div className="space-y-2">
        <label className="text-sm text-zinc-400">Document type *</label>
        {DOC_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setDocumentType(type)}
            className={`w-full flex items-center justify-between px-4 py-3 rounded-xl border text-left ${
              documentType === type
                ? "border-blue-500 bg-blue-500/10"
                : "border-zinc-700"
            }`}
          >
            <span className="text-white">{type}</span>
            <div
              className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                documentType === type
                  ? "border-blue-500 bg-blue-500"
                  : "border-zinc-600"
              }`}
            >
              {documentType === type && (
                <div className="w-2 h-2 rounded-full bg-white" />
              )}
            </div>
          </button>
        ))}
      </div>

      <button
        onClick={handleContinue}
        className="w-full py-4 rounded-full bg-white text-black font-semibold text-base"
      >
        Continue
      </button>
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";
import { useKYCStore } from "@/store/kycStore";

interface DocumentConfirmScreenProps {
  side: "front" | "back";
}

export default function DocumentConfirmScreen({ side }: DocumentConfirmScreenProps) {
  const {
    documentFrontDataURL,
    documentBackDataURL,
    documentExtracted,
    uploadDocumentFront,
    uploadDocumentBack,
    setDocumentFrontDataURL,
    setDocumentBackDataURL,
    setDocumentFrontBlob,
    setDocumentBackBlob,
    setDocumentExtracted,
    documentType,
    setStep,
  } = useKYCStore();

  const [uploading, setUploading] = useState(false);
  const dataURL = side === "front" ? documentFrontDataURL : documentBackDataURL;

  const handleConfirm = async () => {
    setUploading(true);
    let extracted;
    if (side === "front") {
      extracted = await uploadDocumentFront();
    } else {
      extracted = await uploadDocumentBack();
    }
    setUploading(false);

    if (side === "front") {
      setDocumentExtracted(extracted);
      // Skip back if passport
      if (documentType === "Passport") {
        setStep("FACE_SCAN");
      } else {
        setStep("DOCUMENT_BACK");
      }
    } else {
      setStep("FACE_SCAN");
    }
  };

  const handleRetake = () => {
    if (side === "front") {
      setDocumentFrontDataURL(null);
      setDocumentFrontBlob(null);
      setStep("DOCUMENT_FRONT");
    } else {
      setDocumentBackDataURL(null);
      setDocumentBackBlob(null);
      setStep("DOCUMENT_BACK");
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Check your photo</h2>
      <p className="text-zinc-400 text-sm">
        Make sure all text is clearly visible
      </p>

      <p className="text-white font-medium text-sm bg-[#0f0f0f] rounded-lg px-3 py-2 text-center">
        {side === "front" ? "📷 Front side" : "📷 Back side"}
      </p>

      <div className="rounded-xl overflow-hidden border border-zinc-700">
        {dataURL && (
          <img src={dataURL} alt={`Document ${side}`} className="w-full" />
        )}
      </div>

      {uploading && (
        <div className="flex justify-center items-center gap-2 py-2">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
          <span className="text-zinc-400 text-sm">Processing document...</span>
        </div>
      )}

      {documentExtracted && side === "front" && !uploading && (
        <div className="bg-[#0f0f0f] rounded-xl p-4 space-y-2">
          <p className="text-zinc-400 text-xs uppercase tracking-wider">
            Extracted data
          </p>
          {Object.entries(documentExtracted)
            .filter(([, v]) => v && typeof v === "string")
            .slice(0, 10)
            .map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-zinc-500 text-sm capitalize">
                  {k.replace(/_/g, " ")}
                </span>
                <span className="text-white text-sm font-medium">{String(v)}</span>
              </div>
            ))}
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={handleRetake}
          disabled={uploading}
          className="flex-1 py-4 rounded-full border border-zinc-600 text-white text-base"
        >
          ↩ Retake
        </button>
        <button
          onClick={handleConfirm}
          disabled={uploading}
          className={`flex-1 py-4 rounded-full font-semibold text-base ${
            uploading ? "bg-zinc-800 text-zinc-600 cursor-not-allowed" : "bg-white text-black"
          }`}
        >
          {uploading ? "Uploading..." : "Use this photo →"}
        </button>
      </div>
    </div>
  );
}

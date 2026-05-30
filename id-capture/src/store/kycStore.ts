"use client";

import { create } from "zustand";
import { API_BASE } from "@/lib/apiBase";

// ── Types ────────────────────────────────────────────────────────────

export type KYCStep =
  | "INTRO"
  | "LIVENESS"
  | "PHONE_INPUT"
  | "PHONE_OTP"
  | "DOCUMENT_TYPE"
  | "DOCUMENT_FRONT"
  | "DOCUMENT_FRONT_CONFIRM"
  | "DOCUMENT_BACK"
  | "DOCUMENT_BACK_CONFIRM"
  | "FACE_SCAN"
  | "EMAIL_INPUT"
  | "EMAIL_OTP"
  | "VERIFYING"
  | "APPROVED"
  | "REJECTED";

interface KYCState {
  sessionId: string | null;
  step: KYCStep;
  stepHistory: KYCStep[];

  // Liveness
  livenessScore: number;

  // Phone
  phoneNumber: string;
  phoneVerified: boolean;

  // Document
  documentCountry: string;
  documentType: string;
  documentFrontDataURL: string | null;
  documentBackDataURL: string | null;
  documentFrontBlob: Blob | null;
  documentBackBlob: Blob | null;
  documentExtracted: Record<string, string> | null;

  // Face
  faceUserId: string | null;
  faceMatchScore: number;
  faceMatchPassed: boolean;

  // Email
  email: string;
  emailVerified: boolean;

  // Result
  status: "approved" | "rejected" | null;
  rejectionReasons: string[];
  error: string | null;

  // ── Actions ──────────────────────────────────────────────────────
  setStep: (step: KYCStep) => void;
  goBack: () => void;
  createSession: () => Promise<void>;

  setLivenessScore: (score: number) => void;
  setPhoneNumber: (phone: string) => void;
  setPhoneVerified: (v: boolean) => void;
  setDocumentCountry: (c: string) => void;
  setDocumentType: (t: string) => void;
  setDocumentFrontDataURL: (url: string | null) => void;
  setDocumentBackDataURL: (url: string | null) => void;
  setDocumentFrontBlob: (blob: Blob | null) => void;
  setDocumentBackBlob: (blob: Blob | null) => void;
  setDocumentExtracted: (data: Record<string, string> | null) => void;
  setFaceUserId: (id: string | null) => void;
  setFaceMatchScore: (score: number) => void;
  setFaceMatchPassed: (p: boolean) => void;
  setEmail: (e: string) => void;
  setEmailVerified: (v: boolean) => void;
  setStatus: (s: "approved" | "rejected" | null) => void;
  setRejectionReasons: (r: string[]) => void;
  setError: (e: string | null) => void;

  // API helpers
  sendPhoneOTP: () => Promise<boolean>;
  verifyPhoneOTP: (otp: string) => Promise<boolean>;
  sendEmailOTP: () => Promise<boolean>;
  verifyEmailOTP: (otp: string) => Promise<boolean>;
  uploadDocumentFront: () => Promise<Record<string, string> | null>;
  uploadDocumentBack: () => Promise<Record<string, string> | null>;
  submitFace: (
    blob: Blob,
    livenessScore: number,
    landmarks3d?: string,
    qualityScore?: number
  ) => Promise<boolean>;
  submitSession: () => Promise<void>;
  pollStatus: () => Promise<void>;

  reset: () => void;
}

const initialState = {
  sessionId: null,
  step: "INTRO" as KYCStep,
  stepHistory: [] as KYCStep[],
  livenessScore: 0,
  phoneNumber: "",
  phoneVerified: false,
  documentCountry: "TN",
  documentType: "ID card",
  documentFrontDataURL: null,
  documentBackDataURL: null,
  documentFrontBlob: null,
  documentBackBlob: null,
  documentExtracted: null,
  faceUserId: null,
  faceMatchScore: 0,
  faceMatchPassed: false,
  email: "",
  emailVerified: false,
  status: null,
  rejectionReasons: [],
  error: null,
};

export const useKYCStore = create<KYCState>((set, get) => ({
  ...initialState,

  setStep: (step: KYCStep) =>
    set((s) => ({
      step,
      stepHistory: [...s.stepHistory, s.step],
    })),

  goBack: () =>
    set((s) => {
      const history = [...s.stepHistory];
      const prev = history.pop();
      return {
        step: prev || "INTRO",
        stepHistory: history,
      };
    }),

  createSession: async () => {
    try {
      const res = await fetch(`${API_BASE}/api/kyc/session`, {
        method: "POST",
      });
      const data = await res.json();
      set({ sessionId: data.session_id });
    } catch (e) {
      set({ error: "Failed to create session" });
    }
  },

  setLivenessScore: (score) => set({ livenessScore: score }),
  setPhoneNumber: (phone) => set({ phoneNumber: phone }),
  setPhoneVerified: (v) => set({ phoneVerified: v }),
  setDocumentCountry: (c) => set({ documentCountry: c }),
  setDocumentType: (t) => set({ documentType: t }),
  setDocumentFrontDataURL: (url) => set({ documentFrontDataURL: url }),
  setDocumentBackDataURL: (url) => set({ documentBackDataURL: url }),
  setDocumentFrontBlob: (blob) => set({ documentFrontBlob: blob }),
  setDocumentBackBlob: (blob) => set({ documentBackBlob: blob }),
  setDocumentExtracted: (data) => set({ documentExtracted: data }),
  setFaceUserId: (id) => set({ faceUserId: id }),
  setFaceMatchScore: (score) => set({ faceMatchScore: score }),
  setFaceMatchPassed: (p) => set({ faceMatchPassed: p }),
  setEmail: (e) => set({ email: e }),
  setEmailVerified: (v) => set({ emailVerified: v }),
  setStatus: (s) => set({ status: s }),
  setRejectionReasons: (r) => set({ rejectionReasons: r }),
  setError: (e) => set({ error: e }),

  // ── API helpers ──────────────────────────────────────────────────

  sendPhoneOTP: async () => {
    const { sessionId, phoneNumber } = get();
    if (!sessionId) return false;
    try {
      const fd = new FormData();
      fd.append("phone_number", phoneNumber);
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/phone/send`,
        { method: "POST", body: fd }
      );
      return res.ok;
    } catch {
      return false;
    }
  },

  verifyPhoneOTP: async (otp: string) => {
    const { sessionId } = get();
    if (!sessionId) return false;
    try {
      const fd = new FormData();
      fd.append("otp", otp);
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/phone/verify`,
        { method: "POST", body: fd }
      );
      if (res.ok) {
        set({ phoneVerified: true });
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },

  sendEmailOTP: async () => {
    const { sessionId, email } = get();
    if (!sessionId) return false;
    try {
      const fd = new FormData();
      fd.append("email", email);
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/email/send`,
        { method: "POST", body: fd }
      );
      return res.ok;
    } catch {
      return false;
    }
  },

  verifyEmailOTP: async (otp: string) => {
    const { sessionId } = get();
    if (!sessionId) return false;
    try {
      const fd = new FormData();
      fd.append("otp", otp);
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/email/verify`,
        { method: "POST", body: fd }
      );
      if (res.ok) {
        set({ emailVerified: true });
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },

  uploadDocumentFront: async () => {
    const { sessionId, documentFrontBlob } = get();
    if (!sessionId || !documentFrontBlob) return null;
    try {
      const fd = new FormData();
      fd.append("image", documentFrontBlob, "front.jpg");
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/document/front`,
        { method: "POST", body: fd }
      );
      if (res.ok) {
        const data = await res.json();
        set({ documentExtracted: data.extracted });
        return data.extracted;
      }
      return null;
    } catch {
      return null;
    }
  },

  uploadDocumentBack: async () => {
    const { sessionId, documentBackBlob } = get();
    if (!sessionId || !documentBackBlob) return null;
    try {
      const fd = new FormData();
      fd.append("image", documentBackBlob, "back.jpg");
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/document/back`,
        { method: "POST", body: fd }
      );
      if (res.ok) {
        const data = await res.json();
        return data.extracted;
      }
      return null;
    } catch {
      return null;
    }
  },

  submitFace: async (
    blob: Blob,
    livenessScore: number,
    landmarks3d?: string,
    qualityScore?: number
  ) => {
    const { sessionId } = get();
    if (!sessionId) return false;
    try {
      const fd = new FormData();
      fd.append("live_image", blob, "face.jpg");
      fd.append("liveness_score", String(livenessScore));
      if (landmarks3d) fd.append("landmarks_3d", landmarks3d);
      if (qualityScore !== undefined)
        fd.append("quality_score", String(qualityScore));

      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/face`,
        { method: "POST", body: fd }
      );
      if (res.ok) {
        const data = await res.json();
        set({
          faceUserId: data.user_id,
          faceMatchScore: data.similarity,
          faceMatchPassed: data.face_match,
        });
        return data.face_match;
      }
      const err = await res.json().catch(() => ({ detail: "Face enrollment failed" }));
      set({ error: err.detail || "Face enrollment failed" });
      return false;
    } catch (e) {
      set({ error: "Face enrollment failed" });
      return false;
    }
  },

  submitSession: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/submit`,
        { method: "POST" }
      );
      const data = await res.json();
      set({
        status: data.status,
        rejectionReasons: data.rejection_reasons || [],
      });
    } catch (e) {
      set({ error: "Failed to submit verification" });
    }
  },

  pollStatus: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      const res = await fetch(
        `${API_BASE}/api/kyc/session/${sessionId}/status`
      );
      const data = await res.json();
      if (data.status === "approved" || data.status === "rejected") {
        set({
          status: data.status,
          rejectionReasons: data.rejection_reasons || [],
          step: data.status === "approved" ? "APPROVED" : "REJECTED",
        });
      }
    } catch {
      // keep polling
    }
  },

  reset: () => set(initialState),
}));

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "KYC Verification",
  description: "Secure ID capture and verification",
};

export default function KYCLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}

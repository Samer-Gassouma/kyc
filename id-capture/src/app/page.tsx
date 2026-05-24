import Link from "next/link";
import { Shield, ScanText } from "lucide-react";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-950 px-4 text-zinc-100">
      <div className="mb-8 flex items-center gap-3">
        <Shield className="h-8 w-8 text-blue-400" />
        <h1 className="text-2xl font-bold">KYC Verification</h1>
      </div>

      <div className="grid w-full max-w-md gap-4">
        <Link
          href="/kyc"
          className="flex items-center gap-4 rounded-xl bg-zinc-900 p-5 transition-colors hover:bg-zinc-800"
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-500/10">
            <Shield className="h-6 w-6 text-blue-400" />
          </div>
          <div>
            <p className="font-medium text-zinc-100">Full KYC Flow</p>
            <p className="text-sm text-zinc-500">Capture front & back, extract CIN data</p>
          </div>
        </Link>

        <Link
          href="/extract"
          className="flex items-center gap-4 rounded-xl bg-zinc-900 p-5 transition-colors hover:bg-zinc-800"
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-500/10">
            <ScanText className="h-6 w-6 text-green-400" />
          </div>
          <div>
            <p className="font-medium text-zinc-100">Extract CIN Data</p>
            <p className="text-sm text-zinc-500">Upload front & back, get JSON extraction</p>
          </div>
        </Link>

      </div>
    </div>
  );
}

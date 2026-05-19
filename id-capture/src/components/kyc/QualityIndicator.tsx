"use client";

import clsx from "clsx";
import {
  AlertTriangle,
  Eye,
  Focus,
  RotateCcw,
  Sun,
  Zap,
} from "lucide-react";

interface QualityIndicatorProps {
  issues: string[];
  localQuality?: {
    blur_score: number;
    glare_score: number;
    orientation: string;
    brightness: string;
    local_pass: boolean;
  } | null;
}

const ISSUE_ICONS: Record<string, React.ReactNode> = {
  blur: <Eye className="h-3.5 w-3.5" />,
  glare: <Zap className="h-3.5 w-3.5" />,
  center: <Focus className="h-3.5 w-3.5" />,
  straight: <RotateCcw className="h-3.5 w-3.5" />,
  lighting: <Sun className="h-3.5 w-3.5" />,
  dark: <Sun className="h-3.5 w-3.5" />,
  bright: <Sun className="h-3.5 w-3.5" />,
};

function getIconForIssue(issue: string): React.ReactNode {
  const lower = issue.toLowerCase();
  for (const [key, icon] of Object.entries(ISSUE_ICONS)) {
    if (lower.includes(key)) return icon;
  }
  return <AlertTriangle className="h-3.5 w-3.5" />;
}

export default function QualityIndicator({
  issues,
  localQuality,
}: QualityIndicatorProps) {
  if (issues.length === 0 && (!localQuality || localQuality.local_pass)) {
    return null;
  }

  const allIssues = [...issues];

  // Add local quality issues if they exist
  if (localQuality && !localQuality.local_pass) {
    if (localQuality.blur_score <= 0.3) {
      allIssues.push("Image is blurry");
    }
    if (localQuality.glare_score > 0.15) {
      allIssues.push("Glare detected");
    }
    if (localQuality.brightness === "too_dark") {
      allIssues.push("Too dark");
    }
    if (localQuality.brightness === "too_bright") {
      allIssues.push("Too bright");
    }
    if (localQuality.orientation !== "straight") {
      allIssues.push("Card is tilted");
    }
  }

  // Deduplicate and cap at 3 most important issues
  const unique = [...new Set(allIssues)].slice(0, 3);

  return (
    <div className="flex flex-wrap justify-center gap-1.5 px-2">
      {unique.map((issue, idx) => (
        <div
          key={idx}
          className={clsx(
            "flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-medium",
            "bg-yellow-500/20 text-yellow-300 backdrop-blur-sm border border-yellow-500/30"
          )}
        >
          {getIconForIssue(issue)}
          <span>{issue}</span>
        </div>
      ))}
    </div>
  );
}

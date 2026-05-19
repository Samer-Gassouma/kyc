"use client";

import { Check } from "lucide-react";
import clsx from "clsx";

export type KYCStep = "front_id" | "back_id" | "liveness";

interface StepProgressProps {
  currentStep: KYCStep;
  completedSteps: KYCStep[];
}

const STEPS: { key: KYCStep; label: string }[] = [
  { key: "front_id", label: "Front ID" },
  { key: "back_id", label: "Back ID" },
  { key: "liveness", label: "Liveness" },
];

export default function StepProgress({
  currentStep,
  completedSteps,
}: StepProgressProps) {
  return (
    <div className="flex items-center justify-center gap-2 py-4">
      {STEPS.map((step, idx) => {
        const isCompleted = completedSteps.includes(step.key);
        const isCurrent = currentStep === step.key;

        return (
          <div key={step.key} className="flex items-center gap-2">
            {/* Step circle */}
            <div className="flex flex-col items-center gap-1">
              <div
                className={clsx(
                  "flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold transition-all",
                  {
                    "bg-green-500 text-white": isCompleted,
                    "bg-blue-600 text-white ring-2 ring-blue-300": isCurrent && !isCompleted,
                    "bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400":
                      !isCurrent && !isCompleted,
                  }
                )}
              >
                {isCompleted ? (
                  <Check className="h-4 w-4" />
                ) : (
                  idx + 1
                )}
              </div>
              <span
                className={clsx("text-[10px] font-medium", {
                  "text-green-600 dark:text-green-400": isCompleted,
                  "text-blue-600 dark:text-blue-400": isCurrent && !isCompleted,
                  "text-zinc-400": !isCurrent && !isCompleted,
                })}
              >
                {step.label}
              </span>
            </div>

            {/* Connector line */}
            {idx < STEPS.length - 1 && (
              <div
                className={clsx("mb-4 h-0.5 w-10 transition-all", {
                  "bg-green-500": isCompleted,
                  "bg-zinc-200 dark:bg-zinc-700": !isCompleted,
                })}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

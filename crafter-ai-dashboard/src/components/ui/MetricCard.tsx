import React from "react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  trend?: "up" | "down" | "neutral";
  icon?: React.ReactNode;
  className?: string;
}

export function MetricCard({
  label,
  value,
  subValue,
  trend,
  icon,
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "p-4 rounded-xl bg-surface border border-border flex flex-col justify-between hover:border-neutral-700/60 transition-colors",
        className
      )}
    >
      <div className="flex items-center justify-between text-neutral-400 mb-2">
        <span className="text-xs font-mono uppercase tracking-wider">{label}</span>
        {icon && <div className="text-neutral-500">{icon}</div>}
      </div>
      <div>
        <div className="text-2xl font-mono font-semibold tracking-tight text-white">
          {value}
        </div>
        {subValue && (
          <div
            className={cn(
              "text-xs font-mono mt-1",
              trend === "up" && "text-emerald-400",
              trend === "down" && "text-rose-400",
              !trend && "text-neutral-500"
            )}
          >
            {subValue}
          </div>
        )}
      </div>
    </div>
  );
}

import React from "react";
import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  status: "idle" | "starting" | "training" | "stopped" | "error";
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const configs = {
    training: {
      label: "TRAINING",
      color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      dot: "bg-emerald-400 animate-pulse",
    },
    starting: {
      label: "STARTING",
      color: "bg-amber-500/10 text-amber-400 border-amber-500/20",
      dot: "bg-amber-400 animate-ping",
    },
    stopped: {
      label: "STOPPED",
      color: "bg-neutral-800 text-neutral-400 border-neutral-700",
      dot: "bg-neutral-500",
    },
    idle: {
      label: "IDLE",
      color: "bg-neutral-900 text-neutral-400 border-neutral-800",
      dot: "bg-neutral-500",
    },
    error: {
      label: "ERROR",
      color: "bg-rose-500/10 text-rose-400 border-rose-500/20",
      dot: "bg-rose-400",
    },
  };

  const config = configs[status] || configs.idle;

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-mono font-medium border tracking-wider",
        config.color,
        className
      )}
    >
      <span className={cn("w-1.5 h-1.5 rounded-full", config.dot)} />
      {config.label}
    </div>
  );
}

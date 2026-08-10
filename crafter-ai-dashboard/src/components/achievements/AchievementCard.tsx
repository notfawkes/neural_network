"use client";

import React from "react";
import { AchievementItem } from "@/types";

interface AchievementCardProps {
  achievement: AchievementItem;
}

export function AchievementCard({ achievement }: AchievementCardProps) {
  const percent = (achievement.success_rate * 100).toFixed(1);

  return (
    <div
      className={`p-5 rounded-xl border flex flex-col justify-between transition-all ${
        achievement.unlocked
          ? "bg-surface border-neutral-700/80 shadow-sm hover:border-primary-500/50"
          : "bg-[#0f0f0f] border-border-subtle opacity-70 hover:opacity-100"
      }`}
    >
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-2xl p-2 rounded-lg bg-neutral-900 border border-border">
            {achievement.icon}
          </span>
          <span
            className={`text-[10px] font-mono px-2 py-0.5 rounded-full uppercase tracking-wider ${
              achievement.unlocked
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                : "bg-neutral-800 text-neutral-500"
            }`}
          >
            {achievement.category}
          </span>
        </div>

        <h4 className="text-sm font-semibold text-white tracking-tight">
          {achievement.title}
        </h4>
        <span className="text-[11px] font-mono text-neutral-500 block mt-0.5">
          {achievement.name}
        </span>
      </div>

      <div className="mt-4 pt-4 border-t border-border-subtle">
        <div className="flex justify-between items-center text-xs font-mono mb-1.5">
          <span className="text-neutral-500">SUCCESS RATE</span>
          <span
            className={`font-semibold ${
              achievement.unlocked ? "text-emerald-400" : "text-neutral-500"
            }`}
          >
            {percent}%
          </span>
        </div>

        <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              achievement.unlocked ? "bg-emerald-500" : "bg-neutral-700"
            }`}
            style={{ width: `${Math.min(100, achievement.success_rate * 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}

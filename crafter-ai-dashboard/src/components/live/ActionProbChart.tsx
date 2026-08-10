"use client";

import React from "react";
import { CRAFTER_ACTIONS } from "@/lib/utils";

interface ActionProbChartProps {
  probs?: number[];
  selectedAction?: string;
}

export function ActionProbChart({ probs, selectedAction }: ActionProbChartProps) {
  const safeProbs = probs && probs.length === 17 ? probs : new Array(17).fill(0.058);

  return (
    <div className="bg-surface rounded-xl border border-border p-5 flex flex-col justify-between">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-white">Action Probability Distribution</h3>
          <p className="text-xs text-neutral-500 font-mono">17-Action Softmax Layer</p>
        </div>
        {selectedAction && (
          <div className="px-2 py-0.5 rounded bg-primary-500/10 border border-primary-500/20 text-primary-400 text-xs font-mono font-medium">
            SELECTED: {selectedAction}
          </div>
        )}
      </div>

      {/* Bar Chart Area */}
      <div className="grid grid-cols-17 gap-1 h-44 items-end pt-4 border-b border-border-subtle">
        {safeProbs.map((prob, i) => {
          const name = CRAFTER_ACTIONS[i];
          const isSelected = name === selectedAction;
          const heightPercent = Math.min(100, Math.max(4, Math.round(prob * 100)));

          return (
            <div key={name} className="flex flex-col items-center h-full justify-end group">
              {/* Tooltip on hover */}
              <div className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px] font-mono text-white mb-1">
                {(prob * 100).toFixed(0)}%
              </div>

              {/* Bar */}
              <div
                className={`w-full rounded-t transition-all duration-100 ${
                  isSelected
                    ? "bg-primary-500 shadow-lg shadow-primary-500/30"
                    : "bg-neutral-800 hover:bg-neutral-700"
                }`}
                style={{ height: `${heightPercent}%` }}
              />

              {/* Label */}
              <span
                className={`text-[9px] font-mono mt-2 truncate w-full text-center ${
                  isSelected ? "text-primary-400 font-bold" : "text-neutral-500"
                }`}
                title={name}
              >
                {name.slice(0, 3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

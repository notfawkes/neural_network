"use client";

import React from "react";
import { ArrowRight, Layers, Cpu, Compass, Shield } from "lucide-react";

export function ArchDiagram() {
  return (
    <div className="p-6 bg-surface rounded-xl border border-border">
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-white">Network Computational Graph</h3>
        <p className="text-xs text-neutral-500 font-mono mt-0.5">
          End-to-end forward flow from 64×64 visual observations to stochastic action distributions
        </p>
      </div>

      <div className="flex flex-col lg:flex-row items-center justify-between gap-4 py-6 overflow-x-auto">
        {/* Input */}
        <div className="p-4 rounded-xl bg-neutral-900 border border-border flex flex-col items-center min-w-[130px]">
          <span className="text-[10px] font-mono text-neutral-500 mb-1">INPUT</span>
          <span className="text-xs font-bold text-white">RGB Image</span>
          <span className="text-[11px] font-mono text-primary-400 mt-1">3 × 64 × 64</span>
        </div>

        <ArrowRight className="w-4 h-4 text-neutral-600 hidden lg:block shrink-0" />

        {/* CNN */}
        <div className="p-4 rounded-xl bg-neutral-900 border border-border flex flex-col items-center min-w-[160px] relative">
          <div className="flex items-center gap-1.5 text-xs font-bold text-white mb-1">
            <Layers className="w-3.5 h-3.5 text-blue-400" /> CrafterCNN
          </div>
          <span className="text-[10px] font-mono text-neutral-400">3× Conv2d + FC</span>
          <span className="text-[11px] font-mono text-blue-400 mt-1">512 Features</span>
          <div className="text-[9px] font-mono text-neutral-500 mt-2 text-center leading-tight">
            k3, s2, filters 32→64→64
          </div>
        </div>

        <ArrowRight className="w-4 h-4 text-neutral-600 hidden lg:block shrink-0" />

        {/* LSTM */}
        <div className="p-4 rounded-xl bg-neutral-900 border border-primary-500/30 flex flex-col items-center min-w-[150px] relative">
          <div className="flex items-center gap-1.5 text-xs font-bold text-white mb-1">
            <Cpu className="w-3.5 h-3.5 text-purple-400" /> LSTM Core
          </div>
          <span className="text-[10px] font-mono text-neutral-400">Temporal Memory</span>
          <span className="text-[11px] font-mono text-purple-400 mt-1">Hidden: 256</span>
          <div className="text-[9px] font-mono text-neutral-500 mt-2 text-center leading-tight">
            h, c reset on done
          </div>
        </div>

        <ArrowRight className="w-4 h-4 text-neutral-600 hidden lg:block shrink-0" />

        {/* Heads (Actor & Critic) */}
        <div className="flex flex-col gap-3 min-w-[160px]">
          {/* Actor */}
          <div className="p-3 rounded-lg bg-neutral-900 border border-border flex items-center justify-between">
            <div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-white">
                <Compass className="w-3.5 h-3.5 text-emerald-400" /> Actor Head
              </div>
              <span className="text-[10px] font-mono text-neutral-400">Categorical Logits</span>
            </div>
            <span className="text-xs font-mono font-bold text-emerald-400">17 Actions</span>
          </div>

          {/* Critic */}
          <div className="p-3 rounded-lg bg-neutral-900 border border-border flex items-center justify-between">
            <div>
              <div className="flex items-center gap-1.5 text-xs font-bold text-white">
                <Shield className="w-3.5 h-3.5 text-amber-400" /> Critic Head
              </div>
              <span className="text-[10px] font-mono text-neutral-400">Linear Value Head</span>
            </div>
            <span className="text-xs font-mono font-bold text-amber-400">Scalar V(s)</span>
          </div>
        </div>
      </div>
    </div>
  );
}


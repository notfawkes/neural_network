"use client";

import React, { useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Activity,
  Layers,
  Heart,
  Compass,
  Info,
} from "lucide-react";
import { LiveFrameData } from "@/types";

interface GameViewProps {
  data: LiveFrameData | null;
  isPaused: boolean;
  onTogglePause: () => void;
  onStepForward: () => void;
  onReplay: () => void;
  speed: number;
  onChangeSpeed: (speed: number) => void;
}

export function GameView({
  data,
  isPaused,
  onTogglePause,
  onStepForward,
  onReplay,
  speed,
  onChangeSpeed,
}: GameViewProps) {
  const [showHighRes, setShowHighRes] = useState(false);
  const [showPPOInfo, setShowPPOInfo] = useState(false);

  // Determine current active PPO cycle step
  const phase = data?.phase || "COLLECTING_ROLLOUT";
  const ppoStages = [
    { id: "COLLECTING_ROLLOUT", label: "COLLECT ROLLOUT", desc: "Agent steps in Crafter envs" },
    { id: "STORE_ROLLOUT", label: "STORE BUFFER", desc: "Batch stored into rollout buffer" },
    { id: "CALCULATING_ADVANTAGES", label: "GAE ADVANTAGES", desc: "GAE-λ advantage calculation" },
    { id: "PPO_UPDATE", label: "PPO UPDATE", desc: "Actor-Critic gradient updates" },
    { id: "POLICY_UPDATED", label: "UPDATED POLICY", desc: "Weights synced for next rollout" },
  ];

  const currentStageIndex = ppoStages.findIndex((s) => s.id === phase);
  const activeIndex = currentStageIndex >= 0 ? currentStageIndex : 0;

  return (
    <div className="flex flex-col p-6 bg-surface-card rounded-2xl border border-border shadow-xl">
      {/* Top Header: Episode, Step, Status */}
      <div className="w-full flex items-center justify-between mb-4 text-xs font-mono">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-neutral-500">EPISODE</span>
            <span className="text-white font-bold">{data?.episode_id ?? "--"}</span>
          </div>
          <span className="text-neutral-600">/</span>
          <div className="flex items-center gap-1.5">
            <span className="text-neutral-500">ENV STEP</span>
            <span className="text-white font-bold">{data?.step ? data.step.toLocaleString() : "--"}</span>
          </div>
          {data?.health !== undefined && (
            <>
              <span className="text-neutral-600">/</span>
              <div className="flex items-center gap-1 text-rose-400">
                <Heart className="w-3.5 h-3.5 fill-current" />
                <span className="font-bold">{data.health} / 9 HP</span>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              data?.is_alive ? "bg-emerald-400 animate-pulse" : "bg-neutral-600"
            }`}
          />
          <span className={data?.is_alive ? "text-emerald-400 font-semibold" : "text-neutral-400"}>
            {data?.is_alive ? "LIVE AGENT POV" : "TRAINING IDLE"}
          </span>
        </div>
      </div>

      {/* Primary Game Display (Enlarged Nearest-Neighbor Scaled 64x64) */}
      <div className="relative w-full aspect-square max-w-[420px] mx-auto rounded-xl overflow-hidden bg-black border-2 border-neutral-800 shadow-2xl flex items-center justify-center group">
        {data?.frame ? (
          <img
            src={`data:image/jpeg;base64,${data.frame}`}
            alt="Crafter Observation"
            className="w-full h-full object-contain"
            style={{
              imageRendering: showHighRes ? "auto" : "pixelated",
            }}
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-neutral-600">
            <Activity className="w-10 h-10 mb-2 stroke-1 opacity-50 animate-pulse" />
            <p className="text-xs font-mono">INITIALIZING CRAFTER OBSERVATIONS</p>
          </div>
        )}

        {/* Action Overlay Banner */}
        {data?.action && (
          <div className="absolute bottom-3 left-3 right-3 bg-black/85 backdrop-blur-md px-3.5 py-2 rounded-lg border border-white/10 flex items-center justify-between text-xs font-mono shadow-lg">
            <div className="flex items-center gap-2">
              <span className="text-neutral-400 text-[10px] uppercase">DECISION:</span>
              <span className="text-primary-400 font-bold uppercase tracking-wider">{data.action}</span>
            </div>
            {data?.reward !== undefined && (
              <span
                className={`font-mono font-bold ${
                  data.reward > 0
                    ? "text-emerald-400"
                    : data.reward < 0
                    ? "text-rose-400"
                    : "text-neutral-500"
                }`}
              >
                {data.reward > 0 ? `+${data.reward.toFixed(1)}` : data.reward.toFixed(1)} r
              </span>
            )}
          </div>
        )}

        {/* View mode pill (Pixelated 64x64 vs Smoothed) */}
        <div className="absolute top-3 left-3">
          <button
            onClick={() => setShowHighRes(!showHighRes)}
            className="px-2 py-1 rounded bg-black/70 backdrop-blur-sm border border-white/10 text-[10px] font-mono text-neutral-300 hover:text-white flex items-center gap-1.5 transition-colors"
            title="Toggle between exact 64x64 nearest-neighbour pixelation and filtered view"
          >
            <Layers className="w-3 h-3 text-primary-400" />
            {showHighRes ? "SMOOTHED" : "EXACT 64x64 PIXELS"}
          </button>
        </div>

        {/* Player Coordinate Pill */}
        {data?.player_pos && (
          <div className="absolute top-3 right-3 px-2 py-1 rounded bg-black/70 backdrop-blur-sm border border-white/10 text-[10px] font-mono text-neutral-300 flex items-center gap-1.5">
            <Compass className="w-3 h-3 text-emerald-400" />
            <span>POS: [{data.player_pos[0]}, {data.player_pos[1]}]</span>
          </div>
        )}

        {/* Pause Overlay */}
        {isPaused && (
          <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px] flex items-center justify-center">
            <div className="px-4 py-2 rounded-full bg-white/10 border border-white/20 text-white text-xs font-mono font-bold uppercase tracking-widest shadow-2xl">
              STREAM PAUSED
            </div>
          </div>
        )}
      </div>

      {/* Playback & Inspection Controls */}
      <div className="w-full flex items-center justify-between mt-5 pt-4 border-t border-border-subtle">
        <div className="flex items-center gap-2">
          {/* Play / Pause */}
          <button
            onClick={onTogglePause}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 transition-colors"
            title={isPaused ? "Resume Live Stream" : "Pause Live Stream"}
          >
            {isPaused ? <Play className="w-4 h-4 fill-current" /> : <Pause className="w-4 h-4 fill-current" />}
          </button>

          {/* Single Step */}
          <button
            onClick={onStepForward}
            disabled={!isPaused}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Step forward one frame (when paused)"
          >
            <StepForward className="w-4 h-4" />
          </button>

          {/* Replay / Reset */}
          <button
            onClick={onReplay}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 transition-colors"
            title="Reconnect stream"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1 bg-surface px-2 py-1 rounded-lg border border-border text-xs font-mono">
          <span className="text-neutral-500 mr-1">SPEED:</span>
          {[0.5, 1, 2, 5].map((s) => (
            <button
              key={s}
              onClick={() => onChangeSpeed(s)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                speed === s
                  ? "bg-primary-600 text-white font-bold"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* PPO Learning Cycle Visualizer */}
      <div className="mt-5 pt-4 border-t border-border-subtle">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold text-neutral-300 uppercase tracking-wider">
              PPO Training Cycle
            </span>
            <button
              onClick={() => setShowPPOInfo(!showPPOInfo)}
              className="text-neutral-500 hover:text-neutral-300"
              title="Explain PPO Learning Cycle"
            >
              <Info className="w-3.5 h-3.5" />
            </button>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-primary-500/10 text-primary-400 border border-primary-500/20 font-bold">
            {data?.phase_step || data?.phase_epoch || data?.phase || "COLLECTING"}
          </span>
        </div>

        {/* Cycle Stage Progress Indicators */}
        <div className="grid grid-cols-5 gap-1.5 text-center">
          {ppoStages.map((stage, idx) => {
            const isCurrent = idx === activeIndex;
            const isDone = idx < activeIndex;

            return (
              <div
                key={stage.id}
                className={`p-2 rounded-lg border transition-all text-left flex flex-col justify-between ${
                  isCurrent
                    ? "bg-primary-500/15 border-primary-500/40 text-primary-300 shadow-sm"
                    : isDone
                    ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                    : "bg-[#111111] border-border-subtle text-neutral-600"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[9px] font-mono font-bold">0{idx + 1}</span>
                  {isCurrent && (
                    <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-ping" />
                  )}
                </div>
                <span className="text-[10px] font-mono font-semibold truncate leading-tight">
                  {stage.label}
                </span>
              </div>
            );
          })}
        </div>

        {/* Detail text explaining the current step */}
        <p className="text-[11px] font-mono text-neutral-400 mt-2 bg-[#121212] p-2.5 rounded-lg border border-border-subtle">
          <span className="text-primary-400 font-bold">CURRENT STATUS: </span>
          {data?.phase_details || "Agent is interacting with Crafter environment to collect recurrent rollouts."}
        </p>

        {showPPOInfo && (
          <div className="mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-xs text-neutral-300 space-y-1 font-sans">
            <p className="font-semibold text-white">How PPO Learns (Not per-death):</p>
            <p className="text-neutral-400 leading-relaxed">
              Proximal Policy Optimization (PPO) does not update network weights after each individual death. Instead, it collects rollout batches (e.g. 128 timesteps across 4 parallel environments = 512 transitions). Once full, it calculates Generalized Advantage Estimations (GAE-λ), then performs 4 epochs of clipped surrogate updates on Actor and Critic networks with gradient clipping.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import React, { useState, useEffect } from "react";
import {
  X,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Award,
  Layers,
} from "lucide-react";
import { EpisodeDetail } from "@/types";
import { CRAFTER_ACTIONS } from "@/lib/utils";
import { api } from "@/lib/api";

interface EpisodeCompareProps {
  epA: EpisodeDetail;
  epB: EpisodeDetail;
  onClose: () => void;
}

export function EpisodeCompare({ epA, epB, onClose }: EpisodeCompareProps) {
  const [framesA, setFramesA] = useState<string[]>([]);
  const [framesB, setFramesB] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loadingFrames, setLoadingFrames] = useState(true);

  // Load frames for both episodes
  useEffect(() => {
    setLoadingFrames(true);
    Promise.allSettled([
      api.getEpisodeFrames(epA.episode_id),
      api.getEpisodeFrames(epB.episode_id),
    ]).then(([resA, resB]) => {
      if (resA.status === "fulfilled" && resA.value?.frames) setFramesA(resA.value.frames);
      if (resB.status === "fulfilled" && resB.value?.frames) setFramesB(resB.value.frames);
      setLoadingFrames(false);
    });
  }, [epA.episode_id, epB.episode_id]);

  const maxSteps = Math.max(framesA.length, framesB.length, epA.length, epB.length, 1);

  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 100);
    }
    return () => clearInterval(timer);
  }, [isPlaying, maxSteps]);

  const getAchievementsList = (ep: EpisodeDetail) =>
    Object.entries(ep.achievements || {})
      .filter(([_, v]) => v)
      .map(([k]) => k);

  const achA = getAchievementsList(epA);
  const achB = getAchievementsList(epB);

  const actionAIdx = epA.actions[currentStep];
  const actionBIdx = epB.actions[currentStep];

  const actionAName = actionAIdx !== undefined ? CRAFTER_ACTIONS[actionAIdx] || `act_${actionAIdx}` : "--";
  const actionBName = actionBIdx !== undefined ? CRAFTER_ACTIONS[actionBIdx] || `act_${actionBIdx}` : "--";

  const frameA = framesA[Math.min(currentStep, framesA.length - 1)];
  const frameB = framesB[Math.min(currentStep, framesB.length - 1)];

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-6">
      <div className="bg-[#121212] border border-border rounded-2xl max-w-6xl w-full overflow-hidden flex flex-col max-h-[92vh] shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">
                Side-by-Side Episode Learning Comparison
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                POLICY PROGRESSION
              </span>
            </div>
            <p className="text-xs text-neutral-400 font-mono mt-0.5">
              Compare Baseline Policy (Episode #{epA.episode_id}) vs Updated Policy (Episode #{epB.episode_id})
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Comparison Body */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8 overflow-y-auto">
          {/* Column A: Earlier / Baseline Episode */}
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-surface border border-border">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-mono text-neutral-400 font-bold uppercase tracking-wider">
                  BASELINE POLICY
                </span>
                <span className="text-xs font-mono text-neutral-500">
                  Global Step: {epA.global_step.toLocaleString()}
                </span>
              </div>
              <h4 className="text-lg font-bold text-white">Episode #{epA.episode_id}</h4>
            </div>

            {/* Visual Replay Frame A */}
            <div className="w-full aspect-square max-w-[320px] mx-auto rounded-xl overflow-hidden bg-black border border-border relative flex items-center justify-center">
              {frameA ? (
                <img
                  src={`data:image/jpeg;base64,${frameA}`}
                  alt="Episode A Observation"
                  className="w-full h-full object-contain pixelated"
                  style={{ imageRendering: "pixelated" }}
                />
              ) : (
                <div className="text-neutral-600 text-xs font-mono">
                  {loadingFrames ? "LOADING OBSERVATIONS..." : "NO FRAME DATA"}
                </div>
              )}
              <div className="absolute bottom-2 left-2 right-2 bg-black/80 px-2.5 py-1 rounded text-xs font-mono flex justify-between">
                <span className="text-neutral-400 text-[10px]">ACTION:</span>
                <span className="text-neutral-200 font-bold uppercase">{actionAName}</span>
              </div>
            </div>

            {/* Metrics Grid A */}
            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-0.5 text-[10px]">RETURN</span>
                <span className="text-base font-bold text-white">{epA.total_reward.toFixed(1)}</span>
              </div>
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-0.5 text-[10px]">SURVIVAL</span>
                <span className="text-base font-bold text-white">{epA.length} steps</span>
              </div>
            </div>

            {/* Achievements A */}
            <div className="p-4 rounded-xl bg-surface border border-border">
              <span className="text-xs font-mono text-neutral-400 block mb-2 uppercase tracking-wider">
                Achievements Unlocked ({achA.length})
              </span>
              <div className="flex flex-wrap gap-1.5">
                {achA.length > 0 ? (
                  achA.map((a) => (
                    <span
                      key={a}
                      className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-neutral-700 text-xs font-mono"
                    >
                      {a}
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-xs font-mono">None unlocked</span>
                )}
              </div>
            </div>
          </div>

          {/* Column B: Later / Updated Episode */}
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-surface border border-emerald-500/30">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-mono text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" /> UPDATED POLICY
                </span>
                <span className="text-xs font-mono text-neutral-500">
                  Global Step: {epB.global_step.toLocaleString()}
                </span>
              </div>
              <h4 className="text-lg font-bold text-white">Episode #{epB.episode_id}</h4>
            </div>

            {/* Visual Replay Frame B */}
            <div className="w-full aspect-square max-w-[320px] mx-auto rounded-xl overflow-hidden bg-black border-2 border-emerald-500/40 relative flex items-center justify-center">
              {frameB ? (
                <img
                  src={`data:image/jpeg;base64,${frameB}`}
                  alt="Episode B Observation"
                  className="w-full h-full object-contain pixelated"
                  style={{ imageRendering: "pixelated" }}
                />
              ) : (
                <div className="text-neutral-600 text-xs font-mono">
                  {loadingFrames ? "LOADING OBSERVATIONS..." : "NO FRAME DATA"}
                </div>
              )}
              <div className="absolute bottom-2 left-2 right-2 bg-black/80 px-2.5 py-1 rounded text-xs font-mono flex justify-between">
                <span className="text-neutral-400 text-[10px]">ACTION:</span>
                <span className="text-emerald-400 font-bold uppercase">{actionBName}</span>
              </div>
            </div>

            {/* Metrics Grid B */}
            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-0.5 text-[10px]">RETURN</span>
                <span className="text-base font-bold text-emerald-400">
                  {epB.total_reward.toFixed(1)}
                  {epB.total_reward > epA.total_reward && (
                    <span className="text-xs text-emerald-500 ml-1">
                      (+{(epB.total_reward - epA.total_reward).toFixed(1)})
                    </span>
                  )}
                </span>
              </div>
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-0.5 text-[10px]">SURVIVAL</span>
                <span className="text-base font-bold text-emerald-400">
                  {epB.length} steps
                  {epB.length > epA.length && (
                    <span className="text-xs text-emerald-500 ml-1">
                      (+{epB.length - epA.length})
                    </span>
                  )}
                </span>
              </div>
            </div>

            {/* Achievements B */}
            <div className="p-4 rounded-xl bg-surface border border-border">
              <span className="text-xs font-mono text-neutral-400 block mb-2 uppercase tracking-wider">
                Achievements Unlocked ({achB.length})
              </span>
              <div className="flex flex-wrap gap-1.5">
                {achB.length > 0 ? (
                  achB.map((a) => (
                    <span
                      key={a}
                      className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono font-semibold"
                    >
                      {a}
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-xs font-mono">None unlocked</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Synchronized Scrubber */}
        <div className="p-6 border-t border-border bg-[#0f0f0f] space-y-3">
          <input
            type="range"
            min={0}
            max={Math.max(0, maxSteps - 1)}
            value={currentStep}
            onChange={(e) => setCurrentStep(Number(e.target.value))}
            className="w-full accent-primary-500 cursor-pointer h-2 bg-neutral-800 rounded-lg"
          />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentStep((p) => Math.max(0, p - 1))}
                className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="px-4 py-2 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center gap-2 hover:bg-primary-500"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                {isPlaying ? "Pause Sync" : "Play Sync"}
              </button>
              <button
                onClick={() => setCurrentStep((p) => Math.min(maxSteps - 1, p + 1))}
                className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            <span className="text-xs font-mono text-neutral-400">
              Synchronized Step: {currentStep} / {maxSteps - 1}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

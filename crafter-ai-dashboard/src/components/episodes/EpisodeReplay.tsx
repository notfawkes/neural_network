"use client";

import React, { useState, useEffect } from "react";
import {
  Play,
  Pause,
  X,
  ChevronLeft,
  ChevronRight,
  Skull,
  AlertTriangle,
  Compass,
  Trophy,
} from "lucide-react";
import { EpisodeDetail } from "@/types";
import { CRAFTER_ACTIONS } from "@/lib/utils";

interface EpisodeReplayProps {
  episode: EpisodeDetail;
  frames: string[];
  onClose: () => void;
}

export function EpisodeReplay({ episode, frames, onClose }: EpisodeReplayProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const totalFrames = frames.length;

  useEffect(() => {
    let timer: any;
    if (isPlaying && totalFrames > 0) {
      const intervalMs = Math.max(20, Math.round(100 / speed));
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalFrames - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, intervalMs);
    }
    return () => clearInterval(timer);
  }, [isPlaying, totalFrames, speed]);

  const currentActionIdx = episode.actions[currentStep];
  const currentActionName =
    currentActionIdx !== undefined ? CRAFTER_ACTIONS[currentActionIdx] || `action_${currentActionIdx}` : "--";
  const currentReward = episode.rewards[currentStep] ?? 0;
  const currentProbs = episode.action_probs && episode.action_probs[currentStep];
  const currentCritic = episode.critic_values && episode.critic_values[currentStep];
  const currentPos = episode.player_positions && episode.player_positions[currentStep];

  // Find step of first mistake/damage
  const firstDamageStep = episode.rewards.findIndex((r) => r < 0);
  const deathStep = totalFrames > 0 ? totalFrames - 1 : 0;

  const jumpToDamage = () => {
    if (firstDamageStep >= 0) {
      setIsPlaying(false);
      setCurrentStep(firstDamageStep);
    }
  };

  const jumpToDeath = () => {
    setIsPlaying(false);
    setCurrentStep(deathStep);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-6">
      <div className="bg-[#121212] border border-border rounded-2xl max-w-5xl w-full overflow-hidden flex flex-col max-h-[92vh] shadow-2xl">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">
                Episode Memory Explorer #{episode.episode_id}
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-primary-500/10 text-primary-400 border border-primary-500/20 font-bold">
                {episode.model_type}
              </span>
            </div>
            <p className="text-xs text-neutral-400 font-mono mt-0.5">
              Step Recorded: {episode.global_step.toLocaleString()} | Total Steps: {episode.length} | Return: {episode.total_reward.toFixed(1)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-12 gap-6 items-start overflow-y-auto">
          {/* Left: Pixelated Game Screen */}
          <div className="md:col-span-5 flex flex-col items-center">
            <div className="w-full aspect-square max-w-[340px] rounded-xl overflow-hidden bg-black border-2 border-border relative shadow-2xl flex items-center justify-center">
              {frames[currentStep] ? (
                <img
                  src={`data:image/jpeg;base64,${frames[currentStep]}`}
                  alt="Recorded Frame"
                  className="w-full h-full object-contain pixelated"
                  style={{ imageRendering: "pixelated" }}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-neutral-600 text-xs font-mono">
                  NO FRAME BUFFERED
                </div>
              )}

              {/* Death Indicator Badge */}
              {currentStep === totalFrames - 1 && episode.terminated && (
                <div className="absolute top-3 right-3 px-2.5 py-1 rounded bg-rose-500/30 text-rose-300 border border-rose-500/50 text-xs font-mono font-bold flex items-center gap-1.5 shadow-lg backdrop-blur-sm">
                  <Skull className="w-3.5 h-3.5" />
                  <span>AGENT TERMINATED</span>
                </div>
              )}

              {/* Action Banner */}
              <div className="absolute bottom-2 left-2 right-2 bg-black/85 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-white/10 flex items-center justify-between text-xs font-mono">
                <span className="text-neutral-400 text-[10px]">TAKEN:</span>
                <span className="text-primary-400 font-bold uppercase">{currentActionName}</span>
              </div>
            </div>

            {/* Quick Failure Jump Buttons */}
            <div className="flex items-center gap-2 mt-4 w-full justify-center text-xs font-mono">
              {firstDamageStep >= 0 && (
                <button
                  onClick={jumpToDamage}
                  className="px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 hover:bg-rose-500/20 flex items-center gap-1.5 transition-colors"
                >
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Jump to Damage (T={firstDamageStep})
                </button>
              )}
              {episode.terminated && (
                <button
                  onClick={jumpToDeath}
                  className="px-3 py-1.5 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-300 hover:text-white flex items-center gap-1.5 transition-colors"
                >
                  <Skull className="w-3.5 h-3.5" />
                  Jump to Death (T={deathStep})
                </button>
              )}
            </div>
          </div>

          {/* Right: Step Telemetry & Action Distribution */}
          <div className="md:col-span-7 space-y-4">
            {/* Step Stats Grid */}
            <div className="p-4 rounded-xl bg-surface border border-border grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
              <div className="p-2.5 bg-[#141414] rounded-lg border border-border-subtle">
                <span className="text-neutral-500 block text-[10px] mb-0.5">TIMESTEP</span>
                <span className="text-white font-bold">{currentStep} / {Math.max(0, totalFrames - 1)}</span>
              </div>
              <div className="p-2.5 bg-[#141414] rounded-lg border border-border-subtle">
                <span className="text-neutral-500 block text-[10px] mb-0.5">INSTANT REWARD</span>
                <span className={`font-bold ${currentReward > 0 ? "text-emerald-400" : currentReward < 0 ? "text-rose-400" : "text-neutral-300"}`}>
                  {currentReward > 0 ? `+${currentReward}` : currentReward}
                </span>
              </div>
              <div className="p-2.5 bg-[#141414] rounded-lg border border-border-subtle">
                <span className="text-neutral-500 block text-[10px] mb-0.5">CRITIC V(s)</span>
                <span className="text-neutral-300 font-bold">
                  {currentCritic !== null && currentCritic !== undefined ? currentCritic.toFixed(3) : "--"}
                </span>
              </div>
              <div className="p-2.5 bg-[#141414] rounded-lg border border-border-subtle">
                <span className="text-neutral-500 block text-[10px] mb-0.5">COORDINATES</span>
                <span className="text-neutral-300 font-bold">
                  {currentPos ? `[${currentPos[0]}, ${currentPos[1]}]` : "--"}
                </span>
              </div>
            </div>

            {/* Action Softmax Distribution at THIS exact timestep */}
            <div className="p-4 rounded-xl bg-surface border border-border space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-neutral-300 font-semibold">Policy Distribution at Step {currentStep}</span>
                <span className="text-neutral-500 text-[10px]">17 Actions</span>
              </div>

              <div
                className="grid gap-1 h-28 items-end pt-2 border-b border-border-subtle"
                style={{ gridTemplateColumns: "repeat(17, minmax(0, 1fr))" }}
              >
                {CRAFTER_ACTIONS.map((name, i) => {
                  const prob = currentProbs && currentProbs[i] !== undefined ? currentProbs[i] : 1 / 17;
                  const isSelected = i === currentActionIdx;
                  const heightPercent = Math.min(100, Math.max(6, Math.round(prob * 100)));

                  return (
                    <div key={name} className="flex flex-col items-center h-full justify-end group">
                      <div
                        className={`w-full rounded-t transition-all ${
                          isSelected
                            ? "bg-primary-500 shadow-md shadow-primary-500/50"
                            : "bg-neutral-800 group-hover:bg-neutral-700"
                        }`}
                        style={{ height: `${heightPercent}%` }}
                        title={`${name}: ${(prob * 100).toFixed(1)}%`}
                      />
                      <span
                        className={`text-[8px] font-mono mt-1 truncate w-full text-center ${
                          isSelected ? "text-primary-400 font-bold" : "text-neutral-600"
                        }`}
                      >
                        {name.slice(0, 3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Achievements Unlocked in this Run */}
            <div className="p-4 rounded-xl bg-surface border border-border">
              <div className="flex items-center gap-1.5 mb-2.5">
                <Trophy className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-xs font-mono text-neutral-300 font-semibold uppercase tracking-wider">
                  Achievements in this Episode
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(episode.achievements).filter(([_, v]) => v).length > 0 ? (
                  Object.entries(episode.achievements)
                    .filter(([_, v]) => v)
                    .map(([name]) => (
                      <span
                        key={name}
                        className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono font-medium"
                      >
                        {name}
                      </span>
                    ))
                ) : (
                  <span className="text-neutral-600 text-xs font-mono">
                    No achievements unlocked during this exploratory run
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Modal Scrubber & Controls */}
        <div className="p-6 border-t border-border bg-[#0f0f0f] space-y-3">
          <input
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            value={currentStep}
            onChange={(e) => setCurrentStep(Number(e.target.value))}
            className="w-full accent-primary-500 cursor-pointer h-2 bg-neutral-800 rounded-lg"
          />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentStep((p) => Math.max(0, p - 1))}
                className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 transition-colors"
                title="Previous Frame"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="px-4 py-2 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center gap-2 hover:bg-primary-500 transition-colors"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => setCurrentStep((p) => Math.min(totalFrames - 1, p + 1))}
                className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 transition-colors"
                title="Next Frame"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Playback speed */}
            <div className="flex items-center gap-1 bg-surface px-2 py-1 rounded-lg border border-border text-xs font-mono">
              <span className="text-neutral-500 mr-1">SPEED:</span>
              {[0.5, 1, 2, 5].map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                    speed === s ? "bg-primary-600 text-white font-bold" : "text-neutral-400 hover:text-white"
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>

            <span className="text-xs font-mono text-neutral-400">
              Frame {currentStep + 1} of {totalFrames}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

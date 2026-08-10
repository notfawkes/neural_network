"use client";

import React, { useState, useEffect } from "react";
import { Play, Pause, X, ChevronLeft, ChevronRight, Activity } from "lucide-react";
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

  useEffect(() => {
    let timer: any;
    if (isPlaying && frames.length > 0) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= frames.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 100);
    }
    return () => clearInterval(timer);
  }, [isPlaying, frames]);

  const currentActionIdx = episode.actions[currentStep];
  const currentActionName = currentActionIdx !== undefined ? CRAFTER_ACTIONS[currentActionIdx] : "--";
  const currentReward = episode.rewards[currentStep] ?? 0;
  const currentProb = episode.action_probs[currentStep];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-6">
      <div className="bg-[#121212] border border-border rounded-2xl max-w-4xl w-full overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-white">
              Episode Replay #{episode.episode_id}
            </h3>
            <p className="text-xs text-neutral-400 font-mono">
              Total Steps: {episode.length} | Return: {episode.total_reward.toFixed(1)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          {/* Observation View */}
          <div className="flex flex-col items-center">
            <div className="w-72 h-72 rounded-xl overflow-hidden bg-black border border-border relative">
              {frames[currentStep] ? (
                <img
                  src={`data:image/jpeg;base64,${frames[currentStep]}`}
                  alt="Frame"
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <div className="flex items-center justify-center h-full text-neutral-600 text-xs font-mono">
                  NO FRAME DATA
                </div>
              )}
              {currentStep === frames.length - 1 && episode.terminated && (
                <div className="absolute top-2 right-2 px-2 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 text-[10px] font-mono font-bold">
                  AGENT DIED
                </div>
              )}
            </div>
          </div>

          {/* Timestep Breakdown */}
          <div className="space-y-4 font-mono text-xs">
            <div className="p-4 rounded-xl bg-surface border border-border space-y-2">
              <div className="flex justify-between">
                <span className="text-neutral-500">TIMESTEP:</span>
                <span className="text-white font-bold">{currentStep} / {episode.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-500">CHOSEN ACTION:</span>
                <span className="text-primary-400 font-bold uppercase">{currentActionName}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-500">INSTANT REWARD:</span>
                <span className={currentReward >= 0 ? "text-emerald-400" : "text-rose-400"}>
                  {currentReward > 0 ? `+${currentReward}` : currentReward}
                </span>
              </div>
              {episode.critic_values[currentStep] !== undefined && (
                <div className="flex justify-between">
                  <span className="text-neutral-500">CRITIC VALUE:</span>
                  <span className="text-neutral-300">
                    {episode.critic_values[currentStep]?.toFixed(3) ?? "--"}
                  </span>
                </div>
              )}
            </div>

            {/* Achievements in Episode */}
            <div className="p-4 rounded-xl bg-surface border border-border">
              <span className="text-neutral-400 block mb-2 uppercase text-[10px] tracking-wider">
                Achievements in this run:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(episode.achievements).filter(([_, v]) => v).length > 0 ? (
                  Object.entries(episode.achievements)
                    .filter(([_, v]) => v)
                    .map(([name]) => (
                      <span
                        key={name}
                        className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[11px]"
                      >
                        {name}
                      </span>
                    ))
                ) : (
                  <span className="text-neutral-600 text-[11px]">None unlocked</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Scrubber & Controls */}
        <div className="p-6 border-t border-border bg-[#0f0f0f] space-y-4">
          <input
            type="range"
            min={0}
            max={Math.max(0, frames.length - 1)}
            value={currentStep}
            onChange={(e) => setCurrentStep(Number(e.target.value))}
            className="w-full accent-primary-500 cursor-pointer"
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
                className="px-4 py-2 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center gap-2"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => setCurrentStep((p) => Math.min(frames.length - 1, p + 1))}
                className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
            <span className="text-xs font-mono text-neutral-500">
              Frame {currentStep + 1} of {frames.length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

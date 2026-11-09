"use client";

import React from "react";
import { X, Trophy, Flame, Clock, Navigation } from "lucide-react";
import { EpisodeDetail } from "@/types";

interface EpisodeCompareProps {
  epA: EpisodeDetail;
  epB: EpisodeDetail;
  onClose: () => void;
}

export function EpisodeCompare({ epA, epB, onClose }: EpisodeCompareProps) {
  const getAchievementsList = (ep: EpisodeDetail) =>
    Object.entries(ep.achievements).filter(([_, v]) => v).map(([k]) => k);

  const achA = getAchievementsList(epA);
  const achB = getAchievementsList(epB);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-6">
      <div className="bg-[#121212] border border-border rounded-2xl max-w-5xl w-full overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-white">
              Episode Milestone Comparison
            </h3>
            <p className="text-xs text-neutral-400 font-mono">
              Compare Episode #{epA.episode_id} vs Episode #{epB.episode_id}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Comparison Grid */}
        <div className="p-6 grid grid-cols-2 gap-8 overflow-y-auto">
          {/* Episode A */}
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-surface border border-border">
              <span className="text-xs font-mono text-primary-400 font-bold block mb-1">
                BASELINE / EARLIER
              </span>
              <h4 className="text-xl font-bold text-white">Episode #{epA.episode_id}</h4>
              <p className="text-xs text-neutral-500 font-mono mt-0.5">
                Step: {epA.global_step.toLocaleString()}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-1">RETURN</span>
                <span className="text-lg font-bold text-white">{epA.total_reward.toFixed(1)}</span>
              </div>
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-1">DURATION</span>
                <span className="text-lg font-bold text-white">{epA.length} steps</span>
              </div>
            </div>

            {/* Achievements */}
            <div className="p-4 rounded-xl bg-surface border border-border">
              <span className="text-xs font-mono text-neutral-400 block mb-3 uppercase tracking-wider">
                Achievements Unlocked ({achA.length})
              </span>
              <div className="flex flex-wrap gap-1.5">
                {achA.length > 0 ? (
                  achA.map((a) => (
                    <span
                      key={a}
                      className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono"
                    >
                      {a}
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-xs font-mono">None unlocked</span>
                )}
              </div>
            </div>

            {/* Exploration summary */}
            <div className="p-4 rounded-xl bg-surface border border-border font-mono text-xs">
              <span className="text-neutral-400 block mb-2 uppercase tracking-wider">
                Exploration Path
              </span>
              <span className="text-neutral-300">
                {epA.player_positions.length} coordinates recorded
              </span>
            </div>
          </div>

          {/* Episode B */}
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-surface border border-primary-500/30">
              <span className="text-xs font-mono text-emerald-400 font-bold block mb-1">
                LATER / UPDATED
              </span>
              <h4 className="text-xl font-bold text-white">Episode #{epB.episode_id}</h4>
              <p className="text-xs text-neutral-500 font-mono mt-0.5">
                Step: {epB.global_step.toLocaleString()}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-1">RETURN</span>
                <span className="text-lg font-bold text-emerald-400">{epB.total_reward.toFixed(1)}</span>
              </div>
              <div className="p-3 bg-surface rounded-lg border border-border">
                <span className="text-neutral-500 block mb-1">DURATION</span>
                <span className="text-lg font-bold text-white">{epB.length} steps</span>
              </div>
            </div>

            {/* Achievements */}
            <div className="p-4 rounded-xl bg-surface border border-border">
              <span className="text-xs font-mono text-neutral-400 block mb-3 uppercase tracking-wider">
                Achievements Unlocked ({achB.length})
              </span>
              <div className="flex flex-wrap gap-1.5">
                {achB.length > 0 ? (
                  achB.map((a) => (
                    <span
                      key={a}
                      className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono"
                    >
                      {a}
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-xs font-mono">None unlocked</span>
                )}
              </div>
            </div>

            {/* Exploration summary */}
            <div className="p-4 rounded-xl bg-surface border border-border font-mono text-xs">
              <span className="text-neutral-400 block mb-2 uppercase tracking-wider">
                Exploration Path
              </span>
              <span className="text-neutral-300">
                {epB.player_positions.length} coordinates recorded
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


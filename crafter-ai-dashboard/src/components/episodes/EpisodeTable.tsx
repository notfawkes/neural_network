"use client";

import React from "react";
import { EpisodeSummary } from "@/types";
import { Play, CheckSquare, Square, ChevronRight } from "lucide-react";

interface EpisodeTableProps {
  episodes: EpisodeSummary[];
  selectedIds: number[];
  onToggleSelect: (id: number) => void;
  onPlayEpisode: (id: number) => void;
}

export function EpisodeTable({
  episodes,
  selectedIds,
  onToggleSelect,
  onPlayEpisode,
}: EpisodeTableProps) {
  return (
    <div className="bg-surface rounded-xl border border-border overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-[#141414] text-neutral-400 uppercase tracking-wider border-b border-border">
            <tr>
              <th className="p-4 w-12 text-center">Compare</th>
              <th className="p-4">Episode ID</th>
              <th className="p-4">Global Step</th>
              <th className="p-4">Return</th>
              <th className="p-4">Duration</th>
              <th className="p-4">Achievements</th>
              <th className="p-4">Outcome</th>
              <th className="p-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-subtle">
            {episodes.map((ep) => {
              const isSelected = selectedIds.includes(ep.episode_id);
              return (
                <tr
                  key={ep.episode_id}
                  className="hover:bg-surface-hover/50 transition-colors group"
                >
                  <td className="p-4 text-center">
                    <button
                      onClick={() => onToggleSelect(ep.episode_id)}
                      className="text-neutral-500 hover:text-white"
                    >
                      {isSelected ? (
                        <CheckSquare className="w-4 h-4 text-primary-400" />
                      ) : (
                        <Square className="w-4 h-4" />
                      )}
                    </button>
                  </td>
                  <td className="p-4 font-bold text-white">#{ep.episode_id}</td>
                  <td className="p-4 text-neutral-400">{ep.global_step.toLocaleString()}</td>
                  <td
                    className={`p-4 font-semibold ${
                      ep.total_reward >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {ep.total_reward > 0 ? `+${ep.total_reward.toFixed(1)}` : ep.total_reward.toFixed(1)}
                  </td>
                  <td className="p-4 text-neutral-400">{ep.length} steps</td>
                  <td className="p-4">
                    <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-300">
                      {ep.num_achievements} unlocked
                    </span>
                  </td>
                  <td className="p-4">
                    {ep.terminated ? (
                      <span className="text-rose-400">Died (Health=0)</span>
                    ) : (
                      <span className="text-amber-400">Truncated (Max steps)</span>
                    )}
                  </td>
                  <td className="p-4 text-right">
                    <button
                      onClick={() => onPlayEpisode(ep.episode_id)}
                      className="px-3 py-1.5 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 flex items-center gap-1.5 ml-auto text-xs"
                    >
                      <Play className="w-3 h-3 fill-current" /> Replay
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

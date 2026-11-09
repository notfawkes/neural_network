"use client";

import React, { useState, useEffect } from "react";
import { EpisodeTable } from "@/components/episodes/EpisodeTable";
import { EpisodeReplay } from "@/components/episodes/EpisodeReplay";
import { EpisodeCompare } from "@/components/episodes/EpisodeCompare";
import { EpisodeSummary, EpisodeDetail } from "@/types";
import { api } from "@/lib/api";
import { RefreshCw, ArrowLeftRight } from "lucide-react";

export default function EpisodesPage() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [activeReplay, setActiveReplay] = useState<{ episode: EpisodeDetail; frames: string[] } | null>(null);
  const [activeCompare, setActiveCompare] = useState<{ epA: EpisodeDetail; epB: EpisodeDetail } | null>(null);
  const [loading, setLoading] = useState(true);

  const loadEpisodes = async () => {
    try {
      setLoading(true);
      const data = await api.getEpisodes(50, 0);
      setEpisodes(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEpisodes();
  }, []);

  const handleToggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((i) => i !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  };

  const handlePlayEpisode = async (id: number) => {
    try {
      const [epDetail, framesRes] = await Promise.all([
        api.getEpisode(id),
        api.getEpisodeFrames(id, 4),
      ]);
      setActiveReplay({ episode: epDetail, frames: framesRes.frames });
    } catch (err) {
      console.error("Failed to load episode replay:", err);
    }
  };

  const handleOpenCompare = async () => {
    if (selectedIds.length !== 2) return;
    try {
      const [epA, epB] = await Promise.all([
        api.getEpisode(selectedIds[0]),
        api.getEpisode(selectedIds[1]),
      ]);
      setActiveCompare({ epA, epB });
    } catch (err) {
      console.error("Failed to load compare episodes:", err);
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="border-b border-border pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Episode Explorer</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Recorded gameplay sessions, agent failure analysis & milestone trajectories
          </p>
        </div>

        <div className="flex items-center gap-3">
          {selectedIds.length === 2 && (
            <button
              onClick={handleOpenCompare}
              className="px-4 py-2 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center gap-2 shadow-sm hover:bg-primary-500"
            >
              <ArrowLeftRight className="w-3.5 h-3.5" /> Compare Selected (2)
            </button>
          )}

          <button
            onClick={loadEpisodes}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-400 hover:text-white"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Table */}
      {episodes.length > 0 ? (
        <EpisodeTable
          episodes={episodes}
          selectedIds={selectedIds}
          onToggleSelect={handleToggleSelect}
          onPlayEpisode={handlePlayEpisode}
        />
      ) : (
        <div className="p-12 text-center rounded-xl bg-surface border border-border text-neutral-500 text-sm font-mono">
          {loading ? "Loading recorded episodes..." : "No recorded episodes found yet."}
        </div>
      )}

      {/* Replay Modal */}
      {activeReplay && (
        <EpisodeReplay
          episode={activeReplay.episode}
          frames={activeReplay.frames}
          onClose={() => setActiveReplay(null)}
        />
      )}

      {/* Compare Modal */}
      {activeCompare && (
        <EpisodeCompare
          epA={activeCompare.epA}
          epB={activeCompare.epB}
          onClose={() => setActiveCompare(null)}
        />
      )}
    </div>
  );
}


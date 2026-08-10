"use client";

import React, { useState, useEffect } from "react";
import { AchievementCard } from "@/components/achievements/AchievementCard";
import { AchievementItem } from "@/types";
import { api } from "@/lib/api";
import { Trophy, RefreshCw, Layers } from "lucide-react";

export default function AchievementsPage() {
  const [achievements, setAchievements] = useState<AchievementItem[]>([]);
  const [crafterScore, setCrafterScore] = useState(0);
  const [totalEpisodes, setTotalEpisodes] = useState(0);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    try {
      setLoading(true);
      const res = await api.getAchievements();
      setAchievements(res.achievements || []);
      setCrafterScore(res.crafter_score || 0);
      setTotalEpisodes(res.total_episodes || 0);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const categories = ["all", "mining", "survival", "combat", "crafting", "building", "farming"];

  const filtered = achievements.filter((a) =>
    filter === "all" ? true : a.category === filter
  );

  const unlockedCount = achievements.filter((a) => a.unlocked).length;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="border-b border-border pb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Crafter Achievements</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Tracking all 22 benchmark milestones for autonomous capability evaluation
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-4 py-2 rounded-xl bg-surface border border-border flex items-center gap-2">
            <Trophy className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-mono text-neutral-400">Score:</span>
            <span className="text-sm font-mono font-bold text-white">
              {(crafterScore * 100).toFixed(1)}%
            </span>
          </div>

          <button
            onClick={loadData}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-400 hover:text-white"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-5 rounded-xl bg-surface border border-border">
          <span className="text-xs font-mono text-neutral-500 uppercase tracking-wider block mb-1">
            Total Achievements Unlocked
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {unlockedCount} <span className="text-neutral-500 text-lg">/ 22</span>
          </div>
        </div>

        <div className="p-5 rounded-xl bg-surface border border-border">
          <span className="text-xs font-mono text-neutral-500 uppercase tracking-wider block mb-1">
            Crafter Geometric Score
          </span>
          <div className="text-2xl font-bold font-mono text-primary-400">
            {(crafterScore * 100).toFixed(2)}%
          </div>
        </div>

        <div className="p-5 rounded-xl bg-surface border border-border">
          <span className="text-xs font-mono text-neutral-500 uppercase tracking-wider block mb-1">
            Evaluated Episodes
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {totalEpisodes.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 border-b border-border-subtle">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono uppercase tracking-wider transition-all ${
              filter === cat
                ? "bg-primary-600 text-white font-semibold"
                : "bg-surface text-neutral-400 hover:text-white hover:bg-surface-hover"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Grid of 22 Achievement Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filtered.map((ach) => (
          <AchievementCard key={ach.name} achievement={ach} />
        ))}
      </div>
    </div>
  );
}

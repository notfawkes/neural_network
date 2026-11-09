"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  PlayCircle,
  Trophy,
  Flame,
  Clock,
  Layers,
  ArrowUpRight,
  RefreshCw,
  Square,
  Play,
} from "lucide-react";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { MetricCard } from "@/components/ui/MetricCard";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";
import { formatNumber, formatTime } from "@/lib/utils";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

export default function OverviewPage() {
  const [status, setStatus] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [achievements, setAchievements] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);

  const loadData = async () => {
    try {
      const [s, m, a] = await Promise.allSettled([
        api.getTrainingStatus(),
        api.getMetrics(),
        api.getAchievements(),
      ]);
      if (s.status === "fulfilled") setStatus(s.value);
      if (m.status === "fulfilled") setMetrics(m.value);
      if (a.status === "fulfilled") setAchievements(a.value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStartStop = async () => {
    setActionLoading(true);
    try {
      if (status?.is_running) {
        await api.stopTraining();
      } else {
        await api.startTraining({ config_path: "configs/default.yaml" });
      }
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  // Sparkline data
  const rewardData = (metrics?.recent_rewards || []).slice(-30).map((r: number, i: number) => ({
    episode: i,
    reward: Number(r.toFixed(2)),
  }));

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white">System Overview</h1>
            <StatusBadge status={status?.status || "idle"} />
          </div>
          <p className="text-sm text-neutral-400 mt-1">
            Real-time telemetry and reinforcement learning performance
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-400 hover:text-white hover:border-neutral-700 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={handleStartStop}
            disabled={actionLoading}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              status?.is_running
                ? "bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20"
                : "bg-primary-600 text-white hover:bg-primary-500 shadow-sm"
            }`}
          >
            {status?.is_running ? (
              <>
                <Square className="w-4 h-4 fill-current" /> Stop Training
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" /> Start Training
              </>
            )}
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <MetricCard
          label="Global Steps"
          value={formatNumber(status?.global_step || 0)}
          subValue="/ 1,000,000"
          icon={<Layers className="w-4 h-4" />}
        />
        <MetricCard
          label="Episodes"
          value={formatNumber(status?.episode_count || 0, 0)}
          subValue={`Avg len: ${Math.round(status?.mean_ep_length || 0)}`}
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricCard
          label="Mean Reward"
          value={(status?.mean_reward || 0).toFixed(2)}
          trend={(status?.mean_reward || 0) >= 0 ? "up" : "down"}
          subValue="Sliding window"
          icon={<Flame className="w-4 h-4" />}
        />
        <MetricCard
          label="Crafter Score"
          value={`${((status?.crafter_score || 0) * 100).toFixed(1)}%`}
          subValue="Geometric mean"
          icon={<Trophy className="w-4 h-4" />}
        />
        <MetricCard
          label="Speed"
          value={`${Math.round(status?.fps || 0)}`}
          subValue="Steps / sec"
          icon={<Clock className="w-4 h-4" />}
        />
        <MetricCard
          label="Policy Loss"
          value={(status?.policy_loss || 0).toFixed(4)}
          subValue={`KL: ${(status?.approx_kl || 0).toFixed(4)}`}
          icon={<Activity className="w-4 h-4" />}
        />
      </div>

      {/* Middle Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Reward Trend */}
        <Card
          title="Recent Reward Trajectory"
          subtitle="Episode return progression"
          className="lg:col-span-2"
          headerAction={
            <Link
              href="/analysis"
              className="text-xs font-mono text-primary-400 hover:underline flex items-center gap-1"
            >
              Deep Analysis <ArrowUpRight className="w-3 h-3" />
            </Link>
          }
        >
          {rewardData.length > 1 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rewardData}>
                  <XAxis dataKey="episode" stroke="#333" fontSize={11} />
                  <YAxis stroke="#333" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161616",
                      borderColor: "#262626",
                      borderRadius: 8,
                      fontSize: 12,
                      color: "#fff",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-neutral-500 text-sm">
              <Activity className="w-8 h-8 mb-2 stroke-1 opacity-50" />
              <span>No completed episodes yet</span>
              <span className="text-xs text-neutral-600 mt-1">
                Start training to observe reward curve
              </span>
            </div>
          )}
        </Card>

        {/* Top Achievements */}
        <Card
          title="Achievement Unlocks"
          subtitle="Top unlocked capabilities"
          headerAction={
            <Link
              href="/achievements"
              className="text-xs font-mono text-primary-400 hover:underline flex items-center gap-1"
            >
              All 22 <ArrowUpRight className="w-3 h-3" />
            </Link>
          }
        >
          <div className="space-y-3">
            {(achievements?.achievements || [])
              .slice(0, 6)
              .map((item: any) => (
                <div key={item.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="text-base">{item.icon}</span>
                    <span className="text-neutral-300 font-medium">{item.title}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full"
                        style={{ width: `${Math.min(100, item.success_rate * 100)}%` }}
                      />
                    </div>
                    <span className="font-mono text-neutral-400 w-10 text-right">
                      {(item.success_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </Card>
      </div>

      {/* Quick Launch / Shortcuts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link
          href="/live"
          className="p-5 rounded-xl bg-surface border border-border hover:border-neutral-700 flex items-center justify-between group transition-all"
        >
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-500/10 text-primary-400 flex items-center justify-center group-hover:scale-105 transition-transform">
              <PlayCircle className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Live Training View</h4>
              <p className="text-xs text-neutral-400 mt-0.5">
                Watch 64x64 pixelated agent POV, decisions & LSTM memory
              </p>
            </div>
          </div>
          <ArrowUpRight className="w-4 h-4 text-neutral-500 group-hover:text-white transition-colors" />
        </Link>

        <Link
          href="/episodes"
          className="p-5 rounded-xl bg-surface border border-border hover:border-neutral-700 flex items-center justify-between group transition-all"
        >
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center group-hover:scale-105 transition-transform">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Episode Explorer & Replay</h4>
              <p className="text-xs text-neutral-400 mt-0.5">
                Replay failure cases, deaths, and compare agent milestones
              </p>
            </div>
          </div>
          <ArrowUpRight className="w-4 h-4 text-neutral-500 group-hover:text-white transition-colors" />
        </Link>
      </div>
    </div>
  );
}


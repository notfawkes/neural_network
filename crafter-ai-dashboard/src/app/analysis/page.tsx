"use client";

import React, { useState, useEffect } from "react";
import { MetricsChart } from "@/components/analysis/MetricsChart";
import { api } from "@/lib/api";
import { RefreshCw, Download, Image as ImageIcon, Sparkles, X } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

interface FigureItem {
  filename: string;
  title: string;
  description: string;
  size_bytes: number;
  url: string;
  created_at: number;
}

export default function AnalysisPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [figures, setFigures] = useState<FigureItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [selectedFigure, setSelectedFigure] = useState<FigureItem | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const [m, f] = await Promise.allSettled([api.getMetrics(), api.getFigures()]);
      if (m.status === "fulfilled") setMetrics(m.value);
      if (f.status === "fulfilled") setFigures(f.value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateFigures = async () => {
    try {
      setGenerating(true);
      await api.generateFigures();
      const updated = await api.getFigures();
      setFigures(updated);
    } catch (err) {
      console.error("Failed to generate figures:", err);
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Format data series
  const rewardsSeries = (metrics?.recent_rewards || []).map((r: number, i: number) => ({
    episode: i + 1,
    reward: Number(r.toFixed(2)),
  }));

  const lengthsSeries = (metrics?.recent_lengths || []).map((l: number, i: number) => ({
    episode: i + 1,
    length: l,
  }));

  const lossesSeries = (metrics?.recent_losses || []).map((u: any, i: number) => ({
    update: i + 1,
    policy_loss: Number((u.policy_loss || 0).toFixed(4)),
    value_loss: Number((u.value_loss || 0).toFixed(4)),
    entropy: Number((u.entropy_loss ? Math.abs(u.entropy_loss) : 0).toFixed(4)),
    approx_kl: Number((u.approx_kl || 0).toFixed(4)),
  }));

  const achievementsBarData = Object.entries(metrics?.achievement_rates || {}).map(
    ([name, rate]: [string, any]) => ({
      name: name.replace(/_/g, " "),
      rate: Number((rate * 100).toFixed(1)),
    })
  );

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="border-b border-border pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Learning Analysis & Figures</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Empirical convergence curves, policy entropy, value loss & publication-grade evaluation figures
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleGenerateFigures}
            disabled={generating}
            className="px-3.5 py-2 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center gap-2 hover:bg-primary-500 disabled:opacity-50 transition-colors shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5" />
            {generating ? "Rendering Figures..." : "Export Figures"}
          </button>
          <button
            onClick={loadData}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-400 hover:text-white transition-colors"
            title="Reload telemetry"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Grid of Main RL Curves */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MetricsChart
          title="Episode Return Progression"
          subtitle="Reward received per completed episode"
          data={rewardsSeries}
          dataKey="reward"
          xAxisKey="episode"
          color="#3b82f6"
        />

        <MetricsChart
          title="Survival Duration"
          subtitle="Timesteps survived per episode"
          data={lengthsSeries}
          dataKey="length"
          xAxisKey="episode"
          color="#10b981"
          unit="steps"
        />

        <MetricsChart
          title="PPO Policy Loss"
          subtitle="Clipped surrogate objective"
          data={lossesSeries}
          dataKey="policy_loss"
          xAxisKey="update"
          color="#f59e0b"
        />

        <MetricsChart
          title="Critic Value Loss"
          subtitle="Mean squared error between critic and returns"
          data={lossesSeries}
          dataKey="value_loss"
          xAxisKey="update"
          color="#ef4444"
        />

        <MetricsChart
          title="Policy Entropy"
          subtitle="Exploration encouragement metric"
          data={lossesSeries}
          dataKey="entropy"
          xAxisKey="update"
          color="#8b5cf6"
        />

        <MetricsChart
          title="Approximate KL Divergence"
          subtitle="Step size between consecutive policy updates"
          data={lossesSeries}
          dataKey="approx_kl"
          xAxisKey="update"
          color="#ec4899"
        />
      </div>

      {/* Achievement Spectrum */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-md">
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-white">Achievement Success Spectrum</h3>
          <p className="text-xs text-neutral-500 font-mono mt-0.5">
            Success rate (%) across all training episodes
          </p>
        </div>

        {achievementsBarData.length > 0 ? (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={achievementsBarData} layout="vertical">
                <XAxis type="number" domain={[0, 100]} stroke="#444" fontSize={10} />
                <YAxis dataKey="name" type="category" stroke="#666" fontSize={10} width={130} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#161616",
                    borderColor: "#262626",
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                  formatter={(val: any) => [`${val}%`, "Success Rate"]}
                />
                <Bar dataKey="rate" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-44 flex items-center justify-center text-xs font-mono text-neutral-600">
            NO ACHIEVEMENT DATA YET
          </div>
        )}
      </div>

      {/* Generated Figures Section */}
      <div className="space-y-4 pt-4 border-t border-border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-primary-400" />
              Saved Training & Evaluation Figures
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              High-resolution publication figures stored in the <code className="text-primary-400">figures/</code> directory
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-neutral-800 text-neutral-300 border border-neutral-700">
            {figures.length} figures available
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {figures.map((fig) => (
            <div
              key={fig.filename}
              onClick={() => setSelectedFigure(fig)}
              className="bg-surface rounded-xl border border-border hover:border-neutral-700 overflow-hidden cursor-pointer group transition-all hover:shadow-xl flex flex-col justify-between"
            >
              <div>
                <div className="w-full aspect-[16/10] bg-black/60 relative overflow-hidden flex items-center justify-center border-b border-border-subtle">
                  <img
                    src={`http://localhost:8000${fig.url}`}
                    alt={fig.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="px-3 py-1.5 rounded-full bg-black/80 border border-white/20 text-white text-xs font-mono font-bold">
                      View High-Res
                    </span>
                  </div>
                </div>

                <div className="p-4">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono text-neutral-500">
                      {(fig.size_bytes / 1024).toFixed(0)} KB • PNG
                    </span>
                  </div>
                  <h3 className="text-sm font-semibold text-white group-hover:text-primary-400 transition-colors">
                    {fig.title}
                  </h3>
                  <p className="text-xs text-neutral-400 mt-1 line-clamp-2">
                    {fig.description}
                  </p>
                </div>
              </div>

              <div className="p-4 pt-0 flex justify-end">
                <a
                  href={`http://localhost:8000${fig.url}`}
                  download={fig.filename}
                  onClick={(e) => e.stopPropagation()}
                  className="px-2.5 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white text-xs font-mono flex items-center gap-1.5 transition-colors"
                >
                  <Download className="w-3 h-3" /> Save
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Modal Lightbox for Viewing Figure High-Res */}
      {selectedFigure && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-6">
          <div className="bg-[#121212] border border-border rounded-2xl max-w-5xl w-full max-h-[92vh] overflow-hidden flex flex-col shadow-2xl">
            <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface">
              <div>
                <h3 className="text-base font-semibold text-white">{selectedFigure.title}</h3>
                <p className="text-xs text-neutral-400 font-mono mt-0.5">{selectedFigure.description}</p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={`http://localhost:8000${selectedFigure.url}`}
                  download={selectedFigure.filename}
                  className="px-3 py-1.5 rounded-lg bg-primary-600 text-white text-xs font-medium flex items-center gap-1.5 hover:bg-primary-500"
                >
                  <Download className="w-3.5 h-3.5" /> Download
                </a>
                <button
                  onClick={() => setSelectedFigure(null)}
                  className="p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="p-6 overflow-auto flex items-center justify-center bg-black/60">
              <img
                src={`http://localhost:8000${selectedFigure.url}`}
                alt={selectedFigure.title}
                className="max-w-full max-h-[70vh] object-contain rounded-lg shadow-xl"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

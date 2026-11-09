"use client";

import React, { useState, useEffect } from "react";
import { MetricsChart } from "@/components/analysis/MetricsChart";
import { api } from "@/lib/api";
import { RefreshCw } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

export default function AnalysisPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    try {
      setLoading(true);
      const data = await api.getMetrics();
      setMetrics(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
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
          <h1 className="text-2xl font-bold tracking-tight text-white">Learning Analysis</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Empirical convergence curves, policy entropy, value loss & advantage stability
          </p>
        </div>
        <button
          onClick={loadData}
          className="p-2 rounded-lg bg-surface border border-border text-neutral-400 hover:text-white"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
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
      <div className="bg-surface rounded-xl border border-border p-6">
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
    </div>
  );
}


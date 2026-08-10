"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

interface RewardChartProps {
  data: Array<{ step: number; reward: number }>;
}

export function RewardChart({ data }: RewardChartProps) {
  const chartData = data.length > 0 ? data : [{ step: 0, reward: 0 }];

  return (
    <div className="bg-surface rounded-xl border border-border p-5">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-mono uppercase tracking-wider text-neutral-400">
          Live Episode Return
        </h4>
        <span className="text-xs font-mono text-primary-400 font-semibold">
          {chartData[chartData.length - 1]?.reward.toFixed(2) ?? "0.00"}
        </span>
      </div>

      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis dataKey="step" stroke="#333" fontSize={10} hide />
            <YAxis stroke="#333" fontSize={10} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#161616",
                borderColor: "#262626",
                borderRadius: 6,
                fontSize: 11,
              }}
            />
            <Line
              type="monotone"
              dataKey="reward"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

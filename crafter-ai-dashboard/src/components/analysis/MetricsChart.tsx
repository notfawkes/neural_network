"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface MetricsChartProps {
  title: string;
  subtitle?: string;
  data: any[];
  dataKey: string;
  xAxisKey?: string;
  color?: string;
  unit?: string;
}

export function MetricsChart({
  title,
  subtitle,
  data,
  dataKey,
  xAxisKey = "step",
  color = "#3b82f6",
  unit = "",
}: MetricsChartProps) {
  return (
    <div className="bg-surface rounded-xl border border-border p-5 flex flex-col justify-between">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        {subtitle && <p className="text-xs text-neutral-500 font-mono mt-0.5">{subtitle}</p>}
      </div>

      <div className="h-56 w-full">
        {data && data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
              <XAxis dataKey={xAxisKey} stroke="#444" fontSize={10} />
              <YAxis stroke="#444" fontSize={10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#161616",
                  borderColor: "#262626",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "#fff",
                }}
                formatter={(val: any) => [`${val} ${unit}`, title]}
              />
              <Line
                type="monotone"
                dataKey={dataKey}
                stroke={color}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-xs font-mono text-neutral-600">
            NO TELEMETRY RECORDED
          </div>
        )}
      </div>
    </div>
  );
}


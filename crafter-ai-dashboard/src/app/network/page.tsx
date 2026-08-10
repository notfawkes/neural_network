"use client";

import React, { useState } from "react";
import { ArchDiagram } from "@/components/network/ArchDiagram";
import { Card } from "@/components/ui/Card";
import { MetricCard } from "@/components/ui/MetricCard";
import { Layers, Cpu, Compass, Shield, Database } from "lucide-react";

export default function NetworkPage() {
  const [selectedLayer, setSelectedLayer] = useState("all");

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="border-b border-border pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Neural Network Inspector</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Structural topology, convolutional filters, recurrent state norms & parameter weights
          </p>
        </div>
      </div>

      {/* Visual Topology Diagram */}
      <ArchDiagram />

      {/* Layer Specs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* CNN Specs */}
        <Card title="CNN Feature Extractor" subtitle="Visual encoder">
          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Conv2D #1:</span>
              <span className="text-white">3 → 32, k=3, s=2</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Conv2D #2:</span>
              <span className="text-white">32 → 64, k=3, s=2</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Conv2D #3:</span>
              <span className="text-white">64 → 64, k=3, s=2</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Flatten Dimension:</span>
              <span className="text-white">64 × 7 × 7 = 3,136</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">FC Linear:</span>
              <span className="text-white">3,136 → 512</span>
            </div>
          </div>
        </Card>

        {/* LSTM Specs */}
        <Card title="Recurrent LSTM Core" subtitle="Temporal state integration">
          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Input Dimension:</span>
              <span className="text-white">512</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Hidden Dimension:</span>
              <span className="text-white">256</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Number of Layers:</span>
              <span className="text-white">1</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Batch Order:</span>
              <span className="text-white">(T, B, hidden)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">Boundary Reset:</span>
              <span className="text-emerald-400">Zeroed on done</span>
            </div>
          </div>
        </Card>

        {/* Heads Specs */}
        <Card title="Policy & Value Heads" subtitle="Actor & Critic linear projections">
          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Actor Layer:</span>
              <span className="text-white">Linear(256 → 17)</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Actor Init Gain:</span>
              <span className="text-white">0.01 (Orthogonal)</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Critic Layer:</span>
              <span className="text-white">Linear(256 → 1)</span>
            </div>
            <div className="flex justify-between border-b border-border-subtle pb-2">
              <span className="text-neutral-500">Critic Init Gain:</span>
              <span className="text-white">1.0 (Orthogonal)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">Action Distribution:</span>
              <span className="text-primary-400">Categorical</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useEffect } from "react";
import { CheckpointInfo, ExperimentInfo, CompareResponse } from "@/types";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import {
  Play,
  RotateCcw,
  Layers,
  ArrowRightLeft,
  CheckCircle,
  FileCode,
  HardDrive,
} from "lucide-react";

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<ExperimentInfo[]>([]);
  const [models, setModels] = useState<CheckpointInfo[]>([]);
  const [ckptA, setCkptA] = useState("");
  const [ckptB, setCkptB] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    Promise.allSettled([api.getExperiments(), api.getModels()]).then(([expRes, modRes]) => {
      if (expRes.status === "fulfilled") setExperiments(expRes.value);
      if (modRes.status === "fulfilled") {
        setModels(modRes.value);
        if (modRes.value.length >= 2) {
          setCkptA(modRes.value[0].filename);
          setCkptB(modRes.value[1].filename);
        }
      }
    });
  }, []);

  const handleRunCompare = async () => {
    if (!ckptA || !ckptB) return;
    setComparing(true);
    try {
      const res = await api.compareModels({
        checkpoint_a: ckptA,
        checkpoint_b: ckptB,
        num_samples: 3,
      });
      setCompareResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setComparing(false);
    }
  };

  const handleStartExperiment = async (configPath: string) => {
    setStarting(true);
    try {
      await api.startTraining({ config_path: configPath });
      alert("Training started successfully!");
    } catch (err: any) {
      alert("Failed to start training: " + err.message);
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="border-b border-border pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Experiments & Checkpoints</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Model checkpoint management, architecture profiles & counterfactual policy comparison
          </p>
        </div>
      </div>

      {/* Available Experiments */}
      <div>
        <h2 className="text-base font-semibold text-white mb-4">Configurations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {experiments.map((exp) => (
            <div
              key={exp.id}
              className="p-5 rounded-xl bg-surface border border-border flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-primary-500/10 text-primary-400 border border-primary-500/20">
                    {exp.model_type}
                  </span>
                  <span className="text-xs font-mono text-neutral-500">
                    Budget: {(exp.timesteps / 1000).toFixed(0)}k steps
                  </span>
                </div>
                <h3 className="text-base font-bold text-white tracking-tight">{exp.name}</h3>
                <p className="text-xs text-neutral-400 mt-1">{exp.description}</p>
                <div className="mt-3 flex items-center gap-2 text-[11px] font-mono text-neutral-500">
                  <FileCode className="w-3.5 h-3.5" />
                  <span>{exp.config_path}</span>
                </div>
              </div>

              <div className="mt-5 pt-4 border-t border-border-subtle flex justify-end">
                <button
                  onClick={() => handleStartExperiment(exp.config_path)}
                  disabled={starting}
                  className="px-4 py-2 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center gap-2 hover:bg-primary-500 transition-colors"
                >
                  <Play className="w-3.5 h-3.5 fill-current" /> Launch Experiment
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Checkpoints & Policy Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Checkpoint List */}
        <div className="lg:col-span-5 space-y-4">
          <h2 className="text-base font-semibold text-white">Saved Checkpoints</h2>
          <div className="bg-surface rounded-xl border border-border divide-y divide-border-subtle overflow-hidden max-h-96 overflow-y-auto">
            {models.length > 0 ? (
              models.map((m) => (
                <div key={m.model_id} className="p-4 flex items-center justify-between hover:bg-surface-hover/50">
                  <div>
                    <div className="text-xs font-mono font-bold text-white truncate w-48">
                      {m.filename}
                    </div>
                    <div className="text-[11px] font-mono text-neutral-500 mt-0.5 flex gap-2">
                      <span>Step {m.step.toLocaleString()}</span>
                      <span>•</span>
                      <span>{(m.size_bytes / (1024 * 1024)).toFixed(1)} MB</span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-300">
                    {m.tag}
                  </span>
                </div>
              ))
            ) : (
              <div className="p-6 text-center text-xs font-mono text-neutral-600">
                No checkpoints saved yet
              </div>
            )}
          </div>
        </div>

        {/* Policy Comparison Panel */}
        <div className="lg:col-span-7 space-y-4">
          <h2 className="text-base font-semibold text-white">Compare Checkpoint Policies</h2>
          <div className="p-5 bg-surface rounded-xl border border-border space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-mono text-neutral-400 block mb-1">
                  Checkpoint A (Baseline)
                </label>
                <select
                  value={ckptA}
                  onChange={(e) => setCkptA(e.target.value)}
                  className="w-full bg-[#181818] border border-border rounded-lg p-2 text-xs font-mono text-white"
                >
                  {models.map((m) => (
                    <option key={m.filename} value={m.filename}>
                      {m.filename}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-mono text-neutral-400 block mb-1">
                  Checkpoint B (Updated)
                </label>
                <select
                  value={ckptB}
                  onChange={(e) => setCkptB(e.target.value)}
                  className="w-full bg-[#181818] border border-border rounded-lg p-2 text-xs font-mono text-white"
                >
                  {models.map((m) => (
                    <option key={m.filename} value={m.filename}>
                      {m.filename}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={handleRunCompare}
              disabled={comparing || !ckptA || !ckptB}
              className="w-full py-2.5 rounded-lg bg-primary-600 text-white font-medium text-xs flex items-center justify-center gap-2 hover:bg-primary-500 disabled:opacity-50"
            >
              <ArrowRightLeft className="w-4 h-4" />
              {comparing ? "Evaluating Policies..." : "Run Policy Comparison"}
            </button>
          </div>

          {/* Comparison Results */}
          {compareResult && (
            <div className="space-y-4 pt-2">
              <h3 className="text-sm font-semibold text-white font-mono">
                Comparative Inferences on Real Observations
              </h3>
              {compareResult.samples.map((sample, sIdx) => (
                <div
                  key={sIdx}
                  className="p-5 rounded-xl bg-surface border border-border grid grid-cols-1 md:grid-cols-12 gap-4 items-center"
                >
                  <div className="md:col-span-3 flex justify-center">
                    {sample.frame_base64 && (
                      <img
                        src={`data:image/jpeg;base64,${sample.frame_base64}`}
                        alt="Test Observation"
                        className="w-24 h-24 rounded-lg pixelated border border-border"
                      />
                    )}
                  </div>
                  <div className="md:col-span-9 font-mono text-xs space-y-2">
                    <div className="grid grid-cols-2 gap-4 pb-2 border-b border-border-subtle">
                      <div>
                        <span className="text-neutral-500 block text-[10px]">CHECKPOINT A</span>
                        <span className="text-primary-400 font-bold uppercase">
                          {sample.top_action_a}
                        </span>
                        <span className="text-neutral-400 text-[11px] block mt-0.5">
                          Value: {sample.value_a.toFixed(2)} | Entropy: {sample.entropy_a.toFixed(2)}
                        </span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block text-[10px]">CHECKPOINT B</span>
                        <span className="text-emerald-400 font-bold uppercase">
                          {sample.top_action_b}
                        </span>
                        <span className="text-neutral-400 text-[11px] block mt-0.5">
                          Value: {sample.value_b.toFixed(2)} | Entropy: {sample.entropy_b.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


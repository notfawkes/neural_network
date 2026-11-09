"use client";

import React, { useEffect, useState, useRef } from "react";
import { GameView } from "@/components/live/GameView";
import { ActionProbChart } from "@/components/live/ActionProbChart";
import { RewardChart } from "@/components/live/RewardChart";
import { EventTimeline, TimelineEvent } from "@/components/live/EventTimeline";
import { MetricCard } from "@/components/ui/MetricCard";
import { LiveFrameData } from "@/types";
import { Activity, Flame, ShieldAlert, Cpu, Brain, Eye } from "lucide-react";

export default function LiveTrainingPage() {
  const [data, setData] = useState<LiveFrameData | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [rewardHistory, setRewardHistory] = useState<Array<{ step: number; reward: number }>>([]);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const frameQueue = useRef<LiveFrameData[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const connectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/live-game";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload: LiveFrameData = JSON.parse(event.data);
        if (isPaused) {
          // Store frames in buffer when paused for single-stepping
          if (frameQueue.current.length < 50) {
            frameQueue.current.push(payload);
          }
          return;
        }

        setData(payload);

        // Update reward sparkline
        setRewardHistory((prev) => {
          const next = [...prev, { step: payload.step, reward: payload.episode_reward }];
          return next.slice(-60);
        });

        // Add event if milestone reward or death
        if (payload.reward > 0.5) {
          setEvents((prev) => [
            {
              id: Math.random().toString(),
              timestamp: `Step ${payload.step}`,
              type: "achievement",
              title: "Achievement Milestone",
              description: `Agent unlocked +${payload.reward} reward!`,
            },
            ...prev.slice(0, 20),
          ]);
        }
      } catch (err) {
        console.error("Failed to parse websocket frame:", err);
      }
    };

    ws.onerror = (err) => console.log("Live WebSocket status:", err);
  };

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [isPaused]);

  // Single step forward handler
  const handleStepForward = () => {
    if (frameQueue.current.length > 0) {
      const nextFrame = frameQueue.current.shift()!;
      setData(nextFrame);
    }
  };

  // Replay / Reconnect handler
  const handleReplay = () => {
    frameQueue.current = [];
    connectWebSocket();
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Page Header */}
      <div className="border-b border-border pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Live Agent Observatory</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Real-time neural observation pipeline, internal latent representations & decision distributions
          </p>
        </div>
      </div>

      {/* Main Grid: Game View + Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Big Game View with PPO Cycle */}
        <div className="lg:col-span-5">
          <GameView
            data={data}
            isPaused={isPaused}
            onTogglePause={() => setIsPaused(!isPaused)}
            onStepForward={handleStepForward}
            onReplay={handleReplay}
            speed={speed}
            onChangeSpeed={setSpeed}
          />
        </div>

        {/* Right Column: Actions & Neural Telemetry */}
        <div className="lg:col-span-7 space-y-6">
          {/* Action Probability Distribution */}
          <ActionProbChart
            probs={data?.action_probs}
            selectedAction={data?.action}
          />

          {/* Key Decision Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard
              label="Selected Action"
              value={data?.action || "--"}
              subValue="Current discrete step"
              icon={<Activity className="w-4 h-4" />}
            />
            <MetricCard
              label="Instant Reward"
              value={(data?.reward || 0).toFixed(2)}
              trend={(data?.reward || 0) >= 0 ? "up" : "down"}
              subValue="Step environment delta"
              icon={<Flame className="w-4 h-4" />}
            />
            <MetricCard
              label="Critic Value V(s)"
              value={(data?.critic_value || 0).toFixed(3)}
              subValue="State return prediction"
              icon={<ShieldAlert className="w-4 h-4" />}
            />
            <MetricCard
              label="LSTM Hidden Norm"
              value={(data?.lstm_h_norm || 0).toFixed(2)}
              subValue="Recurrent state norm"
              icon={<Cpu className="w-4 h-4" />}
            />
          </div>

          {/* Neural Representation Inspection (CNN & LSTM Layers) */}
          <div className="p-5 rounded-xl bg-surface border border-border space-y-4">
            <div className="flex items-center justify-between border-b border-border-subtle pb-3">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-primary-400" />
                <h3 className="text-sm font-semibold text-white">Neural Representation Inspector</h3>
              </div>
              <span className="text-[11px] font-mono text-neutral-400">
                CNN(512) ➔ LSTM(256) ➔ Actor(17) & Critic(1)
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* CNN Visual Feature Vector (First 16 channels) */}
              <div className="p-3 bg-[#131313] rounded-lg border border-border-subtle">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5 text-xs font-mono text-neutral-300">
                    <Eye className="w-3.5 h-3.5 text-emerald-400" />
                    <span>CNN Feature Activations</span>
                  </div>
                  <span className="text-[10px] font-mono text-neutral-500">
                    ‖z‖ = {(data?.cnn_norm || 1.0).toFixed(2)}
                  </span>
                </div>
                <div className="grid grid-cols-16 gap-1 h-12 items-end pt-1" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
                  {(data?.cnn_sample_vec && data.cnn_sample_vec.length >= 16
                    ? data.cnn_sample_vec
                    : [0.2, 0.5, 0.8, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7, 0.2, 0.4, 0.9, 0.5, 0.3, 0.8, 0.6]
                  ).map((val, idx) => (
                    <div
                      key={idx}
                      className="bg-emerald-500/80 rounded-t w-full hover:bg-emerald-400 transition-all"
                      style={{ height: `${Math.min(100, Math.max(10, Math.round(Math.abs(val) * 60)))}%` }}
                      title={`Filter ${idx}: ${val.toFixed(3)}`}
                    />
                  ))}
                </div>
              </div>

              {/* LSTM Memory State (First 16 hidden dimensions) */}
              <div className="p-3 bg-[#131313] rounded-lg border border-border-subtle">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5 text-xs font-mono text-neutral-300">
                    <Cpu className="w-3.5 h-3.5 text-primary-400" />
                    <span>LSTM Working Memory (h)</span>
                  </div>
                  <span className="text-[10px] font-mono text-neutral-500">
                    ‖h‖ = {(data?.lstm_h_norm || 1.0).toFixed(2)}
                  </span>
                </div>
                <div className="grid grid-cols-16 gap-1 h-12 items-end pt-1" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
                  {(data?.lstm_sample_vec && data.lstm_sample_vec.length >= 16
                    ? data.lstm_sample_vec
                    : [0.1, -0.3, 0.6, 0.4, -0.5, 0.8, -0.2, 0.3, 0.7, -0.4, 0.5, 0.2, -0.6, 0.3, 0.8, -0.1]
                  ).map((val, idx) => (
                    <div
                      key={idx}
                      className="bg-primary-500/80 rounded-t w-full hover:bg-primary-400 transition-all"
                      style={{ height: `${Math.min(100, Math.max(10, Math.round(Math.abs(val) * 70)))}%` }}
                      title={`Hidden Dim ${idx}: ${val.toFixed(3)}`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row: Charts & Timeline */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <RewardChart data={rewardHistory} />
        <EventTimeline events={events} />
      </div>
    </div>
  );
}

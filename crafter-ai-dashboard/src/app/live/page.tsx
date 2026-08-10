"use client";

import React, { useEffect, useState, useRef } from "react";
import { GameView } from "@/components/live/GameView";
import { ActionProbChart } from "@/components/live/ActionProbChart";
import { RewardChart } from "@/components/live/RewardChart";
import { EventTimeline, TimelineEvent } from "@/components/live/EventTimeline";
import { MetricCard } from "@/components/ui/MetricCard";
import { LiveFrameData } from "@/types";
import { Activity, Flame, ShieldAlert, Cpu } from "lucide-react";

export default function LiveTrainingPage() {
  const [data, setData] = useState<LiveFrameData | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [rewardHistory, setRewardHistory] = useState<Array<{ step: number; reward: number }>>([]);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/live-game";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      if (isPaused) return;
      try {
        const payload: LiveFrameData = JSON.parse(event.data);
        setData(payload);

        // Update reward sparkline
        setRewardHistory((prev) => {
          const next = [...prev, { step: payload.step, reward: payload.episode_reward }];
          return next.slice(-50);
        });

        // Add event if reward change or death
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

    return () => {
      ws.close();
    };
  }, [isPaused]);

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Page Header */}
      <div className="border-b border-border pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Live Agent POV</h1>
          <p className="text-sm text-neutral-400 mt-1">
            Real-time neural observation pipeline & decision distributions
          </p>
        </div>
      </div>

      {/* Main Grid: Game View + Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Big Game View */}
        <div className="lg:col-span-5">
          <GameView
            data={data}
            isPaused={isPaused}
            onTogglePause={() => setIsPaused(!isPaused)}
            speed={speed}
            onChangeSpeed={setSpeed}
          />
        </div>

        {/* Right Column: Actions & Network stats */}
        <div className="lg:col-span-7 space-y-6">
          <ActionProbChart
            probs={data?.action_probs}
            selectedAction={data?.action}
          />

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard
              label="Selected Action"
              value={data?.action || "--"}
              subValue="Current decision"
              icon={<Activity className="w-4 h-4" />}
            />
            <MetricCard
              label="Instant Reward"
              value={(data?.reward || 0).toFixed(2)}
              trend={(data?.reward || 0) >= 0 ? "up" : "down"}
              subValue="Step delta"
              icon={<Flame className="w-4 h-4" />}
            />
            <MetricCard
              label="Critic Value"
              value={(data?.critic_value || 0).toFixed(3)}
              subValue="State evaluation"
              icon={<ShieldAlert className="w-4 h-4" />}
            />
            <MetricCard
              label="LSTM h-norm"
              value={(data?.lstm_h_norm || 0).toFixed(2)}
              subValue="Hidden state norm"
              icon={<Cpu className="w-4 h-4" />}
            />
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

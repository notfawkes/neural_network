"use client";

import React, { useState } from "react";
import { Play, Pause, RotateCcw, FastForward, Activity } from "lucide-react";
import { LiveFrameData } from "@/types";

interface GameViewProps {
  data: LiveFrameData | null;
  isPaused: boolean;
  onTogglePause: () => void;
  speed: number;
  onChangeSpeed: (speed: number) => void;
}

export function GameView({
  data,
  isPaused,
  onTogglePause,
  speed,
  onChangeSpeed,
}: GameViewProps) {
  return (
    <div className="flex flex-col items-center justify-center p-6 bg-surface-card rounded-2xl border border-border">
      {/* Viewer Header */}
      <div className="w-full flex items-center justify-between mb-4 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="text-neutral-500">EPISODE</span>
          <span className="text-white font-semibold">{data?.episode_id ?? "--"}</span>
          <span className="text-neutral-600">/</span>
          <span className="text-neutral-500">STEP</span>
          <span className="text-white font-semibold">{data?.step ?? "--"}</span>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              data?.is_alive ? "bg-emerald-400 animate-pulse" : "bg-rose-500"
            }`}
          />
          <span className={data?.is_alive ? "text-emerald-400" : "text-rose-400"}>
            {data?.is_alive ? "AGENT ACTIVE" : "EPISODE ENDED"}
          </span>
        </div>
      </div>

      {/* Screen Container (64x64 scaled up to 320x320 with pixelated rendering) */}
      <div className="relative w-80 h-80 rounded-xl overflow-hidden bg-black border-2 border-neutral-800 shadow-2xl flex items-center justify-center group">
        {data?.frame ? (
          <img
            src={`data:image/jpeg;base64,${data.frame}`}
            alt="Crafter Observation"
            className="w-full h-full object-contain pixelated"
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-neutral-600">
            <Activity className="w-10 h-10 mb-2 stroke-1 opacity-50" />
            <p className="text-xs font-mono">WAITING FOR OBSERVATIONS</p>
          </div>
        )}

        {/* Action Overlay Banner */}
        {data?.action && (
          <div className="absolute bottom-2 left-2 right-2 bg-black/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 flex items-center justify-between text-xs font-mono">
            <span className="text-neutral-400">ACTION:</span>
            <span className="text-primary-400 font-bold uppercase">{data.action}</span>
          </div>
        )}

        {isPaused && (
          <div className="absolute inset-0 bg-black/50 backdrop-blur-[2px] flex items-center justify-center">
            <div className="px-3 py-1.5 rounded-full bg-white/10 text-white text-xs font-mono uppercase tracking-wider">
              PAUSED
            </div>
          </div>
        )}
      </div>

      {/* Playback Controls */}
      <div className="w-full flex items-center justify-between mt-5 pt-4 border-t border-border-subtle">
        <div className="flex items-center gap-2">
          <button
            onClick={onTogglePause}
            className="p-2 rounded-lg bg-surface border border-border text-neutral-300 hover:text-white hover:border-neutral-700 transition-colors"
            title={isPaused ? "Play" : "Pause"}
          >
            {isPaused ? <Play className="w-4 h-4 fill-current" /> : <Pause className="w-4 h-4 fill-current" />}
          </button>
        </div>

        <div className="flex items-center gap-1 bg-surface px-2 py-1 rounded-lg border border-border text-xs font-mono">
          <span className="text-neutral-500 mr-1">SPEED:</span>
          {[1, 2, 5].map((s) => (
            <button
              key={s}
              onClick={() => onChangeSpeed(s)}
              className={`px-2 py-0.5 rounded ${
                speed === s ? "bg-primary-600 text-white" : "text-neutral-400 hover:text-white"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}


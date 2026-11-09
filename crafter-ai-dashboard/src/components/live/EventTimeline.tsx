"use client";

import React from "react";
import { CircleDot, Award, Zap, AlertTriangle } from "lucide-react";

export interface TimelineEvent {
  id: string;
  timestamp: string;
  type: "achievement" | "reward" | "death" | "update";
  title: string;
  description: string;
}

interface EventTimelineProps {
  events: TimelineEvent[];
}

export function EventTimeline({ events }: EventTimelineProps) {
  return (
    <div className="bg-surface rounded-xl border border-border p-5">
      <h4 className="text-xs font-mono uppercase tracking-wider text-neutral-400 mb-4">
        Recent Timeline
      </h4>

      <div className="space-y-4 max-h-56 overflow-y-auto pr-2">
        {events.length > 0 ? (
          events.map((ev) => (
            <div key={ev.id} className="flex items-start gap-3 text-xs">
              <div className="mt-0.5">
                {ev.type === "achievement" && <Award className="w-3.5 h-3.5 text-amber-400" />}
                {ev.type === "reward" && <Zap className="w-3.5 h-3.5 text-emerald-400" />}
                {ev.type === "death" && <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />}
                {ev.type === "update" && <CircleDot className="w-3.5 h-3.5 text-primary-400" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-neutral-200 truncate">{ev.title}</span>
                  <span className="text-[10px] font-mono text-neutral-500">{ev.timestamp}</span>
                </div>
                <p className="text-neutral-400 text-[11px] mt-0.5">{ev.description}</p>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-6 text-neutral-500 text-xs font-mono">
            No live events recorded yet
          </div>
        )}
      </div>
    </div>
  );
}


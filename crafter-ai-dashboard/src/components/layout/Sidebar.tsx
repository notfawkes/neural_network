"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  PlayCircle,
  Film,
  LineChart,
  Cpu,
  Trophy,
  FlaskConical,
  Terminal,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/live", label: "Live Training", icon: PlayCircle },
  { href: "/episodes", label: "Episode Explorer", icon: Film },
  { href: "/analysis", label: "Learning Analysis", icon: LineChart },
  { href: "/network", label: "Neural Network", icon: Cpu },
  { href: "/achievements", label: "Achievements", icon: Trophy },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-[#0d0d0d] flex flex-col justify-between shrink-0 h-screen sticky top-0 select-none">
      <div>
        {/* Brand */}
        <div className="h-16 px-6 flex items-center gap-3 border-b border-border">
          <div className="w-7 h-7 rounded-lg bg-primary-600 flex items-center justify-center font-mono font-bold text-white text-xs shadow-inner">
            CR
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight text-white flex items-center gap-1.5">
              CRAfter AI
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary-500/10 text-primary-400 border border-primary-500/20">
                v0.1
              </span>
            </div>
            <p className="text-[11px] text-neutral-500 font-mono tracking-tight">Visual RL Lab</p>
          </div>
        </div>

        {/* Navigation */}
        <div className="p-3 space-y-1">
          <div className="px-3 py-2 text-[10px] font-mono font-medium uppercase tracking-wider text-neutral-500">
            Platform
          </div>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group",
                  isActive
                    ? "bg-surface-hover text-white border border-neutral-700/60 shadow-sm"
                    : "text-neutral-400 hover:text-neutral-200 hover:bg-surface/60"
                )}
              >
                <Icon
                  className={cn(
                    "w-4 h-4 transition-colors",
                    isActive ? "text-primary-400" : "text-neutral-500 group-hover:text-neutral-300"
                  )}
                />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-border">
        <div className="p-3 rounded-lg bg-surface border border-border-subtle flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <div className="text-xs font-mono text-neutral-400 truncate">
            Engine: PyTorch MPS
          </div>
        </div>
      </div>
    </aside>
  );
}


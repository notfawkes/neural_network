import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number, decimals = 1): string {
  if (num === undefined || num === null || isNaN(num)) return "0";
  if (Math.abs(num) >= 1_000_000) {
    return (num / 1_000_000).toFixed(decimals) + "M";
  }
  if (Math.abs(num) >= 1_000) {
    return (num / 1_000).toFixed(decimals) + "k";
  }
  return num.toFixed(decimals);
}

export function formatTime(seconds: number): string {
  if (!seconds || isNaN(seconds)) return "0s";
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

export const CRAFTER_ACTIONS = [
  "noop", "move_left", "move_right", "move_up", "move_down",
  "do", "sleep",
  "place_stone", "place_table", "place_furnace", "place_plant",
  "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
  "make_wood_sword", "make_stone_sword", "make_iron_sword",
];


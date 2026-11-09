# Dashboard Import Graph & Module Map

This document describes component and library import relationships across `crafter-ai-dashboard`.

---

## Core Dependency Matrix

- `src/types/index.ts`: Shared TypeScript schemas used across all pages and components.
- `src/lib/api.ts`: API service layer mapping frontend requests to FastAPI backend endpoints.
- `src/lib/utils.ts`: Helper utilities including `cn` (Tailwind class merging), `formatNumber`, `formatTime`, and `CRAFTER_ACTIONS`.
- `src/components/ui/`: Reusable primitive components (`StatusBadge`, `MetricCard`, `Card`).
- `src/components/live/`: Real-time streaming components (`GameView`, `ActionProbChart`, `RewardChart`, `EventTimeline`).
- `src/components/episodes/`: Replay and comparison components (`EpisodeTable`, `EpisodeReplay`, `EpisodeCompare`).
- `src/components/analysis/`: Telemetry graph wrappers (`MetricsChart`).
- `src/components/network/`: Layer topology visualization (`ArchDiagram`).
- `src/components/achievements/`: Milestone tracking cards (`AchievementCard`).


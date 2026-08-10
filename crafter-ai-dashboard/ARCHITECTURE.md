# Dashboard Architecture Specification

Detailed frontend architecture, state synchronization, and component hierarchy of `crafter-ai-dashboard`.

---

## Technical Stack

- **Framework**: Next.js 14 App Router
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS (dark mode theme inspired by Linear and Vercel)
- **Data Visualization**: Recharts
- **Icons**: Lucide React
- **Transport**: Fetch API (REST) & native browser WebSocket

---

## State & Real-Time Data Flow

```
FastAPI Server (Port 8000)
       │
       ├── REST Endpoints (cache: "no-store")
       │         │
       │         ▼
       │   React SWR / Polling Hooks (Overview, Episodes, Analysis)
       │
       └── WebSocket Streams
                 │
                 ├──► /ws/training  ──► Telemetry update state
                 └──► /ws/live-game  ──► HTML5 Canvas / Pixelated Image Buffer
```

---

## Layout Hierarchy

```
RootLayout (src/app/layout.tsx)
  ├── Sidebar Navigation (src/components/layout/Sidebar.tsx)
  └── Main Content Area
        ├── Page 1: Overview (/)
        ├── Page 2: Live Training (/live)
        │     ├── GameView (Pixelated 64x64 frame display)
        │     ├── ActionProbChart (17-action vertical bars)
        │     ├── RewardChart (Live returns)
        │     └── EventTimeline (Chronological occurrences)
        ├── Page 3: Episodes (/episodes)
        │     ├── EpisodeTable
        │     ├── EpisodeReplay (Scrubber & frame inspector)
        │     └── EpisodeCompare (Side-by-side milestones)
        ├── Page 4: Analysis (/analysis)
        │     └── MetricsChart (Policy loss, Value loss, Entropy, KL)
        ├── Page 5: Network (/network)
        │     └── ArchDiagram (Interactive layer graph)
        ├── Page 6: Achievements (/achievements)
        │     └── AchievementCard (22 milestones)
        └── Page 7: Experiments (/experiments)
              └── Checkpoint comparison inference
```

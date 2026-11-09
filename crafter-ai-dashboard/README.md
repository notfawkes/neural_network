# CRAfter AI Dashboard

Next.js 14 web application providing real-time visual interpretability for the CRAfter AI reinforcement learning lab.

---

## Key Pages

1. **`/` (Overview)**: Real-time telemetry, step counters, sliding reward sparkline, top achievement unlock rates, quick training controls.
2. **`/live` (Live Agent POV)**: High-resolution pixelated 64×64 observation playback, 17-action categorical distribution bars, critic values, hidden state norms, and timeline events.
3. **`/episodes` (Episode Explorer)**: Searchable list of gameplay sessions with full episode scrubber playback and side-by-side milestone comparisons.
4. **`/analysis` (Learning Analysis)**: Comprehensive convergence charts for policy loss, value loss, entropy, approximate KL divergence, and achievement success spectrum.
5. **`/network` (Neural Network Inspector)**: Computational graph showing CNN layer dimensions, LSTM memory states, and linear head projections.
6. **`/achievements` (Achievements)**: Detailed cards for all 22 Crafter milestones with category filters and Crafter Score computation.
7. **`/experiments` (Experiments & Models)**: Launch configuration profiles, browse checkpoints, and run counterfactual policy comparisons.

---

## Running the Dashboard

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).


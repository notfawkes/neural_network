# Engine Import Dependency Graph

This document details the module dependencies and import chains across `crafter-ai-engine`.

---

## High-Level Dependency Hierarchy

```
crafter-game (base environment)
      │
      ▼
crafter_ai.env.adapter (CrafterEnv, VecCrafterEnv)
      │
      ├──► crafter_ai.models.cnn (CrafterCNN)
      │          │
      │          ▼
      ├──► crafter_ai.models.actor_critic (ActorCritic, RecurrentActorCritic)
      │          │
      │          ▼
      ├──► crafter_ai.training.rollout_buffer (RolloutBuffer)
      │          │
      │          ▼
      └──► crafter_ai.training.ppo / ppo_recurrent (PPOTrainer, RecurrentPPOTrainer)
                 │
                 ├──► crafter_ai.recording (EpisodeRecorder, RecordingStorage)
                 ├──► crafter_ai.evaluation (Evaluator)
                 └──► crafter_ai.utils (CheckpointManager, MetricsTracker, device)
```

---

## File-by-File Import Directory

### `crafter_ai.env.adapter`
- Imports: `numpy`, `typing`, `crafter` (from `crafter-game`)
- Exports: `CrafterEnv`, `VecCrafterEnv`, `make_env`, `ACTION_NAMES`, `ACHIEVEMENT_NAMES`

### `crafter_ai.models.cnn`
- Imports: `torch`, `torch.nn`
- Exports: `CrafterCNN`

### `crafter_ai.models.actor_critic`
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `crafter_ai.models.cnn.CrafterCNN`
- Exports: `ActorCritic`, `RecurrentActorCritic`, `LSTMState`

### `crafter_ai.training.rollout_buffer`
- Imports: `torch`, `numpy`, `typing`
- Exports: `RolloutBuffer`

### `crafter_ai.training.ppo`
- Imports: `torch`, `torch.nn`, `torch.optim`, `numpy`, `crafter_ai.models.actor_critic.ActorCritic`, `crafter_ai.env.adapter.VecCrafterEnv`, `crafter_ai.training.rollout_buffer.RolloutBuffer`, `crafter_ai.utils.checkpoint.CheckpointManager`
- Exports: `PPOTrainer`

### `crafter_ai.training.ppo_recurrent`
- Imports: `torch`, `torch.nn`, `torch.optim`, `numpy`, `crafter_ai.models.actor_critic.RecurrentActorCritic`, `crafter_ai.env.adapter.VecCrafterEnv`, `crafter_ai.training.rollout_buffer.RolloutBuffer`, `crafter_ai.utils.checkpoint.CheckpointManager`
- Exports: `RecurrentPPOTrainer`


export interface TrainingStatus {
  status: "idle" | "starting" | "training" | "stopped" | "error";
  is_running: boolean;
  pid?: number | null;
  global_step: number;
  episode_count: number;
  update_count: number;
  mean_reward: number;
  mean_ep_length: number;
  crafter_score: number;
  fps: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  approx_kl: number;
  elapsed_seconds: number;
  experiment_name: string;
  latest_checkpoint?: string | null;
  last_update?: number | null;
}

export interface MetricsData {
  experiment_name: string;
  total_steps: number;
  total_episodes: number;
  crafter_score: number;
  episode_stats: Record<string, number>;
  achievement_rates: Record<string, number>;
  update_stats: Record<string, number>;
  recent_rewards: number[];
  recent_lengths: number[];
  recent_losses: Array<Record<string, number>>;
}

export interface EpisodeSummary {
  episode_id: number;
  global_step: number;
  total_reward: number;
  length: number;
  terminated: boolean;
  truncated: boolean;
  num_achievements: number;
  achievements: Record<string, boolean>;
  duration_seconds: number;
  experiment_name: string;
  model_type: string;
  has_recording: boolean;
}

export interface EpisodeDetail extends EpisodeSummary {
  actions: number[];
  rewards: number[];
  player_positions: Array<[number, number]>;
  critic_values: Array<number | null>;
  action_probs: Array<number[] | null>;
  lstm_h_norms: Array<number | null>;
}

export interface CheckpointInfo {
  model_id: string;
  filename: string;
  path: string;
  step: number;
  tag: string;
  episode_count: number;
  model_type: string;
  size_bytes: number;
  created_at: number;
}

export interface ActionComparison {
  action_index: number;
  action_name: string;
  prob_a: number;
  prob_b: number;
  delta: number;
}

export interface SampleComparison {
  sample_index: number;
  frame_base64?: string | null;
  value_a: number;
  value_b: number;
  entropy_a: number;
  entropy_b: number;
  top_action_a: string;
  top_action_b: string;
  actions: ActionComparison[];
}

export interface CompareResponse {
  model_a_info: Record<string, any>;
  model_b_info: Record<string, any>;
  samples: SampleComparison[];
}

export interface AchievementItem {
  name: string;
  title: string;
  category: "mining" | "combat" | "survival" | "crafting" | "building" | "farming";
  icon: string;
  success_rate: number;
  unlocked: boolean;
}

export interface AchievementsResponse {
  achievements: AchievementItem[];
  crafter_score: number;
  total_episodes: number;
}

export interface ExperimentInfo {
  id: string;
  name: string;
  description: string;
  model_type: string;
  config_path: string;
  timesteps: number;
}

export interface LiveFrameData {
  frame?: string | null;
  episode_id: number;
  step: number;
  reward: number;
  episode_reward: number;
  action: string;
  action_probs: number[];
  critic_value: number;
  is_alive: boolean;
  lstm_h_norm: number;
}

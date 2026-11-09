const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
      cache: "no-store",
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`API Error ${res.status}: ${err}`);
    }
    return res.json();
  } catch (err: any) {
    console.error(`Failed to fetch ${url}:`, err);
    throw err;
  }
}

export const api = {
  // Training
  getTrainingStatus: () => fetchJson<any>("/api/training/status"),
  getMetrics: () => fetchJson<any>("/api/training/metrics"),
  startTraining: (data: { config_path?: string; resume_checkpoint?: string | null; total_timesteps?: number }) =>
    fetchJson<any>("/api/training/start", { method: "POST", body: JSON.stringify(data) }),
  stopTraining: () => fetchJson<any>("/api/training/stop", { method: "POST" }),
  getLogs: () => fetchJson<{ logs: string[] }>("/api/training/logs"),

  // Episodes
  getEpisodes: (limit = 50, offset = 0) =>
    fetchJson<any[]>(`/api/episodes?limit=${limit}&offset=${offset}`),
  getEpisode: (id: number) => fetchJson<any>(`/api/episodes/${id}`),
  getEpisodeFrames: (id: number, scale = 4) =>
    fetchJson<{ frames: string[]; total_frames: number }>(`/api/episodes/${id}/frames?scale=${scale}`),

  // Models
  getModels: () => fetchJson<any[]>("/api/models"),
  getModelDetail: (id: string) => fetchJson<any>(`/api/models/${id}`),
  compareModels: (data: { checkpoint_a: string; checkpoint_b: string; obs_source_episode_id?: number | null; num_samples?: number }) =>
    fetchJson<any>("/api/models/compare", { method: "POST", body: JSON.stringify(data) }),

  // Achievements & Experiments
  getAchievements: () => fetchJson<any>("/api/achievements"),
  getExperiments: () => fetchJson<any[]>("/api/experiments"),

  // Publication Figures
  getFigures: () => fetchJson<any[]>("/api/figures"),
  generateFigures: () => fetchJson<any>("/api/figures/generate", { method: "POST" }),
};


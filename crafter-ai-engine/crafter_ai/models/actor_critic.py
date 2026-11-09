"""
Actor-Critic Models
===================

Two variants:

1. ActorCritic (MLP-based):
   CNN → 512 → Actor head (17 logits) + Critic head (1 value)

2. RecurrentActorCritic (LSTM-based):
   CNN → 512 → LSTM(256) → Actor head (17 logits) + Critic head (1 value)

Both share the CNN feature extractor. Actor and Critic have separate heads.
The LSTM maintains hidden/cell states between timesteps.
Recurrent states are reset at episode boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .cnn import CrafterCNN

NUM_ACTIONS = 17
CNN_OUT = 512
LSTM_HIDDEN = 256


class ActorCritic(nn.Module):
    """
    CNN + MLP Actor-Critic (baseline, no recurrence).

    Architecture:
      obs → CNN(512) → Actor: Linear(512→17)
                     → Critic: Linear(512→1)

    Parameters
    ----------
    in_channels : int
        CNN input channels.
    num_actions : int
        Size of the action space.
    cnn_out : int
        CNN output feature dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = NUM_ACTIONS,
        cnn_out: int = CNN_OUT,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.cnn_out = cnn_out

        # Shared CNN encoder
        self.cnn = CrafterCNN(in_channels=in_channels, out_features=cnn_out)

        # Actor head
        self.actor = nn.Linear(cnn_out, num_actions)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head
        self.critic = nn.Linear(cnn_out, 1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (B, C, H, W) float32 in [0, 1]

        Returns
        -------
        logits : (B, 17) — raw action logits
        values : (B, 1)  — critic state values
        features : (B, 512) — CNN features (for inspection)
        """
        features = self.cnn(obs)
        logits = self.actor(features)
        values = self.critic(features)
        return logits, values, features

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, value for a batch of observations.

        Parameters
        ----------
        obs : (B, C, H, W)
        action : (B,) optional — if provided, compute log_prob for this action

        Returns
        -------
        action : (B,)
        log_prob : (B,)
        entropy : (B,)
        value : (B, 1)
        probs : (B, 17) — softmax probabilities (for visualization)
        """
        logits, values, features = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        probs = F.softmax(logits, dim=-1)

        return action, log_prob, entropy, values, probs

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get critic value only."""
        features = self.cnn(obs)
        return self.critic(features)

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Get CNN features only."""
        return self.cnn(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent Actor-Critic (CNN + LSTM)
# ─────────────────────────────────────────────────────────────────────────────

LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c) each (num_layers, B, hidden)


class RecurrentActorCritic(nn.Module):
    """
    CNN + LSTM + Actor-Critic.

    Architecture:
      obs → CNN(512) → LSTM(256) → Actor: Linear(256→17)
                                  → Critic: Linear(256→1)

    LSTM hidden state is maintained between timesteps.
    At episode boundaries (done=True), the hidden state for that environment
    is zeroed out.

    Parameters
    ----------
    in_channels : int
        CNN input channels.
    num_actions : int
        Size of the action space.
    cnn_out : int
        CNN output feature dimension.
    lstm_hidden : int
        LSTM hidden state size.
    lstm_layers : int
        Number of LSTM layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = NUM_ACTIONS,
        cnn_out: int = CNN_OUT,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.cnn_out = cnn_out
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Shared CNN encoder
        self.cnn = CrafterCNN(in_channels=in_channels, out_features=cnn_out)

        # Recurrent core
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,  # We use (T, B, input_size) format
        )

        # Actor head
        self.actor = nn.Linear(lstm_hidden, num_actions)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head
        self.critic = nn.Linear(lstm_hidden, 1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> LSTMState:
        """Create zero initial hidden/cell state."""
        h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return h, c

    def forward_sequence(
        self,
        obs: torch.Tensor,
        done: torch.Tensor,
        lstm_state: LSTMState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LSTMState]:
        """
        Process a sequence of observations through CNN + LSTM.

        This handles episode boundaries correctly: when done[t] is True for
        an environment, its hidden state is zeroed before processing step t+1.

        Parameters
        ----------
        obs : (T, B, C, H, W) float32
            Sequence of observations. T=timesteps, B=batch (environments).
        done : (T, B) bool
            Whether each step was a terminal step. Used to reset LSTM state.
        lstm_state : LSTMState
            Initial (h, c) for the sequence.

        Returns
        -------
        logits : (T, B, num_actions)
        values : (T, B, 1)
        lstm_out : (T, B, lstm_hidden)
        new_lstm_state : LSTMState
        """
        T, B, C, H, W = obs.shape

        # Process all obs through CNN: (T*B, C, H, W) → (T*B, cnn_out)
        obs_flat = obs.reshape(T * B, C, H, W)
        cnn_out = self.cnn(obs_flat)  # (T*B, 512)
        cnn_out = cnn_out.reshape(T, B, self.cnn_out)  # (T, B, 512)

        # Process through LSTM step-by-step to handle episode boundaries
        h, c = lstm_state
        lstm_outputs = []

        for t in range(T):
            # Reset hidden states for environments that just ended an episode
            # done[t] is True means THIS step was the first step of a new episode
            # (after autoreset). We zero the state before passing it.
            if t > 0:
                # done[t-1] tells us if the PREVIOUS step was terminal
                reset_mask = done[t - 1].float()  # (B,)
                # Expand mask: (1, B, 1) for broadcasting with (layers, B, hidden)
                reset_mask = reset_mask.unsqueeze(0).unsqueeze(-1)
                h = h * (1.0 - reset_mask)
                c = c * (1.0 - reset_mask)

            # LSTM expects (seq_len=1, batch, input) → output (1, B, hidden)
            x_t = cnn_out[t].unsqueeze(0)  # (1, B, 512)
            lstm_out_t, (h, c) = self.lstm(x_t, (h, c))
            lstm_outputs.append(lstm_out_t.squeeze(0))  # (B, hidden)

        lstm_out = torch.stack(lstm_outputs, dim=0)  # (T, B, hidden)

        # Apply actor and critic heads
        lstm_flat = lstm_out.view(T * B, self.lstm_hidden)
        logits = self.actor(lstm_flat).view(T, B, self.num_actions)
        values = self.critic(lstm_flat).view(T, B, 1)

        return logits, values, lstm_out, (h, c)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        done: torch.Tensor,
        lstm_state: LSTMState,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LSTMState]:
        """
        Get action, log_prob, entropy, value for a single timestep.

        Parameters
        ----------
        obs : (B, C, H, W)
        done : (B,) bool — whether prev step was terminal
        lstm_state : LSTMState — current hidden state
        action : (B,) optional

        Returns
        -------
        action : (B,)
        log_prob : (B,)
        entropy : (B,)
        value : (B, 1)
        probs : (B, 17)
        new_lstm_state : LSTMState
        """
        # Add time dimension: (1, B, C, H, W)
        obs_seq = obs.unsqueeze(0)
        done_seq = done.unsqueeze(0)  # (1, B)

        logits, values, _, new_state = self.forward_sequence(obs_seq, done_seq, lstm_state)

        # Remove time dimension
        logits = logits.squeeze(0)   # (B, 17)
        values = values.squeeze(0)   # (B, 1)

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        probs = F.softmax(logits, dim=-1)

        return action, log_prob, entropy, values, probs, new_state

    def get_value(
        self,
        obs: torch.Tensor,
        done: torch.Tensor,
        lstm_state: LSTMState,
    ) -> Tuple[torch.Tensor, LSTMState]:
        """Get critic value for the last observation in a sequence."""
        obs_seq = obs.unsqueeze(0)
        done_seq = done.unsqueeze(0)
        _, values, _, new_state = self.forward_sequence(obs_seq, done_seq, lstm_state)
        return values.squeeze(0), new_state

    def get_lstm_stats(self, lstm_state: LSTMState) -> Dict[str, float]:
        """Return statistics about the current LSTM hidden state."""
        h, c = lstm_state
        return {
            "h_norm": float(h.norm().item()),
            "c_norm": float(c.norm().item()),
            "h_mean": float(h.mean().item()),
            "c_mean": float(c.mean().item()),
            "h_std": float(h.std().item()),
            "c_std": float(c.std().item()),
        }

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Get CNN feature embeddings (512-d)."""
        return self.cnn(obs)

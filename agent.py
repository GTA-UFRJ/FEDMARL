import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE


class PrioritizedReplayJoint:
    """
    Prioritized Experience Replay (PER) buffer for joint transitions of N agents.

    Stores transitions as (obs, act, r, obs2, done) where obs and obs2 have shape
    (N, d_in), enabling sampling proportional to temporal difference error.

    Args:
        capacity: maximum buffer size
        n_agents: number of agents
        d_in:     local observation dimension per agent
        alpha:    prioritization exponent (0 = uniform, 1 = fully prioritized)
        eps:      numerical stability constant for priorities
        seed:     random seed for reproducibility
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        d_in: int,
        alpha: float = 0.6,
        eps: float = 1e-3,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.obs  = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.act  = np.zeros((capacity, n_agents), dtype=np.uint8)
        self.r    = np.zeros((capacity,), dtype=np.float32)
        self.obs2 = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self.p = np.zeros((capacity,), dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.max_p = 1.0

        self.rng = np.random.default_rng(int(seed))

    def add(self, obs, act, r, obs2, done: bool):
        """
        Adds a transition to the buffer with maximum priority.

        Args:
            obs:  joint observations, shape (N, d_in)
            act:  joint actions, shape (N,)
            r:    scalar global reward
            obs2: next joint observations, shape (N, d_in)
            done: episode termination flag
        """

        self.obs[self.ptr] = obs.astype(np.float32)
        self.act[self.ptr] = act.astype(np.uint8)
        self.r[self.ptr] = float(r)
        self.obs2[self.ptr] = obs2.astype(np.float32)
        self.done[self.ptr] = 1.0 if done else 0.0

        self.p[self.ptr] = self.max_p
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Samples a batch of transitions weighted by priorities.

        Args:
            batch_size: number of transitions to sample
            beta:       importance sampling correction exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple (obs, act, r, obs2, done, idx, weights) where weights are importance sampling weights
        """
        
        bs = min(int(batch_size), self.n)
        assert bs > 0

        pri = self.p[: self.n].astype(np.float64)
        probs = (pri + self.eps) ** self.alpha
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)

        idx = self.rng.choice(self.n, size=bs, replace=False, p=probs)
        w = (self.n * probs[idx]) ** (-beta)
        w = w / (w.max() + 1e-12)

        return (
            self.obs[idx],
            self.act[idx],
            self.r[idx],
            self.obs2[idx],
            self.done[idx],
            idx.astype(np.int64),
            w.astype(np.float32),
        )

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray):
        """
        Updates transition priorities after a training step.

        Args:
            idx:    indices of the sampled transitions
            td_abs: corresponding absolute temporal difference errors
        """

        td_abs = np.asarray(td_abs, dtype=np.float32)
        self.p[idx] = td_abs + self.eps
        self.max_p = float(max(self.max_p, float(td_abs.max(initial=0.0))))



class AgentMLP(nn.Module):
    """
    Individual Q-network for each agent — two-hidden-layer MLP.

    Args:
        d_in:   local observation dimension
        hidden: number of neurons in hidden layers

    Input:  tensor (B, d_in)
    Output: tensor (B, 2) with Q(o, 0) and Q(o, 1)
    """

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class VDNSelector:
    """
    Client selector based on Value Decomposition Networks (VDN) with Double DQN and PER.

    Each agent estimates Q(o_i, a_i) individually. The joint value function is decomposed
    as Qtot = sum Q_i, enabling cooperative training from a single global reward.
    Top-K selection chooses the K clients with highest advantage Q(o,1) - Q(o,0).

    Args:
        n_agents:          number of clients/agents
        d_in:              local observation dimension
        k_select:          number of clients selected per round (K)
        hidden:            neurons in Q-network hidden layers
        lr:                Adam learning rate
        weight_decay:      L2 regularization
        gamma:             discount factor
        grad_clip:         maximum norm for gradient clipping
        target_sync_every: target network sync frequency (in train calls)
        buf_size:          replay buffer capacity
        batch_size:        default batch size
        train_steps:       optimization steps per train call
        per_alpha:         PER prioritization exponent
        per_beta_start:    initial PER beta
        per_beta_end:      final PER beta
        per_beta_steps:    number of steps for beta annealing
        per_eps:           PER numerical stability constant
        double_dqn:        use Double DQN to reduce overestimation bias
        seed:              random seed for reproducibility
    """
    def __init__(
        self,
        n_agents: int,
        d_in: int,
        k_select: int,
        hidden: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 0.90,
        grad_clip: float = 1.0,
        target_sync_every: int = 20,
        buf_size: int = 20000,
        batch_size: int = 128,
        train_steps: int = 20,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 4000,
        per_eps: float = 1e-3,
        double_dqn: bool = True,
        seed: int = 0,
    ):
        
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.k_select = int(k_select)
        self.gamma = float(gamma)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(target_sync_every)
        self.double_dqn = bool(double_dqn)
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_steps = int(per_beta_steps)

        self.q = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)

        self.buf = PrioritizedReplayJoint(
            capacity=buf_size,
            n_agents=self.n_agents,
            d_in=self.d_in,
            alpha=per_alpha,
            eps=per_eps,
            seed=int(seed) + 12345,
        )

        self._train_calls = 0
        self.total_updates = 0
        self.total_samples_drawn = 0

        self.py_rng = random.Random(int(seed) + 777)
        self.np_rng = np.random.default_rng(int(seed) + 999)

    def _beta(self) -> float:
        """
        Computes the current importance sampling beta via linear annealing.

        Returns:
            Beta value linearly interpolated from per_beta_start to per_beta_end
            over per_beta_steps training calls.
        """
        t = min(self._train_calls, self.per_beta_steps)
        frac = t / max(1, self.per_beta_steps)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    @torch.no_grad()
    def _q_all_agents(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes Q-values for all agents  (used internally for Top-K selection).

        Args:
            obs: joint observations, shape (N, d_in)

        Returns:
            Q-values array, shape (N, 2), dtype float64
        """
        
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        return self.q(x).detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes Q-values for all agents  (used externally for logging and ranking).

        Args:
            obs: joint observations, shape (N, d_in)

        Returns:
            Q-values array, shape (N, 2), dtype float32
        """
        
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        return self.q(x).detach().cpu().numpy().astype(np.float32)

    def select_topk_actions(
        self,
        obs: np.ndarray,
        eps: float = 0.15,
        swap_m: int = 2,
        force_random: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Selects the K best clients with exploration via Top-K.

        Args:
            obs:          joint observations, shape (N, d_in)
            eps:          probability of performing a random swap
            swap_m:       number of clients swapped during exploration
            force_random: forces random selection (used during warm-up)

        Returns:
            Tuple (actions, selected) where actions is a binary vector (N,) and selected is a list of indices
        """

        n = obs.shape[0]
        K = min(self.k_select, n)
        q = self._q_all_agents(obs)

        if force_random:
            sel = self.py_rng.sample(range(n), K)
            a = np.zeros(n, dtype=np.uint8)
            a[sel] = 1
            return a, sel

        adv = q[:, 1] - q[:, 0]
        sel = np.argsort(adv)[::-1][:K].tolist()

        if swap_m > 0 and (self.np_rng.random() < eps):
            swap_m = min(swap_m, K)
            sel_set = set(sel)
            not_sel = [i for i in range(n) if i not in sel_set]
            if len(not_sel) > 0:
                out = self.py_rng.sample(sel, swap_m)
                inn = self.py_rng.sample(not_sel, min(swap_m, len(not_sel)))
                sel2 = sel.copy()
                for o, i_new in zip(out, inn):
                    sel2[sel2.index(o)] = i_new
                sel2 = list(dict.fromkeys(sel2))
                while len(sel2) < K:
                    cand = self.py_rng.randrange(n)
                    if cand not in sel2:
                        sel2.append(cand)
                sel = sel2[:K]

        a = np.zeros(n, dtype=np.uint8)
        a[sel] = 1
        return a, sel

    def add_transition(self, obs, act, r, obs2, done: bool):
        """
        Stores a transition in the replay buffer.

        Args:
            obs:  joint observations for the current round, shape (N, d_in)
            act:  binary actions executed, shape (N,)
            r:    global reward for the round
            obs2: joint observations for the next round, shape (N, d_in)
            done: indicates end of federated training
        """

        self.buf.add(obs=obs, act=act, r=r, obs2=obs2, done=done)

    def train(
        self, batch_size: Optional[int] = None, train_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Runs Q-network optimization steps from the replay buffer.

        Args:
            batch_size:  batch size (uses default if None)
            train_steps: number of steps (uses default if None)

        Returns:
            Mean loss over the executed steps, or None if the buffer is empty
        """

        bs_req = int(batch_size) if batch_size is not None else self.batch_size
        steps = int(train_steps) if train_steps is not None else self.train_steps

        if self.buf.n < max(32, bs_req):
            return None

        beta = self._beta()
        self.q.train()
        losses = []

        for _ in range(steps):
            ob, ac, rw, ob2, dn, idx, w_is = self.buf.sample(batch_size=bs_req, beta=beta)
            B = ob.shape[0]
            self.total_samples_drawn += B

            obs  = torch.tensor(ob,  dtype=torch.float32, device=DEVICE)
            act  = torch.tensor(ac,  dtype=torch.long,    device=DEVICE)
            r    = torch.tensor(rw,  dtype=torch.float32, device=DEVICE)
            obs2 = torch.tensor(ob2, dtype=torch.float32, device=DEVICE)
            done = torch.tensor(dn,  dtype=torch.float32, device=DEVICE)
            w    = torch.tensor(w_is, dtype=torch.float32, device=DEVICE)

            N = self.n_agents
            q_cur = self.q(obs.reshape(B * N, self.d_in)).reshape(B, N, 2)
            q_a = q_cur.gather(2, act.unsqueeze(2)).squeeze(2)
            q_tot = q_a.sum(dim=1)

            with torch.no_grad():
                q2_online = self.q(obs2.reshape(B * N, self.d_in)).reshape(B, N, 2)
                q2_tgt    = self.q_tgt(obs2.reshape(B * N, self.d_in)).reshape(B, N, 2)

                K = min(self.k_select, N)
                adv2 = (
                    (q2_online[:, :, 1] - q2_online[:, :, 0])
                    if self.double_dqn
                    else (q2_tgt[:, :, 1] - q2_tgt[:, :, 0])
                )
                top_idx = adv2.topk(K, dim=1).indices
                a2 = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
                a2.scatter_(1, top_idx, 1)

                q2_a = q2_tgt.gather(2, a2.unsqueeze(2)).squeeze(2)
                q2_tot = q2_a.sum(dim=1)
                y = r + (1.0 - done) * self.gamma * q2_tot

            td_abs = (q_tot - y).detach().abs().cpu().numpy().astype(np.float32)
            loss = (w * F.smooth_l1_loss(q_tot, y, reduction="none")).mean()

            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

            self.buf.update_priorities(idx, td_abs)
            losses.append(float(loss.item()))
            self.total_updates += 1

        self._train_calls += 1
        if self._train_calls % self.target_sync_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(np.mean(losses)) if losses else None


def build_context_matrix_vdn(
    projection_mom: np.ndarray,
    probe_now: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
) -> np.ndarray:
    """
    Builds the local observation matrix for the VDN agent.

    Args:
        projection_mom: projection of client weight delta onto the server gradient EMA (proj), shape (N,)
        probe_now:      generalization loss of the global model on local client data (gener), shape (N,)
        staleness:      rounds since each client was last selected, shape (N,)
        streak:         consecutive selection count for each client, shape (N,)

    Returns:
        Observation matrix (N, 5) with columns [bias, proj, gener, estag*, serie*]
    """

    proj  = projection_mom.astype(np.float32)
    probe = probe_now.astype(np.float32)

    s = staleness.astype(np.float32)
    sn = (s / (s.max() + 1e-6)).astype(np.float32)

    t = streak.astype(np.float32)
    tn = np.clip(t / 5.0, 0.0, 1.0).astype(np.float32)

    bias = np.ones(proj.shape[0], dtype=np.float32)
    return np.stack([bias, proj, probe, sn, tn], axis=1).astype(np.float32)

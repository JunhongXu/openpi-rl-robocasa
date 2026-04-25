"""Unified replay buffer for both on-policy (PPO) and off-policy (SAC) usage.

Storage layout: every leaf has shape [capacity, num_envs, ...]. Per-step `add`
writes at `self.write_ptr`; the pointer wraps for SAC-style circular use, but
PPO clients typically `reset()` between rollouts so the buffer stays contiguous
(which is required for `compute_gae`).

The buffer is schema-agnostic — it allocates storage from a sample transition
pytree and stores whatever leaves the user provides. PPO-specific helpers
(`compute_gae`, `iterate_minibatches`) look for the conventional keys
{`reward`, `value`, `done`, `truncated`} and write back {`advantage`, `returns`}.
"""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        sample_transition: dict[str, Any],
    ):
        """Allocate storage mirroring `sample_transition`'s pytree structure.

        Args:
            capacity: max number of timesteps stored per env.
            num_envs: parallel-env axis (use 1 for single-env rollouts).
            sample_transition: one example transition with leading shape [num_envs, ...]
                per leaf. Used purely as a spec — values are ignored.
        """
        self.capacity = capacity
        self.num_envs = num_envs

        def _alloc(leaf):
            arr = np.asarray(leaf)
            assert arr.shape[0] == num_envs, (
                f"leaf leading axis must equal num_envs={num_envs}, got {arr.shape}"
            )
            return np.zeros((capacity, *arr.shape), dtype=arr.dtype)

        self.data: dict[str, Any] = jax.tree.map(_alloc, sample_transition)
        self.write_ptr = 0
        self.size = 0
        self._wrapped = False

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size == self.capacity

    @property
    def is_contiguous(self) -> bool:
        """True iff data is stored in temporal order [0, size) — required for GAE."""
        return not self._wrapped

    def reset(self) -> None:
        self.write_ptr = 0
        self.size = 0
        self._wrapped = False

    def add(self, transition: dict[str, Any]) -> None:
        """Write one timestep. Each leaf must have leading shape [num_envs, ...]."""
        idx = self.write_ptr

        def _write(buf, leaf):
            arr = np.asarray(leaf)
            assert arr.shape[0] == self.num_envs, (
                f"add() leaf leading axis must equal num_envs={self.num_envs}, got {arr.shape}"
            )
            buf[idx] = arr

        jax.tree.map(_write, self.data, transition)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self._wrapped = True

    # ---- SAC-style: random sampling ----------------------------------------

    def sample_random(self, rng: np.random.Generator, batch_size: int) -> dict[str, Any]:
        """Uniformly sample `batch_size` transitions, flattened across (T, N)."""
        assert self.size > 0, "buffer is empty"
        t_idx = rng.integers(0, self.size, size=batch_size)
        n_idx = rng.integers(0, self.num_envs, size=batch_size)
        batch = jax.tree.map(lambda buf: jnp.asarray(buf[t_idx, n_idx]), self.data)
        return batch

    # ---- PPO-style: GAE + sequential minibatches ---------------------------

    def compute_gae(
        self,
        last_value: np.ndarray,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns; write back into `self.data`.

        Requires the buffer to be contiguous (i.e. not yet wrapped — typical PPO use:
        fill, GAE, iterate, reset). Reads {reward, value, done, truncated} of shape
        [size, N] and writes {advantage, returns} of the same shape.

        Args:
            last_value: V(s_{T}) bootstrap, shape [num_envs]. Used for the slot AFTER
                the final stored step.

        Termination semantics:
            - `done=True`: real terminal — bootstrap value is masked to 0.
            - `truncated=True`: time-limit truncation — bootstrap value is kept (so the
              TD target uses last_value), but the GAE chain is cut so credit doesn't
              flow across the artificial boundary.
        """
        assert self.is_contiguous, "compute_gae requires contiguous buffer (call reset())"
        assert {"reward", "value", "done", "truncated"} <= self.data.keys(), (
            "compute_gae needs reward/value/done/truncated in the transition schema"
        )
        T, N = self.size, self.num_envs
        assert last_value.shape == (N,), f"last_value must be [N], got {last_value.shape}"

        rewards = self.data["reward"][:T].astype(np.float32)
        values = self.data["value"][:T].astype(np.float32)
        dones = self.data["done"][:T].astype(np.float32)
        truncs = self.data["truncated"][:T].astype(np.float32)

        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros((N,), dtype=np.float32)
        next_value = np.asarray(last_value, dtype=np.float32)
        # Bootstrap slot has no `done`; treat it as non-terminal.
        next_nonterminal = np.ones((N,), dtype=np.float32)

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            # Cut the GAE chain on either real terminal or time-limit truncation.
            chain_mask = next_nonterminal * (1.0 - truncs[t])
            last_gae = delta + gamma * gae_lambda * chain_mask * last_gae
            advantages[t] = last_gae
            next_value = values[t]
            next_nonterminal = 1.0 - dones[t]

        returns = advantages + values

        # Allocate advantage/returns buffers lazily so the schema doesn't have to
        # include them up front.
        if "advantage" not in self.data:
            self.data["advantage"] = np.zeros((self.capacity, N), dtype=np.float32)
            self.data["returns"] = np.zeros((self.capacity, N), dtype=np.float32)
        self.data["advantage"][:T] = advantages
        self.data["returns"][:T] = returns

    def iterate_minibatches(
        self,
        rng: np.random.Generator,
        batch_size: int,
        num_epochs: int = 1,
    ) -> Iterator[dict[str, Any]]:
        """Flatten valid data to [size*N, ...], shuffle, yield minibatches.

        For PPO: call after `compute_gae`. Iterates over the on-policy data
        `num_epochs` times with a fresh shuffle each epoch.
        """
        assert self.is_contiguous, "iterate_minibatches expects contiguous buffer"
        T, N = self.size, self.num_envs
        total = T * N
        assert batch_size <= total, f"batch_size {batch_size} > {total} samples available"

        # Pre-flatten once per epoch (avoids re-flattening every minibatch).
        for _ in range(num_epochs):
            perm = rng.permutation(total)
            for start in range(0, total - batch_size + 1, batch_size):
                idx = perm[start : start + batch_size]
                t_idx, n_idx = np.divmod(idx, N)
                batch = jax.tree.map(
                    lambda buf, ti=t_idx, ni=n_idx: jnp.asarray(buf[ti, ni]),
                    self.data,
                )
                yield batch

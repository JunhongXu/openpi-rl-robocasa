import jax
import jax.numpy as jnp


def ppo_loss(
    new_logprob: jnp.ndarray,
    old_logprob: jnp.ndarray,
    advantage: jnp.ndarray,
    entropy: jnp.ndarray | None = None,
    *,
    clip_ratio: float = 0.2,
    log_ratio_clip: float = 20.0,
    entropy_coef: float = 0.0,
    normalize_advantage: bool = True,
) -> tuple[jnp.ndarray, dict]:
    """PPO clipped surrogate loss for the pi0 flow-matching policy.

    Shape contract (B = env-timestep batch, K = flow steps per action):
        new_logprob : [B, K]   summed over (ah, ad) by compute_action_logprob
        old_logprob : [B, K]   stored from the rollout
        advantage   : [B]      one per env timestep (shared across K flow sub-steps)
        entropy     : [B, K]   optional, summed over (ah, ad)

    Each env timestep produces ONE observation but K flow sub-steps. We treat each
    flow sub-step as a separate decision sharing the same advantage — the standard
    DPPO / Diffusion-PPO formulation. The surrogate is averaged over both axes so
    the loss magnitude is independent of K.
    """
    assert new_logprob.shape == old_logprob.shape, (
        f"logprob shape mismatch: {new_logprob.shape} vs {old_logprob.shape}"
    )
    assert advantage.shape == new_logprob.shape[:1], (
        f"advantage must be [B], got {advantage.shape} for logprob {new_logprob.shape}"
    )

    if normalize_advantage:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # Broadcast per-env-step advantage to per-flow-step.
    advantage = advantage[:, None]  # [B, 1] -> broadcasts to [B, K]
    advantage = jax.lax.stop_gradient(advantage)

    # Clip the log-ratio first — with high-dim actions, exp() over un-clipped
    # log-ratios can overflow/underflow even when per-dim drift is small.
    log_ratio = new_logprob - old_logprob
    log_ratio = jnp.clip(log_ratio, -log_ratio_clip, log_ratio_clip)
    ratio = jnp.exp(log_ratio)

    surr1 = ratio * advantage
    surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    if entropy is not None and entropy_coef != 0.0:
        entropy_bonus = entropy_coef * jnp.mean(entropy)
        loss = policy_loss - entropy_bonus
    else:
        entropy_bonus = jnp.zeros(())
        loss = policy_loss

    # Schulman k3 KL estimator: unbiased, always >= 0. http://joschu.net/blog/kl-approx.html
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_ratio).astype(jnp.float32))

    info = {
        "policy_loss": policy_loss,
        "entropy_bonus": entropy_bonus,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "mean_ratio": jnp.mean(ratio),
        "mean_advantage": jnp.mean(advantage),
    }
    return loss, info

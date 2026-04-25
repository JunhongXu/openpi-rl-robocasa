import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rl.ppo_loss import ppo_loss


def _make_inputs(B=8, K=10, seed=0, advantage=None, log_ratio=None, entropy=None):
    rng = np.random.default_rng(seed)
    old = rng.standard_normal((B, K)).astype(np.float32) * 100.0  # large magnitude on purpose
    if log_ratio is None:
        log_ratio = rng.standard_normal((B, K)).astype(np.float32) * 0.05
    new = old + log_ratio
    if advantage is None:
        advantage = rng.standard_normal((B,)).astype(np.float32)
    return jnp.asarray(new), jnp.asarray(old), jnp.asarray(advantage), entropy


def test_zero_log_ratio_gives_neg_mean_advantage():
    # When new == old, ratio == 1, both surrogates equal advantage.
    # With normalize_advantage=False, loss = -mean(advantage_broadcast) = -mean(advantage).
    new, old, adv, _ = _make_inputs(log_ratio=jnp.zeros((8, 10), dtype=jnp.float32))
    loss, info = ppo_loss(new, old, adv, normalize_advantage=False)
    np.testing.assert_allclose(loss, -jnp.mean(adv), atol=1e-6)
    np.testing.assert_allclose(info["mean_ratio"], 1.0, atol=1e-6)
    np.testing.assert_allclose(info["approx_kl"], 0.0, atol=1e-6)
    np.testing.assert_allclose(info["clip_frac"], 0.0)


def test_normalized_advantage_makes_loss_zero_at_ratio_one():
    # With ratio==1 and advantage normalized to mean 0, loss should be ~0.
    new, old, adv, _ = _make_inputs(log_ratio=jnp.zeros((8, 10), dtype=jnp.float32))
    loss, _ = ppo_loss(new, old, adv, normalize_advantage=True)
    np.testing.assert_allclose(loss, 0.0, atol=1e-5)


def test_high_dim_logprobs_dont_overflow():
    # Simulate a 1600-dim sum-logprob with magnitudes ~1e4 — this is the regime
    # where naive exp(new - old) overflows. log_ratio_clip should keep ratio finite.
    B, K = 4, 10
    rng = np.random.default_rng(0)
    old = jnp.asarray(rng.standard_normal((B, K)).astype(np.float32) * 1e4)
    # Force a huge logratio to trip the clip.
    new = old + 1e3
    adv = jnp.asarray(rng.standard_normal((B,)).astype(np.float32))
    loss, info = ppo_loss(new, old, adv, log_ratio_clip=20.0, normalize_advantage=False)
    assert jnp.isfinite(loss).item()
    assert jnp.all(jnp.isfinite(info["mean_ratio"])).item()
    # ratio should saturate near exp(20)
    np.testing.assert_allclose(info["mean_ratio"], np.exp(20.0), rtol=1e-5)


def test_clipping_engages_when_ratio_far_from_one():
    # Construct a case where ratio is well outside [1-eps, 1+eps] for every element.
    B, K = 4, 5
    new = jnp.zeros((B, K))
    old = jnp.full((B, K), -1.0)  # log_ratio = +1 → ratio = e ≈ 2.718
    adv = jnp.ones((B,))  # positive advantage everywhere
    _, info = ppo_loss(new, old, adv, clip_ratio=0.2, normalize_advantage=False)
    np.testing.assert_allclose(info["clip_frac"], 1.0)


def test_advantage_is_broadcast_across_K():
    # Advantage [B] is broadcast to [B, K]. Verify by computing the loss two ways.
    B, K = 6, 4
    rng = np.random.default_rng(1)
    new = jnp.asarray(rng.standard_normal((B, K)).astype(np.float32) * 0.01)
    old = jnp.zeros((B, K))
    adv = jnp.asarray(rng.standard_normal((B,)).astype(np.float32))

    loss_actual, _ = ppo_loss(new, old, adv, normalize_advantage=False)

    # Reference: do the broadcast explicitly.
    log_ratio = jnp.clip(new - old, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    adv_broadcast = jnp.broadcast_to(adv[:, None], (B, K))
    surr1 = ratio * adv_broadcast
    surr2 = jnp.clip(ratio, 0.8, 1.2) * adv_broadcast
    loss_ref = -jnp.mean(jnp.minimum(surr1, surr2))

    np.testing.assert_allclose(loss_actual, loss_ref, atol=1e-6)


def test_entropy_bonus_lowers_loss_for_positive_entropy():
    new, old, adv, _ = _make_inputs(log_ratio=jnp.zeros((8, 10), dtype=jnp.float32))
    entropy = jnp.ones((8, 10))
    loss_no_bonus, _ = ppo_loss(new, old, adv, entropy=entropy, entropy_coef=0.0,
                                normalize_advantage=False)
    loss_with_bonus, info = ppo_loss(new, old, adv, entropy=entropy, entropy_coef=0.1,
                                     normalize_advantage=False)
    # entropy_bonus = 0.1 * 1.0 = 0.1; loss = policy_loss - entropy_bonus
    np.testing.assert_allclose(loss_with_bonus, loss_no_bonus - 0.1, atol=1e-6)
    np.testing.assert_allclose(info["entropy_bonus"], 0.1, atol=1e-6)


def test_shape_mismatch_raises():
    new = jnp.zeros((4, 10))
    old = jnp.zeros((4, 5))  # K mismatch
    adv = jnp.zeros((4,))
    with pytest.raises(AssertionError, match="logprob shape mismatch"):
        ppo_loss(new, old, adv)


def test_advantage_shape_mismatch_raises():
    new = jnp.zeros((4, 10))
    old = jnp.zeros((4, 10))
    adv = jnp.zeros((4, 10))  # should be [B], not [B, K]
    with pytest.raises(AssertionError, match="advantage must be"):
        ppo_loss(new, old, adv)


def test_advantage_has_no_gradient():
    # The PPO loss should backprop into new_logprob but NOT into advantage —
    # we use stop_gradient on advantage so a value-head gradient can't leak in.
    B, K = 4, 6
    new = jnp.ones((B, K)) * 0.01
    old = jnp.zeros((B, K))
    adv = jnp.ones((B,))

    grad_new = jax.grad(lambda n: ppo_loss(n, old, adv, normalize_advantage=False)[0])(new)
    grad_adv = jax.grad(lambda a: ppo_loss(new, old, a, normalize_advantage=False)[0])(adv)

    assert jnp.any(grad_new != 0).item(), "loss should have nonzero gradient w.r.t. new_logprob"
    np.testing.assert_array_equal(np.asarray(grad_adv), np.zeros_like(adv))


def test_loss_decreases_when_step_in_gradient_direction():
    # Sanity check: a single SGD step on new_logprob should reduce the loss.
    B, K = 4, 6
    rng = np.random.default_rng(42)
    new = jnp.asarray(rng.standard_normal((B, K)).astype(np.float32) * 0.01)
    old = jnp.zeros((B, K))
    adv = jnp.asarray(rng.standard_normal((B,)).astype(np.float32))

    loss_fn = lambda n: ppo_loss(n, old, adv, normalize_advantage=False)[0]
    loss0 = loss_fn(new)
    grad = jax.grad(loss_fn)(new)
    new_updated = new - 1e-3 * grad
    loss1 = loss_fn(new_updated)
    assert loss1 < loss0


def test_jit_compatible():
    # The loss must be jit-compilable since it'll live inside the PPO update step.
    new, old, adv, _ = _make_inputs()
    loss_jit = jax.jit(lambda n, o, a: ppo_loss(n, o, a)[0])(new, old, adv)
    loss_eager, _ = ppo_loss(new, old, adv)
    np.testing.assert_allclose(loss_jit, loss_eager, atol=1e-6)

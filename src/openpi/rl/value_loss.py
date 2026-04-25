import jax.numpy as jnp
import jax


def compute_gae(reward, value, discount_factor=0.99, gae_lambda=0.95):
    """
    TODO(junhong)
    """

def value_loss(value, target, clip_ratio=0.2):
    """
    TODO(junhong)
    """
    return jnp.mean(jnp.square(value - target))
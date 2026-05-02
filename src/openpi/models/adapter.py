"""
A more efficient PEFT method than LoRA. Based on:
Weight Updates as Activation Shifts A Principled Framework for Steering
"""
import jax.numpy as jnp
import flax.nnx as nnx

class Adapter(nnx.Module):
    def __init__(self, input_dim: int, down_dim: int, up_dim: int, output_dim: int, rngs: nnx.Rngs):
        """
        An activation adapter is a 2-layer MLP that computes the residual activation to shift
        the activation of the original module. This is different from LoRA in that it does not
        modify the weights of the original model. 
        """
        self.input_dim = input_dim
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.output_dim = output_dim

        self.down_proj = nnx.Linear(input_dim, down_dim, rngs=rngs)
        self.up_proj = nnx.Linear(down_dim, up_dim, rngs=rngs)
        self.out_proj = nnx.Linear(up_dim, output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.up_proj(self.down_proj(x))
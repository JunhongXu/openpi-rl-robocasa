import jax
import flax.nnx as nnx
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import Pi0Config 

config = Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
rngs = nnx.Rngs(0)
model = Pi0(config, rngs=rngs)

obs = config.fake_obs(batch_size=2)
act = config.fake_act(batch_size=2)

# training forward
loss = model.compute_loss(jax.random.key(0), obs, act, train=True)
print("loss:", loss.shape, loss.mean())

# inference
actions, trajectory = model.sample_actions(jax.random.key(1), obs, num_steps=4, stochastic=True)
print("actions:", actions.shape)
print("trajectory:", trajectory.keys())

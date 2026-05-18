from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


def build_context_features(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
) -> jnp.ndarray:
    rewards = rewards[..., None]
    dones = dones[..., None]
    return jnp.concatenate(
        [observations, actions, rewards, next_observations, dones],
        axis=-1,
    )


class ContextEncoder(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, context_features: jnp.ndarray):
        x = context_features
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.gelu(x)

        x = jnp.mean(x, axis=1)
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        logvar = jnp.clip(logvar, -10.0, 2.0)
        return mu, logvar


class TaskDecoder(nn.Module):
    hidden_dims: Sequence[int]
    num_tasks: int

    @nn.compact
    def __call__(self, latent: jnp.ndarray):
        x = latent
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.gelu(x)
        return nn.Dense(self.num_tasks)(x)

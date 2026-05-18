import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from functools import partial
from gymnasium import spaces

from common.type_aliases import ReplayBufferSamplesNp
from diffusion.diffusion_policy import DiffPol
from diffusion.dime import DIME, save_model_state
from meta.context_encoder import ContextEncoder, TaskDecoder, build_context_features


class MetaDIME(DIME):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.meta_cfg = getattr(cfg, "meta", None) if cfg is not None else None
        if self.meta_cfg is None:
            raise ValueError("MetaDIME requires cfg.meta to be configured.")
        self.latent_dim = int(getattr(self.meta_cfg, "latent_dim", 0) or 0)
        if self.latent_dim <= 0:
            raise ValueError("cfg.meta.latent_dim must be > 0 for MetaDIME.")

        self.context_encoder = None
        self.context_encoder_state = None
        self.task_decoder = None
        self.task_decoder_state = None
        self.context_feature_dim = None
        self.num_tasks = self._infer_num_tasks(cfg)
        super().__init__(*args, **kwargs)

    def _infer_num_tasks(self, cfg) -> int:
        taskset = getattr(cfg, "taskset", None) if cfg is not None else None
        if taskset is None:
            return int(getattr(self.meta_cfg, "num_tasks", 1) or 1)
        tasks = getattr(taskset, "tasks", None)
        if tasks is None:
            return int(getattr(self.meta_cfg, "num_tasks", 1) or 1)
        return max(int(len(tasks)), 1)

    def _raw_obs_dim(self) -> int:
        if isinstance(self.observation_space, spaces.Dict):
            sample = spaces.flatten(self.observation_space, self.observation_space.sample())
            return int(sample.shape[0])
        return int(np.prod(self.observation_space.shape))

    def _setup_model(self, reset=False) -> None:
        super()._setup_model(reset=reset)

        if (
            self.context_encoder is not None
            and self.context_encoder_state is not None
            and self.task_decoder is not None
            and self.task_decoder_state is not None
            and not reset
        ):
            return

        obs_dim = self._raw_obs_dim()
        action_dim = int(self.action_space.shape[0])
        self.context_feature_dim = obs_dim * 2 + action_dim + 2
        hidden_dims = tuple(int(v) for v in getattr(self.meta_cfg, "encoder_hidden_dims", [256, 256]))
        decoder_hidden_dims = tuple(
            int(v) for v in getattr(self.meta_cfg, "task_decoder_hidden_dims", [128, 128])
        )
        encoder_lr = float(getattr(self.meta_cfg, "encoder_lr", self.cfg.alg.optimizer.lr_actor))

        self.context_encoder = ContextEncoder(
            hidden_dims=hidden_dims,
            latent_dim=self.latent_dim,
        )
        self.task_decoder = TaskDecoder(
            hidden_dims=decoder_hidden_dims,
            num_tasks=self.num_tasks,
        )

        self.key, encoder_key, decoder_key = jax.random.split(self.key, 3)
        dummy_context = jnp.ones(
            (
                1,
                int(getattr(self.meta_cfg, "context_batch_size", 64)),
                self.context_feature_dim,
            ),
            dtype=jnp.float32,
        )
        encoder_params = self.context_encoder.init({"params": encoder_key}, dummy_context)["params"]
        self.context_encoder_state = TrainState.create(
            apply_fn=self.context_encoder.apply,
            params=encoder_params,
            tx=optax.adam(learning_rate=encoder_lr),
        )
        dummy_latent = jnp.ones((1, self.latent_dim), dtype=jnp.float32)
        decoder_params = self.task_decoder.init({"params": decoder_key}, dummy_latent)["params"]
        self.task_decoder_state = TrainState.create(
            apply_fn=self.task_decoder.apply,
            params=decoder_params,
            tx=optax.adam(learning_rate=encoder_lr),
        )

    def prior_latent(self, batch_size: int = 1) -> np.ndarray:
        return np.zeros((int(batch_size), self.latent_dim), dtype=np.float32)

    @staticmethod
    def _ensure_task_batch(context_batch):
        observations = np.asarray(context_batch["observations"], dtype=np.float32)
        actions = np.asarray(context_batch["actions"], dtype=np.float32)
        next_observations = np.asarray(context_batch["next_observations"], dtype=np.float32)
        rewards = np.asarray(context_batch["rewards"], dtype=np.float32)
        dones = np.asarray(context_batch["dones"], dtype=np.float32)

        if observations.ndim == 2:
            observations = observations[None, ...]
            actions = actions[None, ...]
            next_observations = next_observations[None, ...]
            rewards = rewards[None, ...]
            dones = dones[None, ...]

        return {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
            "dones": dones,
        }

    def infer_posterior(self, context_batch, sample: bool = False):
        context_batch = self._ensure_task_batch(context_batch)
        context_features = build_context_features(
            jnp.asarray(context_batch["observations"], dtype=jnp.float32),
            jnp.asarray(context_batch["actions"], dtype=jnp.float32),
            jnp.asarray(context_batch["rewards"], dtype=jnp.float32),
            jnp.asarray(context_batch["next_observations"], dtype=jnp.float32),
            jnp.asarray(context_batch["dones"], dtype=jnp.float32),
        )

        mu, logvar = self.context_encoder_state.apply_fn(
            {"params": self.context_encoder_state.params},
            context_features,
        )

        if sample:
            self.key, latent_key = jax.random.split(self.key)
            eps = jax.random.normal(latent_key, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps
        else:
            z = mu

        return {
            "z": np.asarray(z, dtype=np.float32),
            "mu": np.asarray(mu, dtype=np.float32),
            "logvar": np.asarray(logvar, dtype=np.float32),
        }

    @staticmethod
    def _augment_obs_with_latent(observations: np.ndarray, latents: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        latents = np.asarray(latents, dtype=np.float32)
        if observations.ndim == 3:
            latents = np.repeat(latents[:, None, :], observations.shape[1], axis=1)
        elif observations.ndim == 2 and latents.ndim == 2 and latents.shape[0] == 1 and observations.shape[0] != 1:
            latents = np.repeat(latents, observations.shape[0], axis=0)
        return np.concatenate([observations, latents], axis=-1).astype(np.float32)

    def predict_with_latent(self, observation, latent, deterministic: bool = False):
        return self.policy.predict_with_latent(observation, latent, deterministic=deterministic)

    def infer_latent_from_buffer(
        self,
        task_buffer,
        task_id: str,
        context_batch_size: int,
        sample: bool = False,
        recent: bool = True,
    ) -> np.ndarray:
        context_batch = task_buffer.sample_context(
            task_id,
            int(context_batch_size),
            recent=recent,
        )
        if context_batch is None:
            return self.prior_latent(batch_size=1)[0]
        posterior = self.infer_posterior(context_batch, sample=sample)
        return posterior["z"][0]

    def train_meta(self, task_buffer):
        meta_batch = task_buffer.sample_meta_batch(
            meta_batch_tasks=int(getattr(self.meta_cfg, "meta_batch_tasks", 5)),
            context_batch_size=int(getattr(self.meta_cfg, "context_batch_size", 64)),
            query_batch_size=int(getattr(self.meta_cfg, "query_batch_size", 64)),
            min_task_transitions=int(getattr(self.meta_cfg, "min_task_buffer_size", 128)),
            disjoint_context_query=bool(getattr(self.meta_cfg, "disjoint_context_query", True)),
        )
        if meta_batch is None:
            return None

        posterior = self.infer_posterior(
            {
                "observations": meta_batch["context_observations"],
                "actions": meta_batch["context_actions"],
                "next_observations": meta_batch["context_next_observations"],
                "rewards": meta_batch["context_rewards"],
                "dones": meta_batch["context_dones"],
            },
            sample=bool(getattr(self.meta_cfg, "sample_posterior_during_training", True)),
        )
        latents = posterior["z"]

        query_obs = self._augment_obs_with_latent(meta_batch["query_observations"], latents)
        query_next_obs = self._augment_obs_with_latent(meta_batch["query_next_observations"], latents)
        flat_data = ReplayBufferSamplesNp(
            query_obs.reshape(-1, query_obs.shape[-1]),
            meta_batch["query_actions"].reshape(-1, meta_batch["query_actions"].shape[-1]),
            query_next_obs.reshape(-1, query_next_obs.shape[-1]),
            meta_batch["query_dones"].reshape(-1),
            meta_batch["query_rewards"].reshape(-1),
        )

        rl_metrics = None
        updates_per_call = max(int(self.cfg.alg.utd), 1)
        for _ in range(updates_per_call):
            policy_delay_indices = flax.core.FrozenDict(
                {0: True} if ((self._n_updates + 1) % self.policy_delay) == 0 else {}
            )
            (
                self.policy.qf_state,
                self.policy.actor_state,
                self.policy.target_actor_state,
                self.ent_coef_state,
                self.key,
                rl_metrics,
            ) = self._train(
                self.crossq_style,
                self.use_bnstats_from_live_net,
                self.gamma,
                self.tau,
                self.policy_tau,
                self.target_entropy,
                1,
                flat_data,
                policy_delay_indices,
                self.policy.qf_state,
                self.policy.actor_state,
                self.policy.target_actor_state,
                self.ent_coef_state,
                self.key,
                self.num_timesteps,
                self.policy_q_reduce_fn,
                self.policy.sampler,
                self.policy.target_sampler,
                self.cfg.alg.critic.v_min,
                self.cfg.alg.critic.v_max,
                self.cfg.alg.critic.entr_coeff,
                self.cfg.alg.critic.n_atoms,
            )
            self._n_updates += 1

        context_features = build_context_features(
            jnp.asarray(meta_batch["context_observations"], dtype=jnp.float32),
            jnp.asarray(meta_batch["context_actions"], dtype=jnp.float32),
            jnp.asarray(meta_batch["context_rewards"], dtype=jnp.float32),
            jnp.asarray(meta_batch["context_next_observations"], dtype=jnp.float32),
            jnp.asarray(meta_batch["context_dones"], dtype=jnp.float32),
        )
        (
            self.context_encoder_state,
            self.task_decoder_state,
            encoder_metrics,
            self.key,
        ) = self.update_context_encoder(
            self.context_encoder_state,
            self.task_decoder_state,
            self.policy.target_actor_state,
            self.policy.qf_state,
            self.ent_coef_state,
            context_features,
            jnp.asarray(meta_batch["task_indices"], dtype=jnp.int32),
            jnp.asarray(meta_batch["query_observations"], dtype=jnp.float32),
            jnp.asarray(meta_batch["query_actions"], dtype=jnp.float32),
            jnp.asarray(meta_batch["query_next_observations"], dtype=jnp.float32),
            jnp.asarray(meta_batch["query_rewards"], dtype=jnp.float32),
            jnp.asarray(meta_batch["query_dones"], dtype=jnp.float32),
            int(self.num_timesteps),
            float(self.gamma),
            float(self.cfg.alg.critic.v_min),
            float(self.cfg.alg.critic.v_max),
            int(self.cfg.alg.critic.n_atoms),
            float(self.cfg.alg.critic.entr_coeff),
            float(getattr(self.meta_cfg, "kl_coef", 1e-3)),
            float(getattr(self.meta_cfg, "task_cls_coef", 0.0)),
            self.key,
            self.policy.target_sampler,
        )

        posterior_std = np.exp(0.5 * posterior["logvar"])
        metrics = dict(rl_metrics or {})
        metrics.update({k: float(v) for k, v in encoder_metrics.items()})
        metrics["latent_mean_norm"] = float(np.linalg.norm(posterior["mu"], axis=-1).mean())
        metrics["latent_std_mean"] = float(posterior_std.mean())
        metrics["meta_task_count"] = float(len(meta_batch["task_ids"]))
        return metrics

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "num_atoms"])
    def update_context_encoder(
        context_encoder_state: TrainState,
        task_decoder_state: TrainState,
        actor_state: TrainState,
        qf_state,
        ent_coef_state: TrainState,
        context_features: jnp.ndarray,
        task_indices: jnp.ndarray,
        query_observations: jnp.ndarray,
        query_actions: jnp.ndarray,
        query_next_observations: jnp.ndarray,
        query_rewards: jnp.ndarray,
        query_dones: jnp.ndarray,
        n_env_interacts: int,
        gamma: float,
        v_min: float,
        v_max: float,
        num_atoms: int,
        entr_coeff: float,
        kl_coef: float,
        task_cls_coef: float,
        key,
        sampler,
    ):
        key, latent_key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 5)
        z_atoms = jnp.linspace(v_min, v_max, num_atoms)

        def projection(next_dist, rewards, dones, discount, support):
            delta_z = (v_max - v_min) / (num_atoms - 1)
            batch_size = rewards.shape[0]

            target_z = jnp.clip(
                rewards[:, None] + (1 - dones[:, None]) * discount * support,
                a_min=v_min,
                a_max=v_max,
            )
            b = (target_z - v_min) / delta_z
            l = jnp.floor(b).astype(jnp.int32)
            u = jnp.ceil(b).astype(jnp.int32)
            l = jnp.where((u > 0) & (l == u), l - 1, l)
            u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)

            proj_dist = jnp.zeros_like(next_dist)
            offset = jnp.arange(batch_size)[:, None] * num_atoms
            l_idx = (l + offset).ravel()
            u_idx = (u + offset).ravel()
            l_update = (next_dist * (u.astype(jnp.float32) - b)).ravel()
            u_update = (next_dist * (b - l.astype(jnp.float32))).ravel()
            proj_dist_flat = proj_dist.ravel()
            proj_dist_flat = proj_dist_flat.at[l_idx].add(l_update)
            proj_dist_flat = proj_dist_flat.at[u_idx].add(u_update)
            return proj_dist_flat.reshape(batch_size, num_atoms)

        def binary_cross_entropy(pred, target):
            return -jnp.mean(jnp.sum(target * jnp.log(pred + 1e-15), axis=-1)) + (
                entr_coeff * jnp.mean(jnp.sum(pred * jnp.log(pred + 1e-15), axis=-1))
            )

        def task_cross_entropy(logits, labels):
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))

        def loss_fn(encoder_params, decoder_params):
            mu, logvar = context_encoder_state.apply_fn({"params": encoder_params}, context_features)
            eps = jax.random.normal(latent_key, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps

            z_rep = jnp.repeat(z[:, None, :], query_observations.shape[1], axis=1)
            observations = jnp.concatenate([query_observations, z_rep], axis=-1).reshape(
                -1, query_observations.shape[-1] + z.shape[-1]
            )
            next_observations = jnp.concatenate([query_next_observations, z_rep], axis=-1).reshape(
                -1, query_next_observations.shape[-1] + z.shape[-1]
            )
            actions = query_actions.reshape(-1, query_actions.shape[-1])
            rewards = query_rewards.reshape(-1)
            dones = query_dones.reshape(-1)

            out = DiffPol.sample_action(actor_state, actor_state.params, next_observations, noise_key, sampler)
            next_actions, next_run_costs, next_sto_costs, next_terminal_costs, _, _ = out
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)

            next_q_values = qf_state.apply_fn(
                {
                    "params": qf_state.target_params,
                    "batch_stats": qf_state.target_batch_stats,
                },
                next_observations,
                next_actions,
                rngs={"dropout": dropout_key_target},
                train=False,
            )
            current_q_values = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats,
                },
                observations,
                actions,
                rngs={"dropout": dropout_key_current},
                train=False,
            )

            next_q_values_q1 = next_q_values[0]
            next_q_values_q2 = next_q_values[1]
            current_q1 = current_q_values[0]
            current_q2 = current_q_values[1]

            entr_bon = -(
                (1 - dones[:, None])
                * gamma
                * ent_coef_value
                * (next_run_costs + next_sto_costs + next_terminal_costs)
            )
            target_q1 = projection(next_q_values_q1, rewards + entr_bon.squeeze(), dones, gamma, z_atoms)
            target_q2 = projection(next_q_values_q2, rewards + entr_bon.squeeze(), dones, gamma, z_atoms)
            target_q = jax.lax.stop_gradient(jnp.mean(jnp.stack([target_q1, target_q2], axis=0), axis=0))

            critic_loss = binary_cross_entropy(current_q1, target_q) + binary_cross_entropy(current_q2, target_q)
            kl = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + mu ** 2 - 1.0 - logvar, axis=-1))
            task_logits = task_decoder_state.apply_fn({"params": decoder_params}, mu)
            labels = task_indices.astype(jnp.int32)
            task_cls_loss = task_cross_entropy(task_logits, labels)
            task_acc = jnp.mean((jnp.argmax(task_logits, axis=-1) == labels).astype(jnp.float32))
            total_loss = critic_loss + kl_coef * kl + task_cls_coef * task_cls_loss
            metrics = {
                "encoder_loss": total_loss,
                "encoder_critic_loss": critic_loss,
                "encoder_kl": kl,
                "encoder_task_cls_loss": task_cls_loss,
                "encoder_task_acc": task_acc,
                "encoder_latent_std": jnp.mean(jnp.exp(0.5 * logvar)),
                "encoder_latent_norm": jnp.mean(jnp.linalg.norm(mu, axis=-1)),
            }
            return total_loss, metrics

        (loss_value, metrics), (encoder_grads, decoder_grads) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(context_encoder_state.params, task_decoder_state.params)
        context_encoder_state = context_encoder_state.apply_gradients(grads=encoder_grads)
        task_decoder_state = task_decoder_state.apply_gradients(grads=decoder_grads)
        metrics["encoder_loss"] = loss_value
        return context_encoder_state, task_decoder_state, metrics, key

    def _save_model(self):
        super()._save_model()
        save_model_state(self.context_encoder_state, self.model_save_path, "encoder_state", self.num_timesteps)
        save_model_state(self.task_decoder_state, self.model_save_path, "task_decoder_state", self.num_timesteps)

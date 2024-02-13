# Functions for training

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax

# from purejaxrl.wrappers import LogWrapper, FlattenObservationWrapper
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import wandb


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    actor_mean: jnp.ndarray
    nu1: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return actor_mean, jnp.squeeze(critic, axis=-1)



def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def find_inv(x, fun, fun_inv=None):
        if fun_inv:
            return fun_inv(x)
        up = 1e6
        down = -1e6
        for _ in range(20):
            mid = (down + up) / 2
            cond = fun(mid) > x
            up = jnp.where(cond, mid, up)
            down = jnp.where(1.0 - cond, mid, down)
        return mid

    def train(rng):
        # INIT LR SCHEDULE
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac

        # INIT PHI
        phi_const = config["PHI"]
        if config["PHI_INV"]:
            phi_const_inv = config["PHI_INV"]
        else:
            phi_const_inv = None
        
        # INIT ACTOR-CRITIC NETWORK
        network = ActorCritic(
            action_dim = env.action_space(env_params).n
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Projection from actor_mean => distribution
        def projection_bregman(x, phi, phi_inv):
            a = phi_inv(1)
            b = phi_inv(1 / env.action_space(env_params).n)
            nu1 = a - jnp.max(x)
            nu2 = b - jnp.max(x)
            for i in range(10):
                nu = (nu1 + nu2) / 2
                cond = jnp.sum(nn.relu(phi(x + nu))) > 1
                nu1 = jnp.where(cond, nu, nu1)
                nu2 = jnp.where(1.0 - cond, nu, nu2)
            projected_x = nn.relu(phi(x + nu1))
            projected_x = projected_x / jnp.sum(projected_x)
            return projected_x, nu1

        # TRAIN LOOP
        def _update_step(runner_state, iteration):
            # DEFINE CURRENT PHI
            lr_actor = config["LR_ACTOR"]
            phi = phi_const
            phi_inv = phi_const_inv

            train_state, env_state, last_obs, rng, previous_lr_actor = runner_state
            runner_state = (train_state, env_state, last_obs, rng)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, lr_actor):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                actor_mean, value = network.apply(train_state.params, last_obs)
                actor_mean = actor_mean * lr_actor
                probs, nu1 = jax.vmap(projection_bregman, in_axes=(0, None, None))(
                    actor_mean, phi, phi_inv
                )
                nu1 = nu1[..., None]
                pi = distrax.Categorical(probs=probs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    log_prob,
                    actor_mean,
                    nu1,
                    last_obs,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, np.ones(config["NUM_STEPS"]) * lr_actor
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(carry, batch_info):
                    train_state, previous_lr_actor, lr_actor = carry
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(
                        params, traj_batch, gae, targets, previous_lr_actor, lr_actor
                    ):
                        # RERUN NETWORK
                        actor_mean, value = network.apply(params, traj_batch.obs)
                        actor_mean = actor_mean[
                            jnp.arange(len(actor_mean)), traj_batch.action
                        ]
                        actor_mean = actor_mean[..., None]

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        lambst = jnp.squeeze(traj_batch.nu1)
                        qhat = gae[..., None]
                        qhat = jnp.where(
                            config["NORMALIZE_ADV"],
                            (qhat - qhat.mean()) / (qhat.std() + 1e-8),
                            qhat,
                        )

                        preproj = traj_batch.actor_mean * previous_lr_actor
                        preproj = (
                            preproj[jnp.arange(len(preproj)), traj_batch.action]
                            + lambst
                        )
                        preproj = preproj[..., None]

                        objective = qhat + jax.lax.stop_gradient(
                            jnp.maximum(preproj, phi_inv(1e-6)) / lr_actor
                        )

                        loss_actor = (actor_mean - objective) ** 2

                        loss_actor = loss_actor.mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss
                        return total_loss, (value_loss, loss_actor)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        advantages,
                        targets,
                        previous_lr_actor,
                        lr_actor,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    carry = (train_state, previous_lr_actor, lr_actor)
                    return carry, total_loss

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    previous_lr_actor,
                    lr_actor,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                carry = (train_state, previous_lr_actor, lr_actor)
                carry, total_loss = jax.lax.scan(_update_minbatch, carry, minibatches)
                train_state, previous_lr_actor, lr_actor = carry
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    previous_lr_actor,
                    lr_actor,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
                previous_lr_actor,
                lr_actor,
            )
            update_state, _ = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            extra_info = {
                "metrics": traj_batch.info["returned_episode_returns"].mean() # average return for current iteration
            } 
            rng = update_state[4]

            runner_state = (train_state, env_state, last_obs, rng, lr_actor)
            return runner_state, extra_info

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1.0)
        runner_state, extra_info = jax.lax.scan(
            _update_step,
            runner_state,
            jnp.linspace(0.0, 1.0, int(config["NUM_UPDATES"])),
        )

        return extra_info["metrics"]

    return train


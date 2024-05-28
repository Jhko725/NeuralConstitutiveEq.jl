# %%
# ruff: noqa: F722
from pathlib import Path
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.random import PRNGKey
from jaxtyping import Float, Array
import equinox as eqx
import optimistix as optx
import diffrax
import matplotlib.pyplot as plt
import distrax
import scipy.interpolate as scinterp

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Spherical
from neuralconstitutive.integrate import integrate
from neuralconstitutive.plotting import plot_relaxation_fn
from neuralconstitutive.io import import_data
from neuralconstitutive.utils import normalize_forces, normalize_indentations
from neuralconstitutive.ting import _force_approach, _force_retract

jax.config.update("jax_enable_x64", True)

prob_E_inf = distrax.Uniform(-5, 2)
prob_E1 = distrax.Uniform(-5, 2)
prob_tau = distrax.Uniform(-5, 2)
prob_eps = distrax.Uniform(0.0, 0.1)

params_test = (jnp.asarray(0.0), StandardLinearSolid(1.0, 1.0, 1.0))

# %%
prob_E1.log_prob(jnp.asarray(-7))


# %%
@partial(eqx.filter_vmap, in_axes=eqx.if_array(0))
def log_prior_fn(params):
    sls, eps = params
    log_p_E_inf = prob_E_inf.log_prob(sls.E_inf)
    log_p_E1 = prob_E1.log_prob(sls.E1)
    log_p_tau = prob_tau.log_prob(sls.tau)
    log_p_eps = prob_eps.log_prob(eps)
    return log_p_E_inf + log_p_E1 + log_p_tau + log_p_eps


N_samples = 50

rng_key = jax.random.PRNGKey(0)
rng_key, *sub_keys = jax.random.split(rng_key, 5)
E_inf_samples = prob_E_inf.sample(seed=sub_keys[0], sample_shape=N_samples)
E1_samples = prob_E1.sample(seed=sub_keys[1], sample_shape=N_samples)
tau_samples = prob_tau.sample(seed=sub_keys[2], sample_shape=N_samples)
eps_samples = prob_eps.sample(seed=sub_keys[3], sample_shape=N_samples)
E_inf_samples.shape


@partial(eqx.filter_vmap, in_axes=(0, 0, 0, 0))
def make_params(E1, E_inf, tau, eps):
    return (StandardLinearSolid(E1, E_inf, tau), eps)


params_batched = make_params(E1_samples, E_inf_samples, tau_samples, eps_samples)


# %%
class PiecewiseCubic(diffrax.CubicInterpolation):

    def _interpret_t(self, t, left: bool) -> tuple:
        maxlen = self.ts_size - 2
        index = jnp.searchsorted(
            self.ts, t, side="left" if left else "right", method="compare_all"
        )
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part


def make_smoothed_cubic_spline(indentation, s=1.5e-4):
    tck = scinterp.splrep(indentation.time, indentation.depth, s=s)
    ppoly = scinterp.PPoly.from_spline(tck)
    cubic_interp = PiecewiseCubic(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))
    return cubic_interp


datadir = Path("data/abuhattum_iscience_2022/Interphase rep 2")
name = "interphase_speed 2_2nN"
(app, ret), (f_app_data, f_ret_data) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)
# f_ret_data = jnp.clip(f_ret_data, 0.0)
f_ret_data = jnp.trim_zeros(jnp.clip(f_ret_data, 0.0), "b")
ret = Indentation(ret.time[: len(f_ret_data)], ret.depth[: len(f_ret_data)])
(f_app_data, f_ret_data), _ = normalize_forces(f_app_data, f_ret_data)
(app, ret), (_, h_m) = normalize_indentations(app, ret)

tip = Spherical(2.5e-6 / h_m)

app_interp = make_smoothed_cubic_spline(app)
ret_interp = make_smoothed_cubic_spline(ret)


# %%
@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=eqx.if_array(0))
def log_likelihood_fn(params):
    sls, eps = params
    sls = jtu.tree_map(lambda x: 10**x, sls)
    f_app_pred = _force_approach(app.time, sls, app_interp, tip)
    f_ret_pred = _force_retract(ret.time, sls, (app_interp, ret_interp), tip)
    p_app = distrax.MultivariateNormalDiag(f_app_pred, jnp.ones_like(f_app_pred) * eps)
    p_ret = distrax.MultivariateNormalDiag(f_ret_pred, jnp.ones_like(f_ret_pred) * eps)
    log_p_app = p_app.log_prob(f_app_data)
    log_p_ret = p_ret.log_prob(f_ret_data)
    return log_p_app + log_p_ret


# %%

log_likelihood_fn(params_batched).block_until_ready()
# %%
log_prior_fn(params_batched)
# %%
log_likelihood_fn(params_batched)
# %%

# %%
import abc
from jaxtyping import PyTree, Array, Float
from typing import Callable, NamedTuple
import optimistix as optx


class AdamsTemperedSMCState(eqx.Module):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    lmbda: float
        Current value of the tempering parameter.

    """

    particles: PyTree[Float[Array, " M"]]
    weights: Float[Array, " M"]
    log_likelihood: Float[Array, " M"]
    lmbda: float
    ess: float

    @property
    def n_particles(self):
        return self.weights.shape[0]


def infer_num_particles(particles: PyTree[Float[Array, " M"]]):
    """Infer number of particles from the leading dimension of the first leaf node of particles.

    Adapted from init() at https://github.com/blackjax-devs/blackjax/blob/main/blackjax/smc/base.py
    """
    particles_flat, _ = jtu.tree_flatten(particles)
    return particles_flat[0].shape[0]


def init(particles: PyTree[Float[Array, " M"]]):
    n_particles = infer_num_particles(particles)
    weights = jnp.ones(n_particles) / n_particles
    log_likelihood = log_likelihood_fn(particles)
    return AdamsTemperedSMCState(
        particles, weights, log_likelihood, 0.0, jnp.float_(n_particles)
    )


def reweighting_fn(
    state: AdamsTemperedSMCState,
    ess_decrement_ratio: float = 1e-2,
    solver=optx.Bisection(rtol=1e-6, atol=1e-6, flip=True),
):

    def calculate_new_weights(lmbda_next):
        log_weights_next = (
            jnp.log(state.weights) + (lmbda_next - state.lmbda) * state.log_likelihood
        )
        return jax.nn.softmax(log_weights_next)

    def objective_fn(lmbda_next, ess_decrement_ratio):
        ess_target = state.ess * (1 - ess_decrement_ratio)
        ess = effective_sample_size(calculate_new_weights(lmbda_next))
        return ess / ess_target - 1

    def find_lmbda_next():
        lmbda_init = 0.5 * (1 + state.lmbda)
        sol = optx.root_find(
            objective_fn,
            solver,
            lmbda_init,
            ess_decrement_ratio,
            options={"lower": state.lmbda, "upper": 1.0},
            throw=False,
        )
        return sol.value

    lmbda_next = jax.lax.cond(
        objective_fn(1.0, ess_decrement_ratio) >= 0, lambda: 1.0, find_lmbda_next
    )
    weights_next = calculate_new_weights(lmbda_next)
    ess_next = effective_sample_size(weights_next)

    state_next = AdamsTemperedSMCState(
        state.particles, weights_next, state.log_likelihood, lmbda_next, ess_next
    )
    return state_next


@eqx.filter_jit
def step(
    rng_key: jax.random.PRNGKey,
    state: AdamsTemperedSMCState,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    target_ess_reduction: float = 1e-2,
    resample_ess_threshold: float = 0.5,
    mutation_ratio: float = 0.95,
):

    def calculate_weight_update(lmbda_diff):
        logweights_new = jnp.log(state.weights) + lmbda_diff * loglikelihood_fn(
            state.particles
        )
        return jax.nn.softmax(logweights_new)

    def calculate_lmbda_diff(state):
        ess_target = state.ess * (1 - target_ess_reduction)

        def objective(lmbda_diff, _):
            weights_new = calculate_weight_update(lmbda_diff)
            ess = effective_sample_size(weights_new)
            out = ess - ess_target
            return out

        solver = optx.Bisection(rtol=1e-6, atol=1e-6, flip=True)
        lmbda_diff_max = 1.0 - state.lmbda
        sol = optx.root_find(
            objective,
            solver,
            lmbda_diff_max,
            options=dict(
                lower=0.0,
                upper=lmbda_diff_max,
            ),
            throw=False,
        )
        return jnp.where(
            objective(lmbda_diff_max, None) >= 0, lmbda_diff_max, sol.value
        )

    ## Reweighting step
    lmbda_diff = calculate_lmbda_diff(state)
    weights_new = calculate_weight_update(lmbda_diff)
    lmbda_new = state.lmbda + lmbda_diff
    ess_new = effective_sample_size(weights_new)

    state = AdamsTemperedSMCState(
        state.particles, weights_new, state.log_likelihood, lmbda_new, ess_new
    )

    def should_resample(state) -> bool:
        return jnp.logical_or(
            state.ess < state.n_particles * resample_ess_threshold, state.lmbda >= 1
        )

    def resample_fn(resample_key, state):
        n_particles = state.n_particles
        idx_orig = jnp.arange(n_particles, dtype=jnp.int_)
        idx_new = jax.random.choice(resample_key, idx_orig, idx_orig.shape)
        particles_new = jtu.tree_map(lambda leaf: leaf[idx_new], state.particles)
        weights_new = jnp.ones(n_particles) / n_particles
        print("Resampled!")
        return AdamsTemperedSMCState(
            particles_new,
            weights_new,
            state.log_likelihood,
            state.lmbda,
            jnp.float_(n_particles),
        )

    def identity_fn(_, state):
        return state

    ## Resampling step
    rng_key, resample_key = jax.random.split(rng_key)
    state = jax.lax.cond(
        should_resample(state),
        resample_fn,
        identity_fn,
        resample_key,
        state,
    )

    ## Mutation step
    def should_mutate(state):
        return state.lmbda < 1

    def mutation_fn(mutation_key, state):

        def cond_fn(carry):
            _, _, _, n_accepted = carry
            return n_accepted < mutation_ratio * state.n_particles

        def body_fn(carry):
            n_loop, mutation_key, particles, n_accepted = carry
            mutation_key, sample_key, accept_key = jax.random.split(mutation_key, 3)
            proposal_covar = empirical_covariance(particles, state.weights) / (
                n_loop**2
            )
            particle_values, treedef = jtu.tree_flatten(particles)
            particle_values = jnp.stack(particle_values, axis=-1)
            proposal_dist = distrax.MultivariateNormalFullCovariance(
                particle_values, proposal_covar
            )

            particle_proposed_values = proposal_dist.sample(seed=sample_key)
            particle_proposed = jtu.tree_unflatten(
                treedef, list(particle_proposed_values.T)
            )

            logposterior_current = state.lmbda * loglikelihood_fn(
                particles
            ) + logprior_fn(particles)
            logposterior_proposed = state.lmbda * loglikelihood_fn(
                particle_proposed
            ) + logprior_fn(particle_proposed)

            A = jnp.clip(jnp.exp(logposterior_proposed - logposterior_current), max=1.0)
            idx_accept = A > jax.random.uniform(accept_key, A.shape)

            n_accepted = n_accepted + jnp.sum(idx_accept)
            particle_next = jtu.tree_map(
                lambda a, b: jnp.where(idx_accept, b, a),
                particles,
                particle_proposed,
            )
            return n_loop + 1, mutation_key, particle_next, n_accepted

        carry_init = (1, mutation_key, state.particles, 0.0)
        _, _, particles_new, _ = jax.lax.while_loop(cond_fn, body_fn, carry_init)
        return AdamsTemperedSMCState(
            particles_new, state.weights, state.log_likelihood, state.lmbda, state.ess
        )

    ## Mutation step
    rng_key, mutation_key = jax.random.split(rng_key)
    state = jax.lax.cond(
        should_mutate(state),
        mutation_fn,
        identity_fn,
        mutation_key,
        state,
    )
    return state


def effective_sample_size(weights) -> float:
    return 1.0 / jnp.sum(weights**2)


def empirical_covariance(particles, weights):
    particle_mean = jtu.tree_map(jnp.mean, particles)
    p_mean_flat, _ = jax.flatten_util.ravel_pytree(particle_mean)

    @eqx.filter_vmap
    def weighted_covariance(particle, weight: float):
        p_flat, _ = jax.flatten_util.ravel_pytree(particle)
        x = p_flat - p_mean_flat
        return weight * jnp.outer(x, x)

    cov = jnp.sum(weighted_covariance(particles, weights), axis=0)
    return cov / (1 - jnp.sum(weights**2))


# %%
class Bisection(optx.Bisection):

    def terminate(
        self,
        fn,
        y,
        args,
        options,
        state,
        tags,
    ):
        del fn, y, args, options
        f_small = jnp.abs(state.error) < self.atol
        return f_small, optx.RESULTS.successful


@eqx.filter_jit
def step2(
    rng_key: jax.random.PRNGKey,
    state: AdamsTemperedSMCState,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    ess_decrement_ratio: float = 1e-2,
    solver=Bisection(rtol=1e-6, atol=1e-6, flip=True),
    resample_ess_threshold: float = 0.5,
    mutation_ratio: float = 0.95,
):

    state = reweighting_fn(state, ess_decrement_ratio, solver)

    def should_resample(state) -> bool:
        return jnp.logical_or(
            state.ess < state.n_particles * resample_ess_threshold, state.lmbda >= 1
        )

    def resample_fn(resample_key, state):
        n_particles = state.n_particles
        idx_orig = jnp.arange(n_particles, dtype=jnp.int_)
        idx_new = jax.random.choice(resample_key, idx_orig, idx_orig.shape)
        particles_new = jtu.tree_map(lambda leaf: leaf[idx_new], state.particles)
        weights_new = jnp.ones(n_particles) / n_particles
        print("Resampled!")
        return AdamsTemperedSMCState(
            particles_new,
            weights_new,
            state.log_likelihood,
            state.lmbda,
            jnp.float_(n_particles),
        )

    def identity_fn(_, state):
        return state

    ## Resampling step
    rng_key, resample_key = jax.random.split(rng_key)
    state = jax.lax.cond(
        should_resample(state),
        resample_fn,
        identity_fn,
        resample_key,
        state,
    )

    ## Mutation step
    def should_mutate(state):
        return state.lmbda < 1

    def mutation_fn(mutation_key, state):

        def cond_fn(carry):
            _, _, _, n_accepted = carry
            return n_accepted < mutation_ratio * state.n_particles

        def body_fn(carry):
            n_loop, mutation_key, particles, n_accepted = carry
            mutation_key, sample_key, accept_key = jax.random.split(mutation_key, 3)
            proposal_covar = empirical_covariance(particles, state.weights) / (
                n_loop**2
            )
            particle_values, treedef = jtu.tree_flatten(particles)
            particle_values = jnp.stack(particle_values, axis=-1)
            proposal_dist = distrax.MultivariateNormalFullCovariance(
                particle_values, proposal_covar
            )

            particle_proposed_values = proposal_dist.sample(seed=sample_key)
            particle_proposed = jtu.tree_unflatten(
                treedef, list(particle_proposed_values.T)
            )

            logposterior_current = state.lmbda * loglikelihood_fn(
                particles
            ) + logprior_fn(particles)
            logposterior_proposed = state.lmbda * loglikelihood_fn(
                particle_proposed
            ) + logprior_fn(particle_proposed)

            A = jnp.clip(jnp.exp(logposterior_proposed - logposterior_current), max=1.0)
            idx_accept = A > jax.random.uniform(accept_key, A.shape)

            n_accepted = n_accepted + jnp.sum(idx_accept)
            particle_next = jtu.tree_map(
                lambda a, b: jnp.where(idx_accept, b, a),
                particles,
                particle_proposed,
            )
            return n_loop + 1, mutation_key, particle_next, n_accepted

        carry_init = (1, mutation_key, state.particles, 0.0)
        _, _, particles_new, _ = jax.lax.while_loop(cond_fn, body_fn, carry_init)
        return AdamsTemperedSMCState(
            particles_new, state.weights, state.log_likelihood, state.lmbda, state.ess
        )

    ## Mutation step
    rng_key, mutation_key = jax.random.split(rng_key)
    state = jax.lax.cond(
        should_mutate(state),
        mutation_fn,
        identity_fn,
        mutation_key,
        state,
    )
    return state


# %%
state = init(params_batched)
print(state.particles[0].E1)
print(state.ess)
print(state.lmbda)
# %%
import time

t_start = time.time()
state1 = step2(rng_key, state, log_prior_fn, log_likelihood_fn)
# jax.block_until_ready(state1)
print(state1.particles[0].E1)
print(state1.lmbda)
print(f"Elapsed time: {time.time()-t_start}")
# %%
import time

t_start = time.time()
state1 = step(rng_key, state, log_prior_fn, log_likelihood_fn)
print(state1.particles[0].E1)
print(state1.lmbda)
print(f"Elapsed time: {time.time()-t_start}")
# %%
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
    state = step(rng_key, state, log_prior, log_likelihood)
    state.particles[0].E1.block_until_ready()
# %%
from tqdm import tqdm

lmbda_list = [state.lmbda]
for i in tqdm(jnp.arange(10)):
    state = step(rng_key, state, log_prior, log_likelihood, target_ess_reduction=0.4)
    print(f"iter {i}, lmbda = {state.lmbda}, ess = {state.ess}")
    lmbda_list.append(state.lmbda)
# %%
state.particles[0].E1
# %%
params_batched[0].E1
# %%
state.lmbda
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(jnp.arange(len(lmbda_list)), jnp.stack(lmbda_list))
ax.set_yscale("log", base=10)
ax.set_xlabel("SMC Iteration")
ax.set_ylabel("Tempering parameter")

# %%

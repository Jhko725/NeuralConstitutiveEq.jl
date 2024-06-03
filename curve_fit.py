# %%
# ruff: noqa: F722
from functools import partial
from pathlib import Path

import distrax
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optimistix as optx
from jaxtyping import Array, Float

from neuralconstitutive.constitutive import (
    StandardLinearSolid,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.io import import_data
from neuralconstitutive.smoothing import make_smoothed_cubic_spline
from neuralconstitutive.ting import _force_approach, _force_retract
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import normalize_forces, normalize_indentations

jax.config.update("jax_enable_x64", True)

prob_E_inf = distrax.Uniform(0.0, 10)
prob_E1 = distrax.Uniform(0.0, 10)
prob_tau = distrax.Uniform(1e-5, 1e2)
prob_eps = distrax.Uniform(1e-6, 0.1)

params_test = (jnp.asarray(0.0), StandardLinearSolid(1.0, 1.0, 1.0))


# %%
#@partial(eqx.filter_vmap, in_axes=eqx.if_array(0))
def log_prior_fn(params):
    sls, eps = params
    log_p_E_inf = prob_E_inf.log_prob(sls.E_inf)
    log_p_E1 = prob_E1.log_prob(sls.E1)
    log_p_tau = prob_tau.log_prob(sls.tau)
    log_p_eps = prob_eps.log_prob(eps)
    return log_p_E_inf + log_p_E1 + log_p_tau + log_p_eps


N_samples = 500

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
#@eqx.filter_jit
#@partial(eqx.filter_vmap, in_axes=eqx.if_array(0))
def log_likelihood_fn(params):
    sls, eps = params
    # sls = jtu.tree_map(lambda x: 10**x, sls)
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
from typing import Callable

from jaxtyping import PyTree


class AdamsTemperedSMCState(eqx.Module):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    lmbda: float
        Current value of the tempering parameter.

    """

    particles: PyTree[Float[Array, " M"]]
    weights: Float[Array, " M"]
    log_prior: Float[Array, " M"]
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
    log_prior = log_prior_fn(particles)
    log_likelihood = log_likelihood_fn(particles)
    return AdamsTemperedSMCState(
        particles, weights, log_prior, log_likelihood, 0.0, jnp.float_(n_particles)
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
        state.particles,
        weights_next,
        state.log_prior,
        state.log_likelihood,
        lmbda_next,
        ess_next,
    )
    return state_next


def resampling_fn(
    resample_key, state, log_likelihood_fn, resample_ess_threshold: float = 0.5
):
    n_particles = state.n_particles
    should_resample = (state.ess < n_particles * resample_ess_threshold) | (
        state.lmbda >= 1
    )

    def resample():
        idx_orig = jnp.arange(n_particles, dtype=jnp.int_)
        idx_new = jax.random.choice(resample_key, idx_orig, idx_orig.shape)
        particles_new = jtu.tree_map(lambda leaf: leaf[idx_new], state.particles)
        weights_new = jnp.ones(n_particles) / n_particles
        log_likelihood_new = log_likelihood_fn(state.particles)
        return AdamsTemperedSMCState(
            particles_new,
            weights_new,
            state.log_prior,
            log_likelihood_new,
            state.lmbda,
            jnp.float_(n_particles),
        )

    return jax.lax.cond(should_resample, resample, lambda: state)


def mutating_fn(
    mutation_key, state, log_prior_fn, log_likelihood_fn, mutation_ratio: float = 0.95
):
    logposterior_current = state.lmbda * state.log_likelihood + state.log_prior

    def cond_fn(carry):
        n_loop, _, _, idx_accepted = carry
        accepted_ratio = jnp.sum(idx_accepted) / state.n_particles
        jax.debug.print("Mutated: {accepted_ratio}", accepted_ratio=accepted_ratio)
        return (accepted_ratio < mutation_ratio) & (n_loop <= 40)

    def body_fn(carry):
        n_loop, mutation_key, particles, accepted_prev = carry
        mutation_key, sample_key, accept_key = jax.random.split(mutation_key, 3)
        proposal_covar = empirical_covariance(particles, state.weights) / (n_loop**2)
        particle_values, treedef = jtu.tree_flatten(particles)
        particle_values = jnp.stack(particle_values, axis=-1)

        particle_proposed_values = jax.random.multivariate_normal(
            sample_key, particle_values, proposal_covar
        )

        particle_proposed = jtu.tree_unflatten(
            treedef, list(particle_proposed_values.T)
        )

        logposterior_proposed = state.lmbda * log_likelihood_fn(
            particle_proposed
        ) + log_prior_fn(particle_proposed)

        A = jnp.clip(jnp.exp(logposterior_proposed - logposterior_current), max=1.0)
        accepted = jax.random.bernoulli(accept_key, A)

        particle_next = jtu.tree_map(
            lambda a, b: jnp.where(accepted, b, a),
            particles,
            particle_proposed,
        )
        accepted_total = jnp.logical_or(accepted, accepted_prev)
        return n_loop + 1, mutation_key, particle_next, accepted_total

    def mutate():
        carry_init = (
            1,
            mutation_key,
            state.particles,
            jnp.zeros(state.n_particles, dtype=jnp.bool_),
        )

        _, _, particles_next, _ = jax.lax.while_loop(cond_fn, body_fn, carry_init)
        log_prior_next = log_prior_fn(particles_next)
        log_likelihood_next = log_likelihood_fn(particles_next)

        return AdamsTemperedSMCState(
            particles_next,
            state.weights,
            log_prior_next,
            log_likelihood_next,
            state.lmbda,
            state.ess,
        )

    return jax.lax.cond(state.lmbda < 1, mutate, lambda: state)


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
        state.particles,
        weights_new,
        state.log_prior,
        state.log_likelihood,
        lmbda_new,
        ess_new,
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
            state.log_prior,
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
            particles_new,
            state.weights,
            state.log_prior,
            state.log_likelihood,
            state.lmbda,
            state.ess,
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
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    ess_decrement_ratio: float = 1e-2,
    solver=Bisection(rtol=1e-6, atol=1e-6, flip=True),
    resample_ess_threshold: float = 0.5,
    mutation_ratio: float = 0.95,
):
    rng_key, resample_key, mutation_key = jax.random.split(rng_key, 3)

    state = reweighting_fn(state, ess_decrement_ratio, solver)
    state = resampling_fn(
        resample_key, state, log_likelihood_fn, resample_ess_threshold
    )
    state = mutating_fn(
        mutation_key, state, log_prior_fn, log_likelihood_fn, mutation_ratio
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
for i in tqdm(jnp.arange(100)):
    state = step2(
        rng_key, state, log_prior_fn, log_likelihood_fn, ess_decrement_ratio=0.2
    )
    print(f"iter {i}, lmbda = {state.lmbda}, ess = {state.ess}")
    lmbda_list.append(state.lmbda)
# %%
state.particles[0].E1
# %%
params_batched[0].E1
# %%
state.lmbda
# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(jnp.arange(len(lmbda_list)), jnp.stack(lmbda_list))
ax.set_yscale("log", base=10)
ax.set_xlabel("SMC Iteration")
ax.set_ylabel("Tempering parameter")

# %%
import blackjax
from blackjax.smc import resampling
from blackjax.smc import extend_params

def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """
    @eqx.filter_jit
    def cond(carry):
        i, state, _k = carry
        jax.debug.print("lmbda: {lmbda}", lmbda=state.lmbda)
        return state.lmbda < 1
    
    @eqx.filter_jit
    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        #jax.debug.print("E1: {E1}", E1 = state.particles[0].E1)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

inv_mass_matrix = jnp.eye(4)
hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
)

tempered = blackjax.adaptive_tempered_smc(
    log_prior_fn,
    log_likelihood_fn,
    blackjax.hmc.build_kernel(),
    blackjax.hmc.init,
    extend_params(N_samples, hmc_parameters),
    resampling.systematic,
    0.5,
    num_mcmc_steps=1,
)

rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
initial_smc_state = jax.random.multivariate_normal(
    init_key, jnp.zeros([1]), jnp.eye(1), (N_samples,)
)
initial_smc_state = tempered.init(params_batched)
#%%
n_iter, smc_samples = smc_inference_loop(sample_key, tempered.step, initial_smc_state)
print("Number of steps in the adaptive algorithm: ", n_iter.item())
# %%
smc_samples.particles[1]
# %%

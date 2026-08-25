'''
Random-walk Metropolis-Hastings sampler for drawing a single sample from an
unnormalized density over a rectangular domain.

Usage
-----
    sample = mcmc_sampler(density, space)
    sample, acceptance_rate = mcmc_sampler(density, space, return_diagnostics=True)

Parameters
----------
density : callable
    Unnormalized target density. Must accept a 1-D numpy array and return a
    non-negative scalar.
space : array-like of shape (ndim, 2)
    Bounds for each dimension, e.g. [[-pi, pi]] for 1-D.
n_iter : int
    Total number of MCMC iterations (burn-in + sampling).
burn_in : int
    Number of initial iterations discarded as burn-in.
proposal_std : float
    Initial standard deviation for the isotropic Gaussian proposal.
    Adapted every 50 steps to target an acceptance rate of 0.234.
seed : int or None
    Random seed for reproducibility.
return_diagnostics : bool
    If True, also return the post-burn-in acceptance rate as a second value.

Returns
-------
sample : np.ndarray of shape (ndim,)
    A single sample from the target density (last state of the chain).
acceptance_rate : float (only when return_diagnostics=True)
    Fraction of proposals accepted during the post-burn-in phase.
'''

import numpy as np


def mcmc_sampler(density, space, n_iter=2000, burn_in=500,
                 proposal_std=1.0, seed=None, return_diagnostics=False):
    rng = np.random.default_rng(seed)
    space = np.asarray(space)
    dimension = space.shape[0]

    # Initialise at a point with positive density
    x = rng.uniform(space[:, 0], space[:, 1])
    for _ in range(1000):
        if density(x) > 0:
            break
        x = rng.uniform(space[:, 0], space[:, 1])

    std = float(proposal_std)
    adapt_every = 50
    accepted_window = 0
    accepted_sampling = 0

    for i in range(n_iter):
        proposal = x + rng.normal(0, std, dimension)
        u = rng.uniform()
        ratio = density(proposal) / density(x)
        if u < min(1.0, ratio):
            x = proposal
            accepted_window += 1
            if i >= burn_in:
                accepted_sampling += 1

        # Scalar adaptation every adapt_every steps (during burn-in only)
        if i < burn_in and (i + 1) % adapt_every == 0:
            rate = accepted_window / adapt_every
            if rate < 0.2:
                std *= 0.9
            elif rate > 0.5:
                std *= 1.1
            accepted_window = 0

    sampling_steps = n_iter - burn_in
    acceptance_rate = accepted_sampling / sampling_steps if sampling_steps > 0 else 0.0

    if return_diagnostics:
        return x, acceptance_rate
    return x

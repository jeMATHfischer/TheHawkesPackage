"""Random-walk Metropolis-Hastings sampling over a rectangular domain."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["mcmc_sampler"]


def mcmc_sampler(
    density: Callable[[Any], float],
    space: Any,
    n_iter: int = 2000,
    burn_in: int = 500,
    proposal_std: float = 1.0,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    return_diagnostics: bool = False,
) -> Any:
    """Draw a single sample from an unnormalised density.

    Parameters
    ----------
    density : callable
        Unnormalised target density. Must accept a 1-D numpy array and return a
        non-negative scalar.
    space : array_like of shape (ndim, 2)
        Bounds for each dimension, e.g. ``[[-pi, pi]]`` in 1-D. Used to place
        the initial state; see the caveat below.
    n_iter : int
        Total number of iterations, burn-in included.
    burn_in : int
        Number of leading iterations discarded.
    proposal_std : float
        Initial standard deviation of the isotropic Gaussian proposal. Adapted
        every 50 steps during burn-in towards an acceptance rate of 0.234.
    seed : None, int or numpy.random.Generator
        Source of randomness. Passing a :class:`~numpy.random.Generator` shares
        one stream with the caller and advances it.

        .. versionchanged:: 0.2.0
           A ``Generator`` is now accepted, not only an integer seed.
    return_diagnostics : bool
        If True, also return the post-burn-in acceptance rate.

    Returns
    -------
    sample : numpy.ndarray of shape (ndim,)
        The final state of the chain.
    acceptance_rate : float
        Only when `return_diagnostics` is True: the fraction of proposals
        accepted after burn-in.

    Notes
    -----
    `space` bounds the *initial* draw only. The proposal is an unbounded random
    walk and out-of-domain proposals are not rejected, so the chain may wander
    outside `space`. For a periodic domain this is harmless -- the density is
    periodic too, and callers wrap the result -- but for a non-periodic density
    the returned marginal will be wrong. Constraining the walk to `space` is
    planned for 0.3.0.
    """
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

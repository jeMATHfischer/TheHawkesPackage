r"""Bayesian inference for the processes this package simulates.

Fits the parameters of a Hawkes model to observed events, in blocks as the data
arrives, by sequential Monte Carlo with MCMC rejuvenation. Temporal and
spatio-temporal alike, on any kernel, nonlinearity and
:class:`~hawkes_package.SpatialDomain` the simulator supports -- because the
likelihood is computed from the simulator's own intensity hooks rather than from
a second expression for the same thing.

The shape of a fit
------------------

.. code-block:: python

    import hawkes_package as hp
    from hawkes_package.inference import (
        ExponentialLogLikelihood,
        History,
        IndependentPrior,
        LogNormal,
        SMCSampler,
        exponential_model,
        ks_exponential,
        residuals,
    )

    truth = hp.ExponentialHawkes([2.0, 0.5, 1.0], rng=7)
    truth.simulate(800)
    history = History.from_simulation(truth)

    model = exponential_model()
    smc = SMCSampler(
        ExponentialLogLikelihood(model),
        IndependentPrior((LogNormal(0.5, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
        n_particles=256,
        rng=0,
    )
    for upto in history.times[99::100]:  # blocks, as data would arrive
        smc.update(history, float(upto))

    print(smc.cloud.summary())
    print(smc.diagnostics.summary())

What can go wrong quietly
-------------------------

Three things, and each has something in here that watches for it.

* **The compensator computed too small.** Every unit of
  :math:`\\int\\lambda` that goes missing is a penalty on a high intensity that
  never gets applied, so the background and the excitation both come back too
  large and the fit looks converged. Guarded by exact quadrature at the event
  jumps, an order-``P``-versus-``2P`` check, and the time-rescaling test in
  :mod:`~hawkes_package.inference.diagnostics`.
* **Particle degeneracy read as confidence.** A collapsed cloud reports a very
  tight posterior centred wherever the noise left it.
  :class:`~hawkes_package.inference.smc.SMCDiagnostics` records the effective
  sample size, the distinct-particle fraction and the rejuvenation acceptance
  at every block, and says so in its summary when any of them is bad.
* **The observation window guessed.** ``History.end`` has no default, because
  defaulting it silently changes the model.

What is not here
----------------

Partially observed or thinned data, which creates a genuine latent state and
needs a different algorithm rather than a different setting of this one;
multivariate and mutually-exciting processes; discrete marks. And
:mod:`hawkes_package.mcmc` is untouched -- it remains the spatial location
sampler on the Ogata correctness path, and inference has its own chain in
:mod:`hawkes_package.inference.mcmc`.

.. versionadded:: 0.5.0
"""

from .diagnostics import KSResult, ks_exponential, posterior_report, residuals
from .estimator import HawkesEstimator
from .evolution import Evolution, LiuWest, RandomWalkDrift, Static
from .families import (
    ConstantBase,
    ExponentialKernel,
    GammaKernel,
    GaussianSpatial,
    LinearNonlinearity,
    SoftPlusNonlinearity,
)
from .forecast import posterior_predictive, predictive_counts, predictive_interval
from .likelihood import (
    ExponentialLogLikelihood,
    History,
    LikelihoodState,
    LogLikelihood,
    SpatioTemporalLogLikelihood,
    TemporalLogLikelihood,
)
from .mcmc import ChainResult, batch_posterior, metropolis_chain
from .models import (
    ProcessModel,
    bell_shape_model,
    exponential_model,
    monotone_model,
    spatio_temporal_model,
)
from .parameters import Parameter, ParameterSpec
from .priors import (
    ConstrainedPrior,
    Gamma,
    IndependentPrior,
    LogNormal,
    Marginal,
    Normal,
    Prior,
    Uniform,
    stationarity,
)
from .resample import effective_sample_size, multinomial, systematic
from .smc import (
    ParticleCloud,
    SMCDiagnostics,
    SMCSampler,
    StepRecord,
    block_boundaries,
    fit_smc,
)

__all__ = [
    "ChainResult",
    "ConstantBase",
    "ConstrainedPrior",
    "Evolution",
    "ExponentialKernel",
    "ExponentialLogLikelihood",
    "Gamma",
    "GammaKernel",
    "GaussianSpatial",
    "HawkesEstimator",
    "History",
    "IndependentPrior",
    "KSResult",
    "LikelihoodState",
    "LinearNonlinearity",
    "LiuWest",
    "LogLikelihood",
    "LogNormal",
    "Marginal",
    "Normal",
    "Parameter",
    "ParameterSpec",
    "ParticleCloud",
    "Prior",
    "ProcessModel",
    "RandomWalkDrift",
    "SMCDiagnostics",
    "SMCSampler",
    "SoftPlusNonlinearity",
    "SpatioTemporalLogLikelihood",
    "Static",
    "StepRecord",
    "TemporalLogLikelihood",
    "Uniform",
    "batch_posterior",
    "bell_shape_model",
    "block_boundaries",
    "effective_sample_size",
    "exponential_model",
    "fit_smc",
    "ks_exponential",
    "metropolis_chain",
    "monotone_model",
    "multinomial",
    "posterior_predictive",
    "posterior_report",
    "predictive_counts",
    "predictive_interval",
    "residuals",
    "spatio_temporal_model",
    "stationarity",
    "systematic",
]

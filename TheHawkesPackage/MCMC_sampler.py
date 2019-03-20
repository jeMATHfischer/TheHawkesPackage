'''

This is a naive Markov Chain Monte Carlo based sampler to enable spatial sampling in the Spatio-temporal-Hawkes-Process
class. Might be used for other cases when wanting to sample.
'''

import numpy as np


def mcmc_sampler(density, space):
    x = np.random.choice(space)
    for i in range(1000):
        xi = np.random.normal(0,1)
        proposal = x + xi
        u = np.random.uniform(0,1)
        if u < min(1, density(proposal)/density(x)):
            x = proposal
    return x
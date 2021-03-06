'''

This is a naive Markov Chain Monte Carlo based sampler to enable spatial sampling in the Spatio-temporal-Hawkes-Process
class. Might be used for other cases when wanting to sample.
input space as np.array([[xlow, xhigh],[ylow,yhigh],[zlow,zhigh]]) everything larger is also possible.
If you want a smaller space only add corresponding lists.
Covariance of normal distribution might be necessary to adjust (still diagonal with constant values).
Assuming here that the generated random variables are in fact independent, in particular in the normal case.
Additionally, allowing 1000 steps is also generic, hoping that at that point the sample is sufficiently close to the
density.

'''

import numpy as np


def mcmc_sampler(density, space):
    dimension = len(space)
    print(dimension)
    x = np.array([])
    # if dimension == 2 and len(np.shape(space)) == 1:
    #     x = np.random.uniform(space[0],space[1])
    #     while density(x) == 0:
    #         x = np.random.uniform(space[0], space[1])
    #
    # else:
    for d in range(dimension):
        x = np.append(x,np.random.uniform(space[d][0],space[d][1]))

    while density(x) == 0:
        x = np.array([])
        for d in range(dimension):
            x = np.append(x, np.random.uniform(space[d][0], space[d][1]))

    for i in range(1000):
        xi = np.random.normal(0,1, dimension)
        proposal = x + xi
        u = np.random.uniform(0,1)
        if u < min(1, density(proposal)/density(x)):
            x = proposal

    print(density(x))
    return x

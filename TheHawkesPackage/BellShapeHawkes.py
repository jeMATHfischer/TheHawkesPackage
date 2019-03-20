#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hawkes process simulation that allows smooth kernels with one global maximum. Non-linearities need to be monotonously
growing.
"""

import numpy as np
from scipy.optimize import fmin


class BellShapeHawkes():

    def __init__(self, temporal, nonlinearity=lambda x: x + 2):
        self.temporal = temporal
        self.Events = np.array([0])
        self.Sim_num = 0
        self.nonlinearity = nonlinearity
        self.ext = fmin(lambda x: -self.temporal(x), 0, disp=False)

    def __bound(self, T):
        if T - self.Events[-1] < self.ext:
            return self.nonlinearity(np.sum(np.array([self.temporal(T - j)
                                                               for j in self.Events if j < T]))) + self.temporal(self.ext)
        else:
            return self.nonlinearity(np.sum(np.array([self.temporal(T - j)
                                                               for j in self.Events if j < T])))

    def propogate_by_amount(self, k):
        t = self.Events[-1]
        i = 0

        while i in range(k):
            upper_bd = self.__bound(t)

            u = np.random.rand(1)
            tau = -np.log(u) / upper_bd
            t = t + tau
            s = np.random.rand(1)

            if s <= self.nonlinearity(np.sum(np.array([self.temporal(t - j)
                                                               for j in self.Events if j < t]))) / upper_bd:
                self.Events = np.append(self.Events, t)
                i += 1

        if self.Sim_num == 0:
                self.Events = np.delete(self.Events, 0, 0)

        self.Sim_num += k

    def intensity_over_interval(self, x):
        y = np.sort(np.append(x, self.Events))
        return y, self.nonlinearity(np.array([np.sum(np.array([self.temporal(t - j)
                                                               for j in self.Events if j < t])) for t in y]))




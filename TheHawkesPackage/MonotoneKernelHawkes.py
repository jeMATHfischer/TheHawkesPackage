#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monotonously decreasing kernel & monotonously increasing non-linearity
"""

import numpy as np


class MonotoneKernelHawkes():

    def __init__(self, temporal, nonlinearity=lambda x: x + 2):
        self.temporal = temporal
        self.Events = np.array([0])
        self.Sim_num = 0
        self.nonlinearity = nonlinearity

    def propagate_by_k_events(self, k):
        t = self.Events[-1]
        i = 0

        while i in range(k):
            upper_bd = self.nonlinearity(np.sum(np.array([self.temporal(t - item) for item in self.Events if item < t])))

            u = np.random.rand(1)
            tau = -np.log(u) / upper_bd
            t = t + tau
            s = np.random.rand(1)

            if s <= self.nonlinearity(self.temporal(t - self.Events).sum())/ upper_bd:
                self.Events = np.append(self.Events, t)
                i += 1

        if self.Sim_num == 0:
            self.Events = np.delete(self.Events, 0, 0)

        self.Sim_num += k

    def intensity_over_interval(self, x):
        y = np.sort(np.append(x, self.Events))
        return y, self.nonlinearity(np.array([np.sum(np.array([self.temporal(t - j)
                                                               for j in self.Events if j < t])) for t in y]))




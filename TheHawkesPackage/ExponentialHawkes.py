#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic linear Hawkes process with exponential kernel. Give parameters a,b to see behaviour.
"""

import numpy as np


class ExponentialHawkes():

    def __init__(self, param):
        self.param = param
        self.temporal = lambda x: self.param[1]*np.exp(-self.param[2]*x)
        self.Events = np.array([0])
        self.Sim_num = 0

    def propagate_by_k_events(self, k):
        t = self.Events[-1]
        i = 0

        while i in range(k):
            upper_bd = self.param[0] + np.sum(np.array([self.temporal(self.Events[-1] - y) for y in self.Events]))
            print(upper_bd)

            u = np.random.rand(1)
            tau = -np.log(u) / upper_bd
            t = t + tau
            s = np.random.rand(1)

            if s <= (self.param[0] + np.sum(np.array([self.temporal(t - y) for y in self.Events if y < t])))/ upper_bd:
                self.Events = np.append(self.Events, t)
                i += 1

        if self.Sim_num == 0:
            self.Events = np.delete(self.Events, 0, 0)

        self.Sim_num += k

    def intensity_over_interval(self, x):
        y = np.sort(np.append(x, self.Events))
        return y, np.array([np.sum(np.array([self.temporal(t - j) for j in self.Events if j < t])) for t in y])



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:33:48 2018

@author: jens

Simulate a spatio-temporal Hawkes process on [0,2pi]x[0,infty) with periodic bdry conditions in space.
"""

import numpy as np
import random as rand

from scipy.integrate import quad
from scipy.optimize import fmin

from .MCMC_sampler import mcmc_sampler
# from RandomNumberGen import draw_random_number_from_pdf

rand.seed(42)


class Spatio_Temporal_Hawkes_Process():

    def __init__(self, Base, spatial, temporal, Space=[-np.pi, np.pi], monotone_temporal_kernel=False):
        self.Base = Base
        self.spatial = spatial
        self.temporal = temporal
        self.Events = np.array([[0], [0]])
        self.PoissEvent = np.array([])
        self.Sim_num = 0
        self.monotone_temporal_kernel = monotone_temporal_kernel
        self.Space = Space

        if monotone_temporal_kernel is not True:
            self.temporal_extremum = fmin(lambda t: -self.temporal(t), 0, disp=False)

    def propogate_by_amount(self, k):

        def periodizer(x):
            return (x - np.pi) % (2 * np.pi) - np.pi

        def periodized_spatial(x):
            return self.spatial(periodizer(x))

        def positive_periodized_spatial(x):
            return max(periodized_spatial(x), 0)

        for i in range(k):
            self.PoissEvent = np.append(self.PoissEvent, rand.expovariate(1))

        PoissProcess = np.cumsum(self.PoissEvent)
        mu = quad(self.Base, self.Space[0], self.Space[1])[0]

        for time in PoissProcess[self.Sim_num:]:

            T = self.Events[0, -1]

            def dist_temporal(t):
                return np.array([self.temporal(t - time) for time in self.Events[0, :] if time < t and time != 0])

            def dist_periodized_spatial(x,t):
                return np.array([periodized_spatial(x - location) for location in self.Events[1, :] if self.Events[0, self.Events[1, :] == location] < t])

            def positive_periodized_spatial(x,t):
                return np.array([max(0, periodized_spatial(x - location)) for location in self.Events[1, :] if self.Events[0, self.Events[1, :] == location] < t])

            positive_int_spatial = lambda t: np.array([quad(positive_periodized_spatial
                                                            , self.Space[0] - location, self.Space[1] - location)[0]
                                                       for location in self.Events[1, :] if self.Events[0, self.Events[1, :] == location] < t])

            def Space_free_temporal_at_time_t(t):

                At_t_dist_periodized_spatial = lambda x: dist_periodized_spatial(x, t)

                At_t_full_intensity = lambda x: max(0, self.Base(x) + np.multiply(dist_temporal(t), At_t_dist_periodized_spatial(x)).sum())

                At_t_integrated_intensity = quad(At_t_full_intensity, self.Space[0], self.Space[1])[0]

                return At_t_integrated_intensity

            ''' 
            Find upper bound M(t|H_t) for the intensity. Adjusted to kernels with one global extremum = maximum 
            '''

            def upper_bound_M(t):

                if t - self.Events[0, -1] < self.temporal_extremum:
                    return Space_free_temporal_at_time_t(t) + self.temporal(self.temporal_extremum)
                else:
                    return Space_free_temporal_at_time_t(t)

            while True:
                if self.monotone_temporal_kernel is True:
                    upper_bd = Space_free_temporal_at_time_t(T)
                else:
                    upper_bd = upper_bound_M(T)

                u = np.random.rand(1)
                tau = -np.log(u) / upper_bd
                T = T + tau
                s = np.random.rand(1)

                if s <= Space_free_temporal_at_time_t(T) / upper_bd:
                    EventTime = T
                    break

            Stopped_dist_periodized_spatial = lambda x: dist_periodized_spatial(x, EventTime)

            Stopped_full_intensity = lambda x: max(0, self.Base(x) + np.multiply(dist_temporal(EventTime), Stopped_dist_periodized_spatial(x)).sum())

            Norm_Stopped_full_intensity = quad(Stopped_full_intensity, self.Space[0], self.Space[1])[0]

            def Spatial_density(x):
                x = periodizer(x)
                return max(0, self.Base(x) + np.multiply(dist_temporal(EventTime), np.array([periodized_spatial(x - location) for location in self.Events[1, :] if self.Events[0, self.Events[1, :] == location] < EventTime])).sum())/Norm_Stopped_full_intensity

            EventSpace = periodizer(mcmc_sampler(Spatial_density, np.array([[-np.pi, np.pi]])))

            NewEvent = np.array([EventTime, EventSpace])
            print(self.Events)
            print(NewEvent)
            self.Events = np.append(self.Events, NewEvent, axis=1)

        if self.Sim_num == 0:
            self.Events = self.Events[:, 1:]

        self.Sim_num += k
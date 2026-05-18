#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-exports the improved MCMC sampler for use within the spatio_temporal subpackage.
"""

from ..MCMC_sampler import mcmc_sampler

__all__ = ["mcmc_sampler"]

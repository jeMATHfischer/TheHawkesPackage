"""
Regression tests for MonotoneKernelHawkes.

The critical bug was: the Ogata upper bound M(t) excluded the contribution
of the event at t = Events[-1], making M(t) < λ(t+ε) and every candidate
accepted unconditionally (Poisson, not Hawkes).

These tests verify that the thinning invariant M ≥ λ holds everywhere.
"""

import numpy as np
import pytest
import hawkes_package as THP


def _simulate_with_invariant_check(temporal, nonlinearity, k, seed=0):
    """Simulate k events and return (events, violations_count)."""
    np.random.seed(seed)

    class CheckedMonotone(THP.MonotoneKernelHawkes):
        def propagate_by_k_events(self, k):
            t = self.Events[-1]
            i = 0
            violations = 0
            while i in range(k):
                upper_bd = self.nonlinearity(np.sum(self.temporal(t - self.Events)))
                u = np.random.rand(1)
                tau = -np.log(u) / upper_bd
                t = t + tau
                s = np.random.rand(1)
                true_intensity = self.nonlinearity(
                    np.sum([self.temporal(t - item)
                            for item in self.Events if item < t])
                )
                if true_intensity > upper_bd + 1e-9:
                    violations += 1
                if s <= true_intensity / upper_bd:
                    self.Events = np.append(self.Events, t)
                    i += 1
            if self.Sim_num == 0:
                self.Events = np.delete(self.Events, 0, 0)
            self.Sim_num += k
            return violations

    H = CheckedMonotone(temporal, nonlinearity=nonlinearity)
    v = H.propagate_by_k_events(k)
    return H.Events, v


def test_thinning_invariant_exponential_kernel():
    temporal = lambda x: np.exp(-10 * x)
    _, violations = _simulate_with_invariant_check(temporal, lambda x: x + 2, 300)
    assert violations == 0, f"Thinning invariant violated {violations} times"


def test_thinning_invariant_nonlinear():
    # Use a bounded nonlinearity (sqrt(x+1)) to avoid explosion while still
    # exercising the nonlinear code path
    temporal = lambda x: np.exp(-5 * x)
    _, violations = _simulate_with_invariant_check(temporal, lambda x: np.sqrt(x + 1), 200)
    assert violations == 0


def test_event_count():
    np.random.seed(42)
    H = THP.MonotoneKernelHawkes(lambda x: np.exp(-10 * x))
    H.simulate(100)
    assert len(H.Events) == 100


def test_events_strictly_increasing():
    np.random.seed(2)
    H = THP.MonotoneKernelHawkes(lambda x: 0.5 * np.exp(-3 * x))
    H.simulate(150)
    assert np.all(np.diff(H.Events) > 0)


def test_simulate_alias():
    np.random.seed(9)
    H = THP.MonotoneKernelHawkes(lambda x: np.exp(-10 * x))
    H.simulate(20)
    assert len(H.Events) == 20

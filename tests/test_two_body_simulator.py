# tests/test_two_body_simulator.py

import numpy as np
import pytest

from data_gen.simulators.two_body import Simulator # <-- adjust this import to your layout


class DummyConstants:
    def __init__(self, G=1.0, m1=1.0, m2=1.0, r1=0.5, r2=0.5):
        self.G = G
        self.m1 = m1
        self.m2 = m2
        self.r1 = r1
        self.r2 = r2


class DummyConfig:
    def __init__(self, constants=None):
        self.constants = constants or DummyConstants()
        self.save_dir = 'tmp'
        self.t_span = (0, 20)
        self.num_timesteps = 1025


@pytest.fixture
def sim():
    """Create a Simulator with simple default constants."""
    cfg = DummyConfig()
    return Simulator(cfg)


def test_init_state_shapes(sim):
    """init_state should return 1D q, dq with 2 elements each."""
    q, dq = sim.init_state()
    assert q.shape == (2,)
    assert dq.shape == (2,)

    # r should be positive, angle in [0, 2π)
    r, theta = q
    assert r > 0
    assert 0.0 <= theta < 2 * np.pi


def test_circular_orbit_gives_zero_acceleration(sim):
    """
    With the circular-orbit initial conditions from init_state,
    the radial and angular accelerations should be (almost) zero.
    """
    q, dq = sim.init_state()
    r, theta = q
    dr, dtheta = dq

    y0 = np.array([r, theta, dr, dtheta], dtype=float)

    dydt = sim.dynamics(0.0, y0)
    dr_out, dtheta_out, ddr, ddtheta = dydt

    # First two components should match dq
    assert np.isclose(dr_out, dr)
    assert np.isclose(dtheta_out, dtheta)

    # For a perfect circular orbit, accelerations should be ~0
    assert np.isclose(ddr, 0.0, atol=1e-6)
    assert np.isclose(ddtheta, 0.0, atol=1e-6)


def test_lagrangian_shape(sim):
    """Lagrangian should return a 1D array of length N for batched q, dq."""
    # Build a small batch of states
    N = 5
    # r > 0, theta arbitrary
    r = np.linspace(0.5, 1.5, N)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    q = np.stack([r, theta], axis=-1)

    # Some arbitrary velocities
    dr = np.linspace(-0.1, 0.2, N)
    dtheta = np.linspace(0.3, 0.5, N)
    dq = np.stack([dr, dtheta], axis=-1)

    L = sim.lagrangian(q, dq)
    assert L.shape == (N,)


def test_theta_ignorable_coordinate(sim):
    """
    The Lagrangian should not depend on θ, only on r and velocities.
    Shifting θ by a constant should leave L unchanged.
    """
    N = 4
    r = np.full(N, 1.0)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    q = np.stack([r, theta], axis=-1)

    dr = np.zeros(N)
    dtheta = np.full(N, 0.4)
    dq = np.stack([dr, dtheta], axis=-1)

    L1 = sim.lagrangian(q, dq)

    # Shift theta by some constant
    q_shifted = q.copy()
    q_shifted[:, 1] += 1.234  # arbitrary constant shift

    L2 = sim.lagrangian(q_shifted, dq)

    assert np.allclose(L1, L2, atol=1e-8)


def test_lagrangian_gradients_match_finite_difference(sim):
    """
    Check that lagrangian_grad_q and lagrangian_grad_dq agree with
    numerical finite-difference estimates.
    """
    rng = np.random.default_rng(0)

    N = 5
    # r strictly positive to avoid division by zero in potential
    r = rng.uniform(0.5, 2.0, size=N)
    theta = rng.uniform(0.0, 2 * np.pi, size=N)
    q = np.stack([r, theta], axis=-1)

    dr = rng.uniform(-0.5, 0.5, size=N)
    dtheta = rng.uniform(0.1, 1.0, size=N)  # nonzero angular velocity
    dq = np.stack([dr, dtheta], axis=-1)

    eps = 1e-6

    # --- Numerical grad wrt q ---
    num_grad_q = np.zeros_like(q)
    for j in range(2):  # r, theta
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[:, j] += eps
        q_minus[:, j] -= eps
        L_plus = sim.lagrangian(q_plus, dq)
        L_minus = sim.lagrangian(q_minus, dq)
        num_grad_q[:, j] = (L_plus - L_minus) / (2 * eps)

    # --- Numerical grad wrt dq ---
    num_grad_dq = np.zeros_like(dq)
    for j in range(2):  # dr, dtheta
        dq_plus = dq.copy()
        dq_minus = dq.copy()
        dq_plus[:, j] += eps
        dq_minus[:, j] -= eps
        L_plus = sim.lagrangian(q, dq_plus)
        L_minus = sim.lagrangian(q, dq_minus)
        num_grad_dq[:, j] = (L_plus - L_minus) / (2 * eps)

    # Analytic grads
    grad_q = sim.lagrangian_grad_q(q, dq)
    grad_dq = sim.lagrangian_grad_dq(q, dq)

    assert np.allclose(grad_q, num_grad_q, rtol=1e-4, atol=1e-6)
    assert np.allclose(grad_dq, num_grad_dq, rtol=1e-4, atol=1e-6)

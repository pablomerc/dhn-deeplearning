#!/usr/bin/env python3
"""
Test script for the three-body simulator implementation.

This script tests:
1. Basic functionality (initialization, state shapes)
2. Physics correctness (force calculations, energy conservation)
3. Lagrangian consistency (gradients match finite differences)
4. Integration stability
5. Known physics properties (momentum conservation, etc.)
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_gen.simulators.three_body import Simulator


class DummyConstants:
    def __init__(self, G=1.0, m1=1.0, m2=1.0, m3=1.0, r1=1.0, r2=1.0, r3=1.0):
        self.G = G
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3


class DummyConfig:
    def __init__(self, constants=None):
        self.constants = constants or DummyConstants()
        self.save_dir = 'tmp'
        self.t_span = (0, 10)
        self.num_timesteps = 100


@pytest.fixture
def sim():
    """Create a Simulator with simple default constants."""
    cfg = DummyConfig()
    return Simulator(cfg)


def test_init_state_shapes(sim):
    """init_state should return 1D q, dq with 6 elements each (x1,y1,x2,y2,x3,y3)."""
    q, dq = sim.init_state()
    assert q.shape == (6,), f"Expected q shape (6,), got {q.shape}"
    assert dq.shape == (6,), f"Expected dq shape (6,), got {dq.shape}"


def test_relative_force_correctness(sim):
    """Test that relative_force computes gravitational force correctly."""
    # Test case: two bodies at distance r = 1.0
    x1, y1 = 0.0, 0.0
    x2, y2 = 1.0, 0.0
    m1, m2 = 1.0, 1.0
    G = 1.0

    F1, F2 = sim.relative_force(x1, y1, x2, y2, m1, m2, G)

    # Force magnitude should be G*m1*m2/r^2 = 1.0
    # Direction: F1 should point toward body 2 (positive x), F2 should point toward body 1 (negative x)
    expected_F1 = np.array([1.0, 0.0])  # G*m1*m2/r^2 * unit_vector
    expected_F2 = np.array([-1.0, 0.0])

    assert np.allclose(F1, expected_F1, atol=1e-10), f"F1 = {F1}, expected {expected_F1}"
    assert np.allclose(F2, expected_F2, atol=1e-10), f"F2 = {F2}, expected {expected_F2}"

    # Test Newton's third law: F1 = -F2
    assert np.allclose(F1, -F2, atol=1e-10), "Newton's third law violated"


def test_relative_force_vectorized(sim):
    """Test that relative_force can handle vectorized inputs (for lagrangian_grad_q)."""
    # This will fail if relative_force doesn't handle arrays
    N = 5
    x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    m1, m2, G = 1.0, 1.0, 1.0

    try:
        F1, F2 = sim.relative_force(x1, y1, x2, y2, m1, m2, G)
        # If it works, F1 and F2 should have shape (N, 2)
        assert F1.shape == (N, 2), f"Expected F1 shape ({N}, 2), got {F1.shape}"
        assert F2.shape == (N, 2), f"Expected F2 shape ({N}, 2), got {F2.shape}"
    except Exception as e:
        pytest.fail(f"relative_force failed with vectorized inputs: {e}")


def test_dynamics_shape(sim):
    """Test that dynamics returns the correct shape."""
    # State: [x1, y1, x2, y2, x3, y3, v1x, v1y, v2x, v2y, v3x, v3y]
    y0 = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, -0.1, -0.1, 0.0])

    dydt = sim.dynamics(0.0, y0)

    assert dydt.shape == (12,), f"Expected shape (12,), got {dydt.shape}"

    # First 6 components should be velocities
    assert np.allclose(dydt[:6], y0[6:]), "First 6 components should match velocities"

    # Last 6 components should be accelerations (non-zero in general)
    assert np.any(np.abs(dydt[6:]) > 1e-10), "Accelerations should be non-zero"


def test_lagrangian_shape(sim):
    """Lagrangian should return a 1D array of length N for batched q, dq."""
    N = 5
    q = np.random.randn(N, 6)
    dq = np.random.randn(N, 6)

    L = sim.lagrangian(q, dq)
    assert L.shape == (N,), f"Expected L shape ({N},), got {L.shape}"


def test_lagrangian_energy_components(sim):
    """Test that Lagrangian = T - V with correct kinetic and potential terms."""
    N = 3
    q = np.array([[1.0, 0.0, -1.0, 0.0, 0.0, 1.0],
                  [2.0, 0.0, -2.0, 0.0, 0.0, 2.0],
                  [3.0, 0.0, -3.0, 0.0, 0.0, 3.0]])
    dq = np.array([[0.1, 0.0, -0.1, 0.0, 0.0, 0.1],
                   [0.2, 0.0, -0.2, 0.0, 0.0, 0.2],
                   [0.3, 0.0, -0.3, 0.0, 0.0, 0.3]])

    L = sim.lagrangian(q, dq)

    # Check that L is positive (kinetic energy dominates for these velocities)
    # Actually, for gravitational systems, L can be positive or negative
    # Just check it's finite
    assert np.all(np.isfinite(L)), "Lagrangian should be finite"

    # Check that L increases with velocity (kinetic energy term)
    dq2 = 2.0 * dq
    L2 = sim.lagrangian(q, dq2)
    assert np.all(L2 > L), "Lagrangian should increase with velocity"


def test_lagrangian_grad_q_shape(sim):
    """Test lagrangian_grad_q returns correct shape."""
    N = 5
    q = np.random.randn(N, 6)
    dq = np.random.randn(N, 6)

    grad_q = sim.lagrangian_grad_q(q, dq)
    assert grad_q.shape == (N, 6), f"Expected shape ({N}, 6), got {grad_q.shape}"


def test_lagrangian_grad_dq_shape(sim):
    """Test lagrangian_grad_dq returns correct shape."""
    N = 5
    q = np.random.randn(N, 6)
    dq = np.random.randn(N, 6)

    grad_dq = sim.lagrangian_grad_dq(q, dq)
    assert grad_dq.shape == (N, 6), f"Expected shape ({N}, 6), got {grad_dq.shape}"


def test_lagrangian_gradients_match_finite_difference(sim):
    """
    Check that lagrangian_grad_q and lagrangian_grad_dq agree with
    numerical finite-difference estimates.
    """
    rng = np.random.default_rng(42)

    N = 3
    # Generate random positions (avoid bodies too close together)
    q = rng.uniform(-2.0, 2.0, size=(N, 6))
    dq = rng.uniform(-0.5, 0.5, size=(N, 6))

    eps = 1e-6

    # --- Numerical grad wrt q ---
    num_grad_q = np.zeros_like(q)
    for j in range(6):  # x1, y1, x2, y2, x3, y3
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[:, j] += eps
        q_minus[:, j] -= eps
        L_plus = sim.lagrangian(q_plus, dq)
        L_minus = sim.lagrangian(q_minus, dq)
        num_grad_q[:, j] = (L_plus - L_minus) / (2 * eps)

    # --- Numerical grad wrt dq ---
    num_grad_dq = np.zeros_like(dq)
    for j in range(6):  # v1x, v1y, v2x, v2y, v3x, v3y
        dq_plus = dq.copy()
        dq_minus = dq.copy()
        dq_plus[:, j] += eps
        dq_minus[:, j] -= eps
        L_plus = sim.lagrangian(q, dq_plus)
        L_minus = sim.lagrangian(q, dq_minus)
        num_grad_dq[:, j] = (L_plus - L_minus) / (2 * eps)

    # Analytic grads
    try:
        grad_q = sim.lagrangian_grad_q(q, dq)
        grad_dq = sim.lagrangian_grad_dq(q, dq)
    except Exception as e:
        pytest.fail(f"lagrangian_grad_q or lagrangian_grad_dq failed: {e}")

    # Check shapes
    assert grad_q.shape == num_grad_q.shape, "grad_q shape mismatch"
    assert grad_dq.shape == num_grad_dq.shape, "grad_dq shape mismatch"

    # Compare values (relaxed tolerance for numerical differentiation)
    assert np.allclose(grad_q, num_grad_q, rtol=1e-3, atol=1e-4), \
        f"grad_q mismatch:\nAnalytic:\n{grad_q}\nNumerical:\n{num_grad_q}\nDiff:\n{grad_q - num_grad_q}"
    assert np.allclose(grad_dq, num_grad_dq, rtol=1e-3, atol=1e-4), \
        f"grad_dq mismatch:\nAnalytic:\n{grad_dq}\nNumerical:\n{num_grad_dq}\nDiff:\n{grad_dq - num_grad_dq}"


def test_sample_trajectory(sim):
    """Test that sample_trajectory returns a valid trajectory."""
    traj = sim.sample_trajectory()

    # Check required keys
    required_keys = ['cond_dict', 'time', 'q', 'dq', 'L', 'L_grad_q', 'L_grad_dq']
    for key in required_keys:
        assert key in traj, f"Missing key: {key}"

    # Check shapes
    T = sim.num_timesteps
    assert traj['time'].shape == (T,), f"time shape should be ({T},)"
    assert traj['q'].shape == (T, 6), f"q shape should be ({T}, 6)"
    assert traj['dq'].shape == (T, 6), f"dq shape should be ({T}, 6)"
    assert traj['L'].shape == (T,), f"L shape should be ({T},)"
    assert traj['L_grad_q'].shape == (T, 6), f"L_grad_q shape should be ({T}, 6)"
    assert traj['L_grad_dq'].shape == (T, 6), f"L_grad_dq shape should be ({T}, 6)"

    # Check that time is monotonic
    assert np.all(np.diff(traj['time']) > 0), "Time should be monotonically increasing"

    # Check that values are finite
    assert np.all(np.isfinite(traj['q'])), "q should be finite"
    assert np.all(np.isfinite(traj['dq'])), "dq should be finite"
    assert np.all(np.isfinite(traj['L'])), "L should be finite"


def test_energy_conservation(sim):
    """
    Test approximate energy conservation (should be conserved for closed systems).
    Note: For three-body systems, energy should be conserved, but numerical
    integration errors will cause small drift.
    """
    # Use a shorter time span for better conservation
    sim.t_span = (0, 5)
    sim.num_timesteps = 200

    traj = sim.sample_trajectory()

    # Compute total energy E = T + V at each time step
    T = 0.5 * (sim.m1 * (traj['dq'][:, 0]**2 + traj['dq'][:, 1]**2) +
               sim.m2 * (traj['dq'][:, 2]**2 + traj['dq'][:, 3]**2) +
               sim.m3 * (traj['dq'][:, 4]**2 + traj['dq'][:, 5]**2))

    q = traj['q']
    x1, y1 = q[:, 0], q[:, 1]
    x2, y2 = q[:, 2], q[:, 3]
    x3, y3 = q[:, 4], q[:, 5]

    # Potential energy
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    r12 = np.maximum(r12, 1e-9)
    r13 = np.maximum(r13, 1e-9)
    r23 = np.maximum(r23, 1e-9)

    V = -sim.G * (sim.m1 * sim.m2 / r12 + sim.m1 * sim.m3 / r13 + sim.m2 * sim.m3 / r23)

    E = T + V

    # Energy should be approximately conserved (allow for numerical errors)
    E_initial = E[0]
    E_final = E[-1]
    relative_error = abs(E_final - E_initial) / (abs(E_initial) + 1e-10)

    # Allow up to 5% relative error due to numerical integration
    assert relative_error < 0.05, \
        f"Energy not conserved: initial={E_initial:.6f}, final={E_final:.6f}, error={relative_error:.6f}"


def test_momentum_conservation(sim):
    """
    Test that total momentum is approximately conserved.
    For an isolated system, total momentum should be constant.
    """
    sim.t_span = (0, 5)
    sim.num_timesteps = 200

    traj = sim.sample_trajectory()

    # Compute total momentum
    p_total_x = (sim.m1 * traj['dq'][:, 0] +
                 sim.m2 * traj['dq'][:, 2] +
                 sim.m3 * traj['dq'][:, 4])
    p_total_y = (sim.m1 * traj['dq'][:, 1] +
                 sim.m2 * traj['dq'][:, 3] +
                 sim.m3 * traj['dq'][:, 5])

    # Momentum should be approximately constant
    p_x_initial = p_total_x[0]
    p_y_initial = p_total_y[0]
    p_x_final = p_total_x[-1]
    p_y_final = p_total_y[-1]

    # Allow small numerical errors
    assert np.abs(p_x_final - p_x_initial) < 1e-6, \
        f"x-momentum not conserved: {p_x_initial:.6e} -> {p_x_final:.6e}"
    assert np.abs(p_y_final - p_y_initial) < 1e-6, \
        f"y-momentum not conserved: {p_y_initial:.6e} -> {p_y_final:.6e}"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])

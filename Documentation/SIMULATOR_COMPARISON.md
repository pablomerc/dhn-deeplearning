# Comprehensive Side-by-Side Comparison: Single vs Double Pendulum Simulators

## Overview
Both simulators inherit from `BaseSimulator` and implement the same interface methods. This document provides a detailed breakdown of every method, highlighting similarities and differences.

---

## 1. IMPORTS

### Single Pendulum
```python
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base_simulator import BaseSimulator
```

### Double Pendulum
```python
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base_simulator import BaseSimulator
```

**Analysis:** ✅ **IDENTICAL** - Both use the same imports.

---

## 2. CLASS DEFINITION

### Single Pendulum
```python
class Simulator(BaseSimulator):
```

### Double Pendulum
```python
class Simulator(BaseSimulator):
```

**Analysis:** ✅ **IDENTICAL** - Both inherit from `BaseSimulator`.

---

## 3. `__init__` METHOD

### Single Pendulum
```python
def __init__(self, config):
  super(Simulator, self).__init__(config)

  constants = config.constants
  self.g = self.load_constant(constants.g, 'g')
  self.m = self.load_constant(constants.m, 'm')
  self.l = self.load_constant(constants.l, 'l')
```

**Constants loaded:**
- `g` (gravity)
- `m` (mass - single value)
- `l` (length - single value)

### Double Pendulum
```python
def __init__(self, config):
  super(Simulator, self).__init__(config)

  constants = config.constants
  self.g = self.load_constant(constants.g, 'g')
  self.m1 = self.load_constant(constants.m1, 'm1')
  self.m2 = self.load_constant(constants.m2, 'm2')
  self.l1 = self.load_constant(constants.l1, 'l1')
  self.l2 = self.load_constant(constants.l2, 'l2')
```

**Constants loaded:**
- `g` (gravity)
- `m1` (mass of first bob)
- `m2` (mass of second bob)
- `l1` (length of first arm)
- `l2` (length of second arm)

**Analysis:** 
- ✅ **Same structure** - Both call `super()` and use `load_constant()` pattern
- ❌ **Different constants** - Single has 3 constants (g, m, l), Double has 5 constants (g, m1, m2, l1, l2)
- **Pattern:** Double extends single by adding indexed versions (m1/m2, l1/l2)

---

## 4. `init_state` METHOD

### Single Pendulum
```python
def init_state(self):
  q0 = np.pi / 2
  dq0 = 0.0
  return q0, dq0
```

**Returns:**
- `q0`: Scalar (single angle in radians)
- `dq0`: Scalar (single angular velocity)

**Initial conditions:**
- Angle: π/2 radians (90 degrees)
- Angular velocity: 0.0

### Double Pendulum
```python
def init_state(self):
  q0 = np.array([np.pi / 2, np.pi / 2 + 0])
  dq0 = np.array([0.0, 0.0])
  return q0, dq0
```

**Returns:**
- `q0`: NumPy array of shape (2,) - [q1, q2]
- `dq0`: NumPy array of shape (2,) - [dq1, dq2]

**Initial conditions:**
- q1: π/2 radians (90 degrees)
- q2: π/2 + 0 radians (90 degrees, same as q1)
- dq1: 0.0
- dq2: 0.0

**Analysis:**
- ❌ **Different return types** - Single returns scalars, Double returns arrays
- ✅ **Same initial angle** - Both start at π/2
- ✅ **Same initial velocity** - Both start at 0.0
- **Pattern:** Double generalizes single by using arrays instead of scalars

---

## 5. `dynamics` METHOD

### Single Pendulum
```python
def dynamics(self, t, y):
  q, dq = y
  ddq = -(self.g / self.l) * np.sin(q)
  return [dq, ddq]
```

**Input:**
- `t`: Time (not used in calculation)
- `y`: State vector [q, dq] (2 elements)

**Physics:**
- Simple harmonic motion equation: `ddq = -(g/l) * sin(q)`
- Only gravity and length affect acceleration

**Output:**
- Returns list: `[dq, ddq]` (2 elements)
- First element: velocity (dq)
- Second element: acceleration (ddq)

### Double Pendulum
```python
def dynamics(self, t, y):
  q1, q2, dq1, dq2 = y
  a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * np.cos(q1 - q2)
  a2 = (self.l1 / self.l2) * np.cos(q1 - q2)
  f1 = (
    - (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (dq2**2) * np.sin(q1 - q2)
    - (self.g / self.l1) * np.sin(q1)
  )
  f2 = (self.l1 / self.l2) * (dq1**2) * np.sin(q1 - q2) - (self.g / self.l2) * np.sin(q2)
  g1 = (f1 - a1 * f2) / (1 - a1 * a2)
  g2 = (f2 - a2 * f1) / (1 - a1 * a2)
  ddq1 = g1
  ddq2 = g2
  return np.array([dq1, dq2, ddq1, ddq2])
```

**Input:**
- `t`: Time (not used in calculation)
- `y`: State vector [q1, q2, dq1, dq2] (4 elements)

**Physics:**
- Coupled differential equations
- Uses coupling coefficients `a1` and `a2` based on length ratios and mass ratios
- `f1` and `f2` are intermediate force terms:
  - `f1`: Includes coupling term from dq2² and gravity term for q1
  - `f2`: Includes coupling term from dq1² and gravity term for q2
- `g1` and `g2` solve the coupled system using matrix inversion approach
- Final accelerations `ddq1` and `ddq2` account for mutual coupling

**Output:**
- Returns NumPy array: `[dq1, dq2, ddq1, ddq2]` (4 elements)
- First two elements: velocities (dq1, dq2)
- Last two elements: accelerations (ddq1, ddq2)

**Analysis:**
- ❌ **Completely different complexity** - Single is simple ODE, Double is coupled system
- ❌ **Different input/output sizes** - Single: 2→2, Double: 4→4
- ❌ **Different return types** - Single returns list, Double returns NumPy array
- **Pattern:** Double adds coupling terms and solves coupled equations

---

## 6. `lagrangian` METHOD

### Single Pendulum
```python
def lagrangian(self, q, dq):
  T = 0.5 * self.m * (self.l ** 2) * (dq ** 2)    # Kinetic energy
  V = self.m * self.g * self.l * (1 - np.cos(q))    # Potential energy
  return T - V
```

**Input:**
- `q`: Shape (N, 1) - array of angles
- `dq`: Shape (N, 1) - array of angular velocities

**Kinetic Energy (T):**
- `T = 0.5 * m * l² * dq²`
- Simple rotational kinetic energy

**Potential Energy (V):**
- `V = m * g * l * (1 - cos(q))`
- Height-based potential energy (zero at q=0, maximum at q=π)

**Output:**
- Scalar or array: `T - V` (Lagrangian)

### Double Pendulum
```python
def lagrangian(self, q, dq):
  q1, q2 = q[:, 0], q[:, 1]
  dq1, dq2 = dq[:, 0], dq[:, 1]
  # kinetic energy (T)
  T1 = 0.5 * self.m1 * (self.l1 * dq1)**2
  T2 = 0.5 * self.m2 * (
    (self.l1 * dq1)**2 + (self.l2 * dq2)**2 +
    2 * self.l1 * self.l2 * dq1 * dq2 * np.cos(q1 - q2)
  )
  T = T1 + T2

  # potential energy (V)
  y1 = -self.l1 * np.cos(q1)
  y2 = y1 - self.l2 * np.cos(q2)
  V = self.m1 * self.g * y1 + self.m2 * self.g * y2
  return T - V
```

**Input:**
- `q`: Shape (N, 2) - array of [q1, q2] angles
- `dq`: Shape (N, 2) - array of [dq1, dq2] angular velocities

**Kinetic Energy (T):**
- `T1`: Kinetic energy of first bob = `0.5 * m1 * (l1 * dq1)²`
- `T2`: Kinetic energy of second bob = `0.5 * m2 * [(l1*dq1)² + (l2*dq2)² + 2*l1*l2*dq1*dq2*cos(q1-q2)]`
  - The cross term `2*l1*l2*dq1*dq2*cos(q1-q2)` accounts for coupling between the two pendulums
- `T = T1 + T2`

**Potential Energy (V):**
- `y1`: Vertical position of first bob = `-l1 * cos(q1)`
- `y2`: Vertical position of second bob = `y1 - l2 * cos(q2)` (relative to first bob)
- `V = m1 * g * y1 + m2 * g * y2`

**Output:**
- Scalar or array: `T - V` (Lagrangian)

**Analysis:**
- ❌ **Different input shapes** - Single: (N,1), Double: (N,2)
- ❌ **Different complexity** - Single has simple terms, Double has coupling terms
- ✅ **Same structure** - Both compute T and V separately, then return T-V
- **Pattern:** Double adds coupling terms in kinetic energy and accounts for relative positions

---

## 7. `lagrangian_grad_q` METHOD

### Single Pendulum
```python
def lagrangian_grad_q(self, q, dq):
  """Gradient of L with respect to q."""
  return -self.m * self.g * self.l * np.sin(q)
```

**Input:**
- `q`: Shape (N, 1) - array of angles
- `dq`: Shape (N, 1) - array of angular velocities (not used)

**Calculation:**
- `∂L/∂q = -m * g * l * sin(q)`
- Only potential energy contributes (kinetic energy doesn't depend on q)

**Output:**
- Shape (N, 1) or scalar: gradient with respect to q

### Double Pendulum
```python
def lagrangian_grad_q(self, q, dq):
  q1, q2 = q[:, 0], q[:, 1]
  dq1, dq2 = dq[:, 0], dq[:, 1]
  grad_q1 = (
    - self.m2 * self.l1 * self.l2 * dq1 * dq2 * np.sin(q1 - q2)
    - (self.m1 + self.m2) * self.g * self.l1 * np.sin(q1)
  )
  grad_q2 = (
    self.m2 * self.l1 * self.l2 * dq1 * dq2 * np.sin(q1 - q2)
    - self.m2 * self.g * self.l2 * np.sin(q2)
  )
  return np.stack([grad_q1, grad_q2], axis=-1)
```

**Input:**
- `q`: Shape (N, 2) - array of [q1, q2] angles
- `dq`: Shape (N, 2) - array of [dq1, dq2] angular velocities

**Calculation:**
- `grad_q1`: 
  - Coupling term: `-m2 * l1 * l2 * dq1 * dq2 * sin(q1 - q2)`
  - Gravity term: `-(m1 + m2) * g * l1 * sin(q1)`
- `grad_q2`:
  - Coupling term: `+m2 * l1 * l2 * dq1 * dq2 * sin(q1 - q2)` (opposite sign!)
  - Gravity term: `-m2 * g * l2 * sin(q2)`

**Output:**
- Shape (N, 2): `[grad_q1, grad_q2]` stacked along last axis

**Analysis:**
- ❌ **Different input/output shapes** - Single: (N,1)→(N,1), Double: (N,2)→(N,2)
- ❌ **Different complexity** - Single has no coupling, Double has coupling terms
- ❌ **Different return format** - Single returns directly, Double uses `np.stack()`
- ✅ **Same concept** - Both compute ∂L/∂q
- **Pattern:** Double adds coupling terms that depend on both q and dq

---

## 8. `lagrangian_grad_dq` METHOD

### Single Pendulum
```python
def lagrangian_grad_dq(self, q, dq):
  """Gradient of L with respect to dq."""
  return self.m * (self.l ** 2) * dq
```

**Input:**
- `q`: Shape (N, 1) - array of angles (not used)
- `dq`: Shape (N, 1) - array of angular velocities

**Calculation:**
- `∂L/∂dq = m * l² * dq`
- This is the generalized momentum (canonical momentum)

**Output:**
- Shape (N, 1) or scalar: gradient with respect to dq

### Double Pendulum
```python
def lagrangian_grad_dq(self, q, dq):
  q1, q2 = q[:, 0], q[:, 1]
  dq1, dq2 = dq[:, 0], dq[:, 1]
  grad_dq1 = (
    (self.m1 + self.m2) * self.l1**2 * dq1 +
    self.m2 * self.l1 * self.l2 * dq2 * np.cos(q1 - q2)
  )
  grad_dq2 = (
    self.m2 * self.l2**2 * dq2 +
    self.m2 * self.l1 * self.l2 * dq1 * np.cos(q1 - q2)
  )
  return np.stack([grad_dq1, grad_dq2], axis=-1)
```

**Input:**
- `q`: Shape (N, 2) - array of [q1, q2] angles
- `dq`: Shape (N, 2) - array of [dq1, dq2] angular velocities

**Calculation:**
- `grad_dq1`:
  - Self term: `(m1 + m2) * l1² * dq1`
  - Coupling term: `m2 * l1 * l2 * dq2 * cos(q1 - q2)`
- `grad_dq2`:
  - Self term: `m2 * l2² * dq2`
  - Coupling term: `m2 * l1 * l2 * dq1 * cos(q1 - q2)`

**Output:**
- Shape (N, 2): `[grad_dq1, grad_dq2]` stacked along last axis

**Analysis:**
- ❌ **Different input/output shapes** - Single: (N,1)→(N,1), Double: (N,2)→(N,2)
- ❌ **Different complexity** - Single is linear in dq, Double has coupling terms
- ❌ **Different return format** - Single returns directly, Double uses `np.stack()`
- ✅ **Same concept** - Both compute ∂L/∂dq (generalized momentum)
- **Pattern:** Double adds coupling terms that depend on both q and dq

---

## 9. `sample_trajectory` METHOD

### Single Pendulum
```python
def sample_trajectory(self):
  q0, dq0 = self.init_state()

  # Initial conditions: [initial angle, initial angular velocity]
  y0 = [q0, dq0]
          
  # Solve the differential equation for this length
  sol = solve_ivp(self.dynamics, self.t_span, y0, dense_output=True, max_step=0.01)
  
  # Get time points and solutions (q, dq)
  t_vals = np.linspace(self.t_span[0], self.t_span[1], self.num_timesteps)
  y_vals = sol.sol(t_vals)
  q_vals, dq_vals = y_vals
  q_vals, dq_vals = q_vals[:, None], dq_vals[:, None]
  
  # Compute Lagrangian at each time step
  L_vals = self.lagrangian(q_vals, dq_vals)
  L_grad_q = self.lagrangian_grad_q(q_vals, dq_vals)
  L_grad_dq = self.lagrangian_grad_dq(q_vals, dq_vals)
  
  # Save the trajectory, length, and Lagrangian
  trajectory = {
    'cond_dict': self.cond_dict,
    'time': t_vals,
    'q': q_vals,
    'dq': dq_vals,
    'L': L_vals,
    'L_grad_q': L_grad_q,
    'L_grad_dq': L_grad_dq,
  }
 
  return trajectory
```

**Key Steps:**
1. Get initial state: `q0, dq0 = self.init_state()`
2. Create initial condition vector: `y0 = [q0, dq0]` (list, 2 elements)
3. Solve ODE: `solve_ivp()` with `dense_output=True`, `max_step=0.01`
4. Extract solutions: `y_vals = sol.sol(t_vals)` then unpack directly
5. Reshape: `q_vals[:, None], dq_vals[:, None]` to ensure (N, 1) shape
6. Compute Lagrangian quantities
7. Return dictionary with all trajectory data

**Output shape handling:**
- `y_vals` from `sol.sol()` is unpacked directly: `q_vals, dq_vals = y_vals`
- Then reshaped to (N, 1) using `[:, None]`

### Double Pendulum
```python
def sample_trajectory(self):
  q0, dq0 = self.init_state()

  # Initial conditions: [initial angle, initial angular velocity]
  y0 = np.concatenate([q0, dq0], axis=0)
          
  # Solve the differential equation for this length
  sol = solve_ivp(self.dynamics, self.t_span, y0, dense_output=True, max_step=0.01)
  
  # Get time points and solutions (q, dq)
  t_vals = np.linspace(self.t_span[0], self.t_span[1], self.num_timesteps)
  y_vals = sol.sol(t_vals)
  q_vals, dq_vals = y_vals[:2].transpose((1, 0)), y_vals[2:].transpose((1, 0))
  
  # Compute Lagrangian at each time step
  L_vals = self.lagrangian(q_vals, dq_vals)
  L_grad_q = self.lagrangian_grad_q(q_vals, dq_vals)
  L_grad_dq = self.lagrangian_grad_dq(q_vals, dq_vals)
  
  # Save the trajectory, length, and Lagrangian
  trajectory = {
    'cond_dict': self.cond_dict,
    'time': t_vals,
    'q': q_vals,
    'dq': dq_vals,
    'L': L_vals,
    'L_grad_q': L_grad_q,
    'L_grad_dq': L_grad_dq,
  }
 
  return trajectory
```

**Key Steps:**
1. Get initial state: `q0, dq0 = self.init_state()`
2. Create initial condition vector: `y0 = np.concatenate([q0, dq0], axis=0)` (NumPy array, 4 elements)
3. Solve ODE: `solve_ivp()` with `dense_output=True`, `max_step=0.01` (same parameters)
4. Extract solutions: `y_vals = sol.sol(t_vals)` then slice and transpose
   - `q_vals = y_vals[:2].transpose((1, 0))` - first 2 rows, transpose to (N, 2)
   - `dq_vals = y_vals[2:].transpose((1, 0))` - last 2 rows, transpose to (N, 2)
5. Compute Lagrangian quantities (same as single)
6. Return dictionary with all trajectory data

**Output shape handling:**
- `y_vals` from `sol.sol()` has shape (4, N) - 4 states × N time steps
- Sliced: `y_vals[:2]` gives q1, q2 (shape 2×N), `y_vals[2:]` gives dq1, dq2 (shape 2×N)
- Transposed: `.transpose((1, 0))` converts (2, N) → (N, 2)

**Analysis:**
- ✅ **Same overall structure** - Both follow identical workflow
- ❌ **Different initial condition format** - Single uses list, Double uses `np.concatenate()`
- ❌ **Different solution extraction** - Single unpacks directly, Double slices and transposes
- ❌ **Different output shapes** - Single: (N,1), Double: (N,2)
- ✅ **Same ODE solver parameters** - Both use `dense_output=True`, `max_step=0.01`
- ✅ **Same trajectory dictionary structure** - Both return same keys
- **Pattern:** Double needs array concatenation and more complex reshaping

---

## 10. `visualize` METHOD

### Single Pendulum

#### Plot 1: Time vs Angle (q-t curve)
```python
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(traj['time'], traj['q'][:, 0], label=f'q1', color=self.color_palette[4])
plt.title(f'Angular displacement vs Time, l={self.l:.2f} m')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.tight_layout()
save_path = os.path.join(self.vis_dir, f'q-t_{i_data}.png')
plt.savefig(save_path)
```

**Features:**
- Single line plot (q1 only)
- Color: `color_palette[4]` (blue)
- Title includes single length value

#### Plot 2: Angle vs Angular Velocity (dq-q curve)
```python
plt.figure(figsize=(10, 6))
plt.plot(traj['q'][:, 0], traj['dq'][:, 0], label='q1', color=self.color_palette[4])
plt.title(f'Angular velocity vs Angular displacement, l={self.l:.2f} m')
plt.xlabel('Angle [rad]')
plt.ylabel('Angular velocity [rad/s]')
plt.legend()
plt.tight_layout()
save_path = os.path.join(self.vis_dir, f'dq-q_{i_data}.png')
plt.savefig(save_path)
```

**Features:**
- Single line plot (q1 vs dq1)
- Color: `color_palette[4]` (blue)
- Title includes single length value

#### Plot 3: Animation (video)
```python
t_per_frame = 5

x = self.l * np.sin(traj['q'][:, 0])
y = - self.l * np.cos(traj['q'][:, 0])

fig, ax = plt.subplots()
l_max = self.l
ax.set_xlim(-l_max-0.1, l_max+0.1)
ax.set_ylim(-l_max-0.1, 0.1)
ax.set_aspect('equal')
title = ax.text(0.5, 1.05, '', ha='center', va='center', transform=ax.transAxes)

ax.axis("off")
plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Pendulum line and bob
trace, = ax.plot([], [], '-', lw=2, alpha=0.7, color=self.color_palette[2])
line, = ax.plot([], [], 'o-', lw=2, color=self.color_palette[0])

# Trace history for the second pendulum bob
trace_x, trace_y = [], []   

def update(frame):
  t = frame * t_per_frame
  line.set_data([0, x[t]], [0, y[t]])

  # Append the trace of the second bob
  trace_x.append(x[t])
  trace_y.append(y[t])
  trace.set_data(trace_x, trace_y)

  title.set_text(f'Time step: {t:04d}')

  return line, trace, title

# Create the animation
ani = FuncAnimation(fig, update, frames=self.num_timesteps // t_per_frame, blit=True)
save_path = os.path.join(self.vis_dir, f'video_{i_data}.gif')
ani.save(save_path, writer='pillow', fps=15)
```

**Features:**
- Single pendulum: `x = l * sin(q)`, `y = -l * cos(q)`
- `l_max = self.l` (single length)
- Y-limits: `-l_max-0.1` to `0.1`
- Line data: `[0, x[t]]`, `[0, y[t]]` (2 points: origin to bob)
- Trace tracks single bob position
- Comment says "second pendulum bob" but it's actually the single bob (copy-paste artifact)

### Double Pendulum

#### Plot 1: Time vs Angle (q-t curve)
```python
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(traj['time'], traj['q'][:, 0], label=f'q1', color=self.color_palette[1])
plt.plot(traj['time'], traj['q'][:, 1], label=f'q2', color=self.color_palette[4])
plt.title(f'Angular displacement vs Time, l1={self.l1:.2f} m, l2={self.l2:.2f} m')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.tight_layout()
save_path = os.path.join(self.vis_dir, f'q-t_{i_data}.png')
plt.savefig(save_path)
```

**Features:**
- Two line plots (q1 and q2)
- Colors: `color_palette[1]` (orange) for q1, `color_palette[4]` (blue) for q2
- Title includes both length values

#### Plot 2: Angle vs Angular Velocity (dq-q curve)
```python
plt.figure(figsize=(10, 6))
plt.plot(traj['q'][:, 0], traj['dq'][:, 0], label='q1', color=self.color_palette[1])
plt.plot(traj['q'][:, 1], traj['dq'][:, 1], label='q2', color=self.color_palette[4])
plt.title(f'Angular velocity vs Angular displacement, l1={self.l1:.2f} m, l2={self.l2:.2f} m')
plt.xlabel('Angle [rad]')
plt.ylabel('Angular velocity [rad/s]')
plt.legend()
plt.tight_layout()
save_path = os.path.join(self.vis_dir, f'dq-q_{i_data}.png')
plt.savefig(save_path)
```

**Features:**
- Two line plots (q1 vs dq1 and q2 vs dq2)
- Colors: `color_palette[1]` (orange) for q1, `color_palette[4]` (blue) for q2
- Title includes both length values

#### Plot 3: Animation (video)
```python
t_per_frame = 5

x1 = self.l1 * np.sin(traj['q'][:, 0])
y1 = - self.l1 * np.cos(traj['q'][:, 0])
x2 = x1 + self.l2 * np.sin(traj['q'][:, 1])
y2 = y1 - self.l2 * np.cos(traj['q'][:, 1])

fig, ax = plt.subplots()
l_max = self.l1 + self.l2
ax.set_xlim(-l_max-0.1, l_max+0.1)
ax.set_ylim(-l_max-0.1, 0.5)
ax.set_aspect('equal')
title = ax.text(0.5, 1.05, '', ha='center', va='center', transform=ax.transAxes)

ax.axis("off")
plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Pendulum line and bob
trace, = ax.plot([], [], '-', lw=2, alpha=0.7, color=self.color_palette[2])
line, = ax.plot([], [], 'o-', lw=2, color=self.color_palette[0])

# Trace history for the second pendulum bob
trace_x, trace_y = [], []   

def update(frame):
  t = frame * t_per_frame
  line.set_data([0, x1[t], x2[t]], [0, y1[t], y2[t]])

  # Append the trace of the second bob
  trace_x.append(x2[t])
  trace_y.append(y2[t])
  trace.set_data(trace_x, trace_y)

  title.set_text(f'Time step: {t:04d}')

  return line, trace, title

# Create the animation
ani = FuncAnimation(fig, update, frames=self.num_timesteps // t_per_frame, blit=True)
save_path = os.path.join(self.vis_dir, f'video_{i_data}.gif')
ani.save(save_path, writer='pillow', fps=15)
```

**Features:**
- Two pendulums: 
  - First: `x1 = l1 * sin(q1)`, `y1 = -l1 * cos(q1)`
  - Second: `x2 = x1 + l2 * sin(q2)`, `y2 = y1 - l2 * cos(q2)` (relative to first)
- `l_max = self.l1 + self.l2` (sum of lengths)
- Y-limits: `-l_max-0.1` to `0.5`
- Line data: `[0, x1[t], x2[t]]`, `[0, y1[t], y2[t]]` (3 points: origin → bob1 → bob2)
- Trace tracks second bob position only

**Analysis:**
- ✅ **Same overall structure** - Both create 3 visualizations
- ❌ **Different number of plots** - Single: 1 line per plot, Double: 2 lines per plot
- ❌ **Different colors** - Single uses `[4]`, Double uses `[1]` and `[4]`
- ❌ **Different animation complexity** - Single: 2 points, Double: 3 points
- ❌ **Different coordinate calculations** - Single: direct, Double: relative
- ✅ **Same animation parameters** - Both use `t_per_frame=5`, `fps=15`
- **Pattern:** Double extends single by adding second pendulum visualization

---

## SUMMARY: KEY PATTERNS FOR CREATING A 3RD SIMULATOR

### 1. **Constants Pattern**
- Single: 3 constants (g, m, l)
- Double: 5 constants (g, m1, m2, l1, l2)
- **Pattern:** Add indexed versions as complexity increases

### 2. **State Dimension Pattern**
- Single: 1D state (scalar q, scalar dq)
- Double: 2D state (array q[2], array dq[2])
- **Pattern:** Use arrays instead of scalars for multi-DOF systems

### 3. **Dynamics Pattern**
- Single: Simple ODE `ddq = -(g/l) * sin(q)`
- Double: Coupled ODEs with coupling coefficients
- **Pattern:** Add coupling terms and solve coupled system

### 4. **Lagrangian Pattern**
- Single: Simple T and V
- Double: T with coupling terms, V with relative positions
- **Pattern:** Add cross-terms in kinetic energy, account for relative positions in potential

### 5. **Gradient Pattern**
- Single: Direct derivatives
- Double: Coupling terms in both gradients
- **Pattern:** Add coupling terms that depend on both q and dq

### 6. **Trajectory Sampling Pattern**
- Single: List initial conditions, direct unpacking
- Double: Array concatenation, slicing and transposing
- **Pattern:** Use `np.concatenate()` for arrays, slice/transpose for multi-DOF

### 7. **Visualization Pattern**
- Single: 1 line per plot, 2-point animation
- Double: 2 lines per plot, 3-point animation
- **Pattern:** Add lines/points proportional to number of DOF

### 8. **Array Shape Pattern**
- Single: (N, 1) for q and dq
- Double: (N, 2) for q and dq
- **Pattern:** Last dimension = number of DOF

---

## CHECKLIST FOR NEW SIMULATOR

When creating a 3rd simulator, ensure:

- [ ] Inherit from `BaseSimulator`
- [ ] Load all required constants in `__init__` using `load_constant()`
- [ ] `init_state()` returns appropriate dimensions (scalar for 1 DOF, array for N DOF)
- [ ] `dynamics()` handles correct state vector size (2*DOF elements)
- [ ] `lagrangian()` handles correct input shapes (N, DOF)
- [ ] `lagrangian_grad_q()` returns correct shape (N, DOF)
- [ ] `lagrangian_grad_dq()` returns correct shape (N, DOF)
- [ ] `sample_trajectory()` properly formats initial conditions and extracts solutions
- [ ] `visualize()` creates appropriate number of plots/lines for the system
- [ ] All array operations use correct shapes and dimensions
- [ ] Animation shows all moving parts correctly

---

## NOTES

1. **BaseSimulator** provides:
   - `save_dir`, `vis_dir`
   - `t_span`, `num_timesteps`
   - `cond_dict`, `constant_counter`
   - `color_palette`
   - `load_constant()` method

2. **Common ODE solver parameters:**
   - `dense_output=True` (allows interpolation)
   - `max_step=0.01` (time step control)

3. **Trajectory dictionary always contains:**
   - `cond_dict`
   - `time`
   - `q`
   - `dq`
   - `L`
   - `L_grad_q`
   - `L_grad_dq`

4. **Visualization always creates:**
   - `q-t_{i_data}.png` (time vs angle)
   - `dq-q_{i_data}.png` (angle vs velocity)
   - `video_{i_data}.gif` (animation)


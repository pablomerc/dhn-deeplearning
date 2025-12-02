import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base_simulator import BaseSimulator


class Simulator(BaseSimulator):
    def __init__(self, config):
        super(Simulator, self).__init__(config)

        constants = config.constants

        self.G = self.load_constant(constants.G, 'G')
        self.m1 = self.load_constant(constants.m1, 'm1')
        self.m2 = self.load_constant(constants.m2, 'm2')
        self.m3 = self.load_constant(constants.m3, 'm3')

        self.r1 = self.load_constant(constants.r1, 'r1')
        self.r2 = self.load_constant(constants.r2, 'r2')
        self.r3 = self.load_constant(constants.r3, 'r3')

        # Extend color palette to 6 colors for three-body visualization
        # (x1, y1, x2, y2, x3, y3 need 6 colors)
        if len(self.color_palette) < 6:
            self.color_palette = self.color_palette + ['#D4A574']  # Add a 6th color

    def init_state(self):
        theta1 = np.random.uniform(0, 2*np.pi)
        theta2 = np.random.uniform(0, 2*np.pi)
        theta3 = np.random.uniform(0, 2*np.pi)

        x1 = self.r1 * np.cos(theta1)
        y1 = self.r1 * np.sin(theta1)

        x2 = self.r2 * np.cos(theta2)
        y2 = self.r2 * np.sin(theta2)

        x3 = self.r3 * np.cos(theta3)
        y3 = self.r3 * np.sin(theta3)

        # Velocity perpendicular to the radius vector
        v_mag = 1.0

        def tangential_velocity(x, y, v_mag):
            r = np.hypot(x, y)
            if r < 1e-9:
                return np.array([0.0, 0.0])
            return v_mag * np.array([-y, x]) / r

        v1 = tangential_velocity(x1, y1, v_mag)
        v2 = tangential_velocity(x2, y2, v_mag)
        v3 = tangential_velocity(x3, y3, v_mag)

        q = np.array([x1, y1, x2, y2, x3, y3])
        dq = np.array([v1[0], v1[1], v2[0], v2[1], v3[0], v3[1]])

        return q, dq

    def relative_force(self, x1, y1, x2, y2, m1, m2, G):
        # Handle both scalar and vectorized inputs
        x1, y1, x2, y2 = np.asarray(x1), np.asarray(y1), np.asarray(x2), np.asarray(y2)

        # Compute relative position vector
        dx = x2 - x1
        dy = y2 - y1

        # Compute distance
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 1e-9)  # Avoid division by zero

        # Force magnitude: G * m1 * m2 / r^2
        # Force vector: F_mag * (r_vec / r) = G * m1 * m2 / r^3 * r_vec
        F_mag = G * m1 * m2 / (r**3)

        # Force vectors
        # For scalar case: F1 shape (2,), for vectorized: F1 shape (N, 2)
        if r.ndim == 0:  # Scalar case
            F1 = np.array([F_mag * dx, F_mag * dy])
            F2 = -F1
        else:  # Vectorized case
            F1 = np.stack([F_mag * dx, F_mag * dy], axis=-1)
            F2 = -F1

        return F1, F2

    def dynamics(self, t, y):
        # State: [x1, y1, x2, y2, x3, y3, v1x, v1y, v2x, v2y, v3x, v3y]
        x1, y1, x2, y2, x3, y3, v1x, v1y, v2x, v2y, v3x, v3y = y
        m1, m2, m3, G = self.m1, self.m2, self.m3, self.G

        # Pairwise forces
        F12, F21 = self.relative_force(x1, y1, x2, y2, m1, m2, G)
        F13, F31 = self.relative_force(x1, y1, x3, y3, m1, m3, G)
        F23, F32 = self.relative_force(x2, y2, x3, y3, m2, m3, G)

        # Sum forces on each body
        F1x, F1y = F12[0] + F13[0], F12[1] + F13[1]
        F2x, F2y = F21[0] + F23[0], F21[1] + F23[1]
        F3x, F3y = F31[0] + F32[0], F31[1] + F32[1]

        # Accelerations
        a1x, a1y = F1x / m1, F1y / m1
        a2x, a2y = F2x / m2, F2y / m2
        a3x, a3y = F3x / m3, F3y / m3

        # dy/dt in the same order as y
        return np.array([
            v1x, v1y,
            v2x, v2y,
            v3x, v3y,
            a1x, a1y,
            a2x, a2y,
            a3x, a3y
        ])

    def lagrangian(self, q, dq):
      """
      q:  (N, 6) -> [x1, y1, x2, y2, x3, y3]
      dq: (N, 6) -> [v1x, v1y, v2x, v2y, v3x, v3y]
      returns: (N,) L values
      """
      x1, y1, x2, y2, x3, y3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4], q[:, 5]
      v1x, v1y, v2x, v2y, v3x, v3y = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3], dq[:, 4], dq[:, 5]

      # Kinetic energy
      T1 = 0.5 * self.m1 * (v1x**2 + v1y**2)
      T2 = 0.5 * self.m2 * (v2x**2 + v2y**2)
      T3 = 0.5 * self.m3 * (v3x**2 + v3y**2)
      T = T1 + T2 + T3

      # Pairwise distances
      dx12, dy12 = x2 - x1, y2 - y1
      dx13, dy13 = x3 - x1, y3 - y1
      dx23, dy23 = x3 - x2, y3 - y2

      eps = 1e-9
      r12 = np.sqrt(dx12**2 + dy12**2)
      r13 = np.sqrt(dx13**2 + dy13**2)
      r23 = np.sqrt(dx23**2 + dy23**2)
      r12 = np.maximum(r12, eps)
      r13 = np.maximum(r13, eps)
      r23 = np.maximum(r23, eps)

      # Potential energy: sum over pairs
      V12 = - self.G * self.m1 * self.m2 / r12
      V13 = - self.G * self.m1 * self.m3 / r13
      V23 = - self.G * self.m2 * self.m3 / r23
      V = V12 + V13 + V23

      return T - V

    def lagrangian_grad_q(self, q, dq):
        # Unpack batch dimension
        x1, y1, x2, y2, x3, y3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4], q[:, 5]
        m1, m2, m3 = self.m1, self.m2, self.m3
        G = self.G

        # Pairwise forces (returns arrays of shape (N, 2) for vectorized inputs)
        F12, F21 = self.relative_force(x1, y1, x2, y2, m1, m2, G)
        F13, F31 = self.relative_force(x1, y1, x3, y3, m1, m3, G)
        F23, F32 = self.relative_force(x2, y2, x3, y3, m2, m3, G)

        # Force sums (dL/dq = total gravitational forces)
        # Note: dL/dq = -dV/dq, and F = -dV/dq, so dL/dq = F
        dL_dx1 = F12[:, 0] + F13[:, 0]
        dL_dy1 = F12[:, 1] + F13[:, 1]

        dL_dx2 = F21[:, 0] + F23[:, 0]
        dL_dy2 = F21[:, 1] + F23[:, 1]

        dL_dx3 = F31[:, 0] + F32[:, 0]
        dL_dy3 = F31[:, 1] + F32[:, 1]

        return np.stack([dL_dx1, dL_dy1, dL_dx2, dL_dy2, dL_dx3, dL_dy3], axis=-1)

    def lagrangian_grad_dq(self, q, dq):
        # Unpack velocities
        v1x, v1y = dq[:, 0], dq[:, 1]
        v2x, v2y = dq[:, 2], dq[:, 3]
        v3x, v3y = dq[:, 4], dq[:, 5]

        m1, m2, m3 = self.m1, self.m2, self.m3

        # dL/d(dq)
        dL_dv1x = m1 * v1x
        dL_dv1y = m1 * v1y

        dL_dv2x = m2 * v2x
        dL_dv2y = m2 * v2y

        dL_dv3x = m3 * v3x
        dL_dv3y = m3 * v3y

        return np.stack([dL_dv1x, dL_dv1y,
                        dL_dv2x, dL_dv2y,
                        dL_dv3x, dL_dv3y], axis=-1)


    def sample_trajectory(self):
        q0, dq0 = self.init_state()

        # Initial state vector: [x1,y1,x2,y2,x3,y3,v1x,v1y,v2x,v2y,v3x,v3y]
        y0 = np.concatenate([q0, dq0], axis=0)

        # Integrate
        sol = solve_ivp(self.dynamics, self.t_span, y0, dense_output=True, max_step=0.01)

        # Extract time points
        t_vals = np.linspace(self.t_span[0], self.t_span[1], self.num_timesteps)
        y_vals = sol.sol(t_vals)  # shape = (12, T)

        # Split into positions and velocities
        q_vals = y_vals[:6].transpose((1, 0))   # shape: (T, 6)
        dq_vals = y_vals[6:].transpose((1, 0))  # shape: (T, 6)

        # Compute Lagrangian and gradients
        L_vals = self.lagrangian(q_vals, dq_vals)
        L_grad_q = self.lagrangian_grad_q(q_vals, dq_vals)
        L_grad_dq = self.lagrangian_grad_dq(q_vals, dq_vals)

        return {
            'cond_dict': self.cond_dict,
            'time': t_vals,
            'q': q_vals,
            'dq': dq_vals,
            'L': L_vals,
            'L_grad_q': L_grad_q,
            'L_grad_dq': L_grad_dq,
        }

    def visualize(self, traj, i_data):
        os.makedirs(self.vis_dir, exist_ok=True)

        # Unpack trajectory
        t = traj['time']
        q = traj['q']   # (T, 6): [x1, y1, x2, y2, x3, y3]
        dq = traj['dq'] # (T, 6): [v1x, v1y, v2x, v2y, v3x, v3y]

        x1, y1 = q[:, 0], q[:, 1]
        x2, y2 = q[:, 2], q[:, 3]
        x3, y3 = q[:, 4], q[:, 5]

        v1x, v1y = dq[:, 0], dq[:, 1]
        v2x, v2y = dq[:, 2], dq[:, 3]
        v3x, v3y = dq[:, 4], dq[:, 5]

        ##################################################
        # Positions vs time
        ##################################################
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(t, x1, label='x1', color=self.color_palette[0])
        plt.plot(t, y1, label='y1', color=self.color_palette[1])
        plt.plot(t, x2, label='x2', color=self.color_palette[2])
        plt.plot(t, y2, label='y2', color=self.color_palette[3])
        plt.plot(t, x3, label='x3', color=self.color_palette[4])
        plt.plot(t, y3, label='y3', color=self.color_palette[5])
        plt.title(f'Positions vs Time, m1={self.m1:.2f}, m2={self.m2:.2f}, m3={self.m3:.2f}, G={self.G:.2f}')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [arb. units]')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'q-t_{i_data}.png')
        plt.savefig(save_path)
        plt.close()

        ##################################################
        # Velocities vs time
        ##################################################
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(t, v1x, label='v1x', color=self.color_palette[0])
        plt.plot(t, v1y, label='v1y', color=self.color_palette[1])
        plt.plot(t, v2x, label='v2x', color=self.color_palette[2])
        plt.plot(t, v2y, label='v2y', color=self.color_palette[3])
        plt.plot(t, v3x, label='v3x', color=self.color_palette[4])
        plt.plot(t, v3y, label='v3y', color=self.color_palette[5])
        plt.title(f'Velocities vs Time, m1={self.m1:.2f}, m2={self.m2:.2f}, m3={self.m3:.2f}, G={self.G:.2f}')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [arb. units]')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'dq-t_{i_data}.png')
        plt.savefig(save_path)
        plt.close()

        ##################################################
        # Orbit trajectories in x–y plane
        ##################################################
        plt.figure(figsize=(10, 10), dpi=100)
        plt.plot(x1, y1, '-', label='Body 1', color=self.color_palette[0], linewidth=1.8)
        plt.plot(x2, y2, '-', label='Body 2', color=self.color_palette[2], linewidth=1.8)
        plt.plot(x3, y3, '-', label='Body 3', color=self.color_palette[4], linewidth=1.8)

        # Start markers
        plt.scatter([x1[0]], [y1[0]], s=50, color=self.color_palette[0], marker='o', zorder=5)
        plt.scatter([x2[0]], [y2[0]], s=50, color=self.color_palette[2], marker='o', zorder=5)
        plt.scatter([x3[0]], [y3[0]], s=50, color=self.color_palette[4], marker='o', zorder=5)

        # End markers
        plt.scatter([x1[-1]], [y1[-1]], s=50, color=self.color_palette[0], marker='s', zorder=5)
        plt.scatter([x2[-1]], [y2[-1]], s=50, color=self.color_palette[2], marker='s', zorder=5)
        plt.scatter([x3[-1]], [y3[-1]], s=50, color=self.color_palette[4], marker='s', zorder=5)

        # Optional: center of mass path
        total_mass = self.m1 + self.m2 + self.m3
        x_com = (self.m1 * x1 + self.m2 * x2 + self.m3 * x3) / total_mass
        y_com = (self.m1 * y1 + self.m2 * y2 + self.m3 * y3) / total_mass
        plt.plot(x_com, y_com, '--', color='gray', alpha=0.7, label='Center of Mass')
        plt.scatter([x_com[0]], [y_com[0]], s=60, color='gray', marker='+', zorder=6)

        plt.title(f'3-Body Orbit Trajectories, m1={self.m1:.2f}, m2={self.m2:.2f}, m3={self.m3:.2f}')
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'orbit_{i_data}.png')
        plt.savefig(save_path)
        plt.close()

        ##################################################
        # Animated orbit
        ##################################################
        t_per_frame = 5

        all_x = np.concatenate([x1, x2, x3])
        all_y = np.concatenate([y1, y2, y3])
        x_range = np.max(all_x) - np.min(all_x)
        y_range = np.max(all_y) - np.min(all_y)
        max_range = max(x_range, y_range) * 1.2 if max(x_range, y_range) > 0 else 1.0
        x_center = 0.5 * (np.max(all_x) + np.min(all_x))
        y_center = 0.5 * (np.max(all_y) + np.min(all_y))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_aspect('equal')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'3-Body Orbit Animation, m1={self.m1:.2f}, m2={self.m2:.2f}, m3={self.m3:.2f}, G={self.G:.2f}')

        # Center of mass marker
        com_marker, = ax.plot([], [], '+', color='gray', markersize=10, markeredgewidth=2, label='CoM')

        # Bodies
        body1, = ax.plot([], [], 'o', color=self.color_palette[0], markersize=8, label='Body 1')
        body2, = ax.plot([], [], 'o', color=self.color_palette[2], markersize=8, label='Body 2')
        body3, = ax.plot([], [], 'o', color=self.color_palette[4], markersize=8, label='Body 3')

        # Traces
        trace1, = ax.plot([], [], '-', color=self.color_palette[0], alpha=0.5, linewidth=1.5)
        trace2, = ax.plot([], [], '-', color=self.color_palette[2], alpha=0.5, linewidth=1.5)
        trace3, = ax.plot([], [], '-', color=self.color_palette[4], alpha=0.5, linewidth=1.5)

        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend(loc='upper right')

        trace1_x, trace1_y = [], []
        trace2_x, trace2_y = [], []
        trace3_x, trace3_y = [], []

        def update(frame):
            idx = frame * t_per_frame
            if idx >= len(t):
                idx = len(t) - 1

            # Positions at this time
            x1_t, y1_t = x1[idx], y1[idx]
            x2_t, y2_t = x2[idx], y2[idx]
            x3_t, y3_t = x3[idx], y3[idx]

            x_com_t = (self.m1 * x1_t + self.m2 * x2_t + self.m3 * x3_t) / total_mass
            y_com_t = (self.m1 * y1_t + self.m2 * y2_t + self.m3 * y3_t) / total_mass

            # Update bodies
            body1.set_data([x1_t], [y1_t])
            body2.set_data([x2_t], [y2_t])
            body3.set_data([x3_t], [y3_t])

            # Update CoM
            com_marker.set_data([x_com_t], [y_com_t])

            # Update traces
            trace1_x.append(x1_t); trace1_y.append(y1_t)
            trace2_x.append(x2_t); trace2_y.append(y2_t)
            trace3_x.append(x3_t); trace3_y.append(y3_t)
            trace1.set_data(trace1_x, trace1_y)
            trace2.set_data(trace2_x, trace2_y)
            trace3.set_data(trace3_x, trace3_y)

            # Update time text
            time_text.set_text(f'Time: {t[idx]:.2f} s')

            return body1, body2, body3, com_marker, trace1, trace2, trace3, time_text

        num_frames = min(self.num_timesteps // t_per_frame, len(t) // t_per_frame)
        ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=50)
        save_path = os.path.join(self.vis_dir, f'video_{i_data}.gif')
        ani.save(save_path, writer='pillow', fps=20)
        plt.close()

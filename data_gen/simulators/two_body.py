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

    self.G = self.load_constant(constants.G, 'G') # Gravitational constant
    self.m1 = self.load_constant(constants.m1, 'm1') # Mass of body 1
    self.m2 = self.load_constant(constants.m2, 'm2') # Mass of body 2

    self.r1 = self.load_constant(constants.r1, 'r1') # orbital radius of body 1
    self.r2 = self.load_constant(constants.r2, 'r2') # orbital radius of body 2

    self.mu = self.m1*self.m2/(self.m1 + self.m2)

  def init_state(self):
    r = self.r1 + self.r2 # Assume they are on opposite sides of the CoM??
    theta = np.random.uniform(0, 2*np.pi)

    dr = 0

    # Circular angular velocity at radius r
    dtheta_circ = np.sqrt(self.G * (self.m1 + self.m2) / (r ** 3))

    # Scale factor for ellipticity:
    #   = 1.0  -> circular
    #   < 1.0  -> tighter bound ellipse
    #   between 1.0 and sqrt(2) -> ellipse with higher energy
    v_factor = 0.7   # <-- pick something != 1 for ellipse

    dtheta = v_factor * dtheta_circ


    q = np.array([r,theta])
    dq = np.array([dr,dtheta])

    return q, dq

  def dynamics(self, t, y):
    r, theta, dr, dtheta = y

    M = self.m1 + self.m2

    ddr = r * dtheta**2 - self.G * M / r**2
    ddtheta = -2 * dr * dtheta / r

    return np.array([dr, dtheta, ddr, ddtheta])

  def lagrangian(self, q, dq):
    r = q[:, 0]
    dr = dq[:, 0]
    dtheta = dq[:, 1]

    T = 0.5 * self.mu * (dr**2 + r**2 * dtheta**2)
    V = - self.G * self.m1 * self.m2 / r

    return T - V

  def lagrangian_grad_q(self, q, dq):
    r, theta = q[:, 0], q[:, 1]
    dr, dtheta = dq[:, 0], dq[:, 1]

    grad_r = self.mu * r * (dtheta**2) - self.G * self.m1 * self.m2 / (r**2)

    grad_theta = np.zeros_like(r)

    return np.stack([grad_r,grad_theta],axis=-1)


  def lagrangian_grad_dq(self,q,dq):
    r, theta = q[:, 0], q[:, 1]
    dr, dtheta = dq[:, 0], dq[:, 1]

    grad_dr = self.mu * dr
    grad_dtheta = self.mu * (r**2) * dtheta

    return np.stack([grad_dr,grad_dtheta], axis = -1)



  def sample_trajectory(self):
    q0,dq0 = self.init_state()

    # Initial conditions
    y0 = np.concatenate([q0, dq0], axis = 0)

    # Solve the differential equation for this length

    sol = solve_ivp(self.dynamics, self.t_span, y0, dense_output=True, max_step=0.01)

    # Get the t,q,dq points
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

  def visualize(self, traj, i_data):
    os.makedirs(self.vis_dir, exist_ok=True)

    r = traj['q'][:, 0]
    theta = traj['q'][:, 1]
    dr = traj['dq'][:, 0]
    dtheta = traj['dq'][:, 1]
    M = self.m1 + self.m2

    ##################################################
    # Plot q-t curve (r and theta vs time)
    ##################################################
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(traj['time'], r, label='r (radial distance)', color=self.color_palette[1])
    plt.plot(traj['time'], theta, label='θ (angle)', color=self.color_palette[4])
    plt.title(f'Positions vs Time, m1={self.m1:.2f}, m2={self.m2:.2f}, G={self.G:.2f}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [r: distance, θ: rad]')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(self.vis_dir, f'q-t_{i_data}.png')
    plt.savefig(save_path)
    plt.close()

    ##################################################
    # Plot dq-t curve (velocities vs time)
    ##################################################
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(traj['time'], dr, label='dr/dt (radial velocity)', color=self.color_palette[1])
    plt.plot(traj['time'], dtheta, label='dθ/dt (angular velocity)', color=self.color_palette[4])
    plt.title(f'Velocities vs Time, m1={self.m1:.2f}, m2={self.m2:.2f}, G={self.G:.2f}')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [dr/dt: distance/s, dθ/dt: rad/s]')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(self.vis_dir, f'dq-t_{i_data}.png')
    plt.savefig(save_path)
    plt.close()

    ##################################################
    # Plot phase space: r-dr and theta-dtheta
    ##################################################
    plt.figure(figsize=(10, 6))
    plt.plot(r, dr, label='r vs dr/dt', color=self.color_palette[1], alpha=0.7)
    plt.plot(theta, dtheta, label='θ vs dθ/dt', color=self.color_palette[4], alpha=0.7)
    plt.title(f'Phase Space, m1={self.m1:.2f}, m2={self.m2:.2f}, G={self.G:.2f}')
    plt.xlabel('Position [r: distance, θ: rad]')
    plt.ylabel('Velocity [dr/dt: distance/s, dθ/dt: rad/s]')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(self.vis_dir, f'dq-q_{i_data}.png')
    plt.savefig(save_path)
    plt.close()

    ##################################################
    # Plot orbit trajectory (2D orbit path)
    ##################################################
    # Convert to Cartesian coordinates for the reduced mass particle
    x_reduced = r * np.cos(theta)
    y_reduced = r * np.sin(theta)

    # Calculate positions of the two bodies relative to center of mass
    # Body 1: -m2/(m1+m2) * r_vec, Body 2: m1/(m1+m2) * r_vec
    x1 = -self.m2 / M * x_reduced
    y1 = -self.m2 / M * y_reduced
    x2 = self.m1 / M * x_reduced
    y2 = self.m1 / M * y_reduced

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(x1, y1, '-', label='Body 1 orbit', color=self.color_palette[1], alpha=0.6, linewidth=1.5)
    plt.plot(x2, y2, '-', label='Body 2 orbit', color=self.color_palette[4], alpha=0.6, linewidth=1.5)
    plt.plot(0, 0, 'k+', markersize=12, label='Center of Mass', markeredgewidth=2)
    plt.plot(x1[0], y1[0], 'o', color=self.color_palette[1], markersize=8, label='Body 1 start')
    plt.plot(x2[0], y2[0], 'o', color=self.color_palette[4], markersize=8, label='Body 2 start')
    plt.plot(x1[-1], y1[-1], 's', color=self.color_palette[1], markersize=8, label='Body 1 end')
    plt.plot(x2[-1], y2[-1], 's', color=self.color_palette[4], markersize=8, label='Body 2 end')
    plt.title(f'Orbit Trajectory, m1={self.m1:.2f}, m2={self.m2:.2f}, G={self.G:.2f}')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(self.vis_dir, f'orbit_{i_data}.png')
    plt.savefig(save_path)
    plt.close()

    ##################################################
    # Plot animated orbit video
    ##################################################
    t_per_frame = 5

    # Determine plot limits based on maximum orbit extent
    r_max = np.max(r)
    plot_margin = r_max * 0.2
    xlim = [-r_max - plot_margin, r_max + plot_margin]
    ylim = [-r_max - plot_margin, r_max + plot_margin]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'Two-Body Orbit Animation, m1={self.m1:.2f}, m2={self.m2:.2f}, G={self.G:.2f}')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.grid(True, alpha=0.3)

    # Center of mass marker
    com_marker, = ax.plot([0], [0], 'k+', markersize=12, markeredgewidth=2, label='CoM')

    # Body 1 and 2 markers
    body1, = ax.plot([], [], 'o', color=self.color_palette[1], markersize=10, label='Body 1')
    body2, = ax.plot([], [], 'o', color=self.color_palette[4], markersize=10, label='Body 2')

    # Orbit traces
    trace1, = ax.plot([], [], '-', color=self.color_palette[1], alpha=0.4, linewidth=1.5)
    trace2, = ax.plot([], [], '-', color=self.color_palette[4], alpha=0.4, linewidth=1.5)

    # Connection line between bodies
    connection, = ax.plot([], [], '--', color='gray', alpha=0.5, linewidth=1)

    # Time display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper right')

    # Trace history
    trace1_x, trace1_y = [], []
    trace2_x, trace2_y = [], []

    def update(frame):
      t = frame * t_per_frame
      if t >= len(r):
        t = len(r) - 1

      # Current positions
      x1_t = x1[t]
      y1_t = y1[t]
      x2_t = x2[t]
      y2_t = y2[t]

      # Update body positions
      body1.set_data([x1_t], [y1_t])
      body2.set_data([x2_t], [y2_t])

      # Update connection line
      connection.set_data([x1_t, x2_t], [y1_t, y2_t])

      # Update traces
      trace1_x.append(x1_t)
      trace1_y.append(y1_t)
      trace2_x.append(x2_t)
      trace2_y.append(y2_t)
      trace1.set_data(trace1_x, trace1_y)
      trace2.set_data(trace2_x, trace2_y)

      # Update time display
      time_text.set_text(f'Time: {traj["time"][t]:.2f} s')

      return body1, body2, trace1, trace2, connection, time_text

    # Create the animation
    num_frames = min(self.num_timesteps // t_per_frame, len(r) // t_per_frame)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=50)
    save_path = os.path.join(self.vis_dir, f'video_{i_data}.gif')
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()

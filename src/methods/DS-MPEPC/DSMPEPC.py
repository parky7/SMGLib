"""
DS-MPEPC implementation focused on the 4-robot position-swap (circle) scenario
from Figure 7 of:

  S. H. Arul, J. J. Park, and D. Manocha,
  "DS-MPEPC: Safe and Deadlock-Avoiding Robot Navigation in Cluttered Dynamic
  Scenes", arXiv:2303.10133, 2023.

Background (MPEPC) follows:

  J. J. Park, C. Johnson, and B. Kuipers,
  "Robot Navigation with Model Predictive Equilibrium Point Control",
  IROS 2012.

What is implemented
-------------------
- Smooth pose-following control law (Park & Kuipers, ICRA 2011), used to
  parameterize the space of closed-loop trajectories by z* = (r, theta, delta, vmax).
- Constant-velocity predictions of the other agents over the planning horizon.
- DS-MPEPC trajectory cost:
      J = sum_k [ ps_k * J_progress_k + J_action_k + (1 - ps_k) * J_collision_k ]
        + J_terminal(x_N)
  with
      p_c = exp(-d_o^2 / sigma_d^2) * (1 - a * exp(-(1/TTC)^2 / sigma_{1/TTC}^2))
      ps_k = prod_{i<=k} (1 - p_c_i)
      C_TTG = exp(-(1/TTG)^2 / sigma_{1/TTG}^2)
      C_TTC = exp(-(1/TTC)^2 / sigma_{1/TTC}^2)
      J_terminal = - p_s_N * C_TTG * C_TTC
  Note: the paper's Eq. (5) prints the anticipatory exponent without a negative
  sign, but Lemma 4.2 and the intended limits fix the sign as above.
- Sampling-based minimization of J over z* at every planning cycle, seeded with
  the previous optimum, a stopping sample, and a set of goal-directed samples.
- Decentralized multi-agent loop: each agent plans independently.
"""

import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.patches as patches
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / 'src'))
from utils import StandardizedEnvironment


# ======================================================================
# Parameters (from the DS-MPEPC paper, Section V-A)
# ======================================================================
# Smooth control law (Park & Kuipers)
K1 = 1.5
K2 = 3.0
BETA = 0.4
LAMBDA = 2.0
R_THRESH = 1.2
VMAX = 1.0

# DS-MPEPC cost
A_ANTICIP = 0.7                # paper value
SIGMA_INV_TTC = 0.5            # paper value
SIGMA_INV_TTG = 1e-3           # paper value
SIGMA_D_STATIC = 0.1           # MPEPC default
SIGMA_D_DYNAMIC = 0.2          # MPEPC default

# Cost weights
W_PROGRESS = 1.0
W_ACTION_V = 0.15
W_ACTION_W = 0.05
W_COLLISION = 0.4
W_TERMINAL = 5.0

# Planning / simulation
T_HORIZON = 5.0
DT_PLAN = 0.2
N_HORIZON = int(round(T_HORIZON / DT_PLAN))   # 25
DT_SIM = 0.1

# Sampling
N_RANDOM_SAMPLES = 180


# ======================================================================
# Utilities
# ======================================================================
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def pose_to_ego(robot_pose, target_pose):
    """Return (r, theta, delta) for robot -> target in the egocentric frame."""
    dx = target_pose[0] - robot_pose[0]
    dy = target_pose[1] - robot_pose[1]
    r = float(np.hypot(dx, dy))
    los = float(np.arctan2(dy, dx))
    theta = wrap_pi(target_pose[2] - los)
    delta = wrap_pi(robot_pose[2] - los)
    return r, theta, delta


def ego_to_target(robot_pose, r, theta, delta):
    """Inverse of pose_to_ego: reconstruct the absolute target pose."""
    los = wrap_pi(robot_pose[2] - delta)
    tx = robot_pose[0] + r * np.cos(los)
    ty = robot_pose[1] + r * np.sin(los)
    tpsi = wrap_pi(los + theta)
    return np.array([tx, ty, tpsi])


def control_cmd(robot_pose, target_pose, vmax):
    """Smooth pose-following control law, Eq. (15)-(18) of Park & Kuipers 2012."""
    r, theta, delta = pose_to_ego(robot_pose, target_pose)
    if r < 1e-4:
        return 0.0, 0.0
    kappa = -(1.0 / r) * (
        K2 * (delta - np.arctan(-K1 * theta))
        + (1.0 + K1 / (1.0 + (K1 * theta) ** 2)) * np.sin(delta)
    )
    v_kappa = vmax / (1.0 + BETA * abs(kappa) ** LAMBDA)
    v = min((vmax / R_THRESH) * r, v_kappa)
    omega = kappa * v
    return float(v), float(omega)


def simulate_trajectory(robot_pose, target_pose, vmax, dt=DT_PLAN, n_steps=N_HORIZON):
    """Forward-simulate the closed-loop trajectory parameterized by z*."""
    poses = np.zeros((n_steps + 1, 3))
    vels = np.zeros((n_steps, 2))
    poses[0] = robot_pose
    cur = robot_pose.copy()
    for k in range(n_steps):
        v, w = control_cmd(cur, target_pose, vmax)
        cur = cur.copy()
        cur[0] += v * np.cos(cur[2]) * dt
        cur[1] += v * np.sin(cur[2]) * dt
        cur[2] = wrap_pi(cur[2] + w * dt)
        poses[k + 1] = cur
        vels[k] = (v, w)
    return poses, vels


# ======================================================================
# Time-to-collision between two moving disks
# ======================================================================
def ttc_disks(p1, v1, p2, v2, r_sum):
    """Smallest non-negative t such that ||(p1+t v1) - (p2+t v2)|| = r_sum.
    Returns np.inf if they never come within r_sum."""
    dp = p1 - p2
    dv = v1 - v2
    a = float(dv @ dv)
    b = 2.0 * float(dp @ dv)
    c = float(dp @ dp - r_sum * r_sum)
    if c <= 0.0:
        return 0.0
    if a < 1e-12:
        return np.inf
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return np.inf
    sq = float(np.sqrt(disc))
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    if t1 > 0.0:
        return t1
    if t2 > 0.0:
        return 0.0
    return np.inf


# ======================================================================
# DS-MPEPC collision probability (Eq. 5, with sign consistent with Lemma 4.2)
# ======================================================================
def pc_eq5(d_eff, ttc, sigma_d):
    """Modified collision probability for a trajectory sample."""
    if d_eff <= 0.0:
        return 1.0
    reactive = np.exp(-(d_eff ** 2) / (sigma_d ** 2))
    if np.isinf(ttc):
        anticip = 1.0 - A_ANTICIP                  # 1/TTC -> 0
    elif ttc <= 0.0:
        anticip = 1.0                              # already colliding
    else:
        anticip = 1.0 - A_ANTICIP * np.exp(-((1.0 / ttc) ** 2) / (SIGMA_INV_TTC ** 2))
    return float(reactive * anticip)


# ======================================================================
# Trajectory cost
# ======================================================================
def trajectory_cost(poses, vels, goal_xy, agent_radius,
                    static_obs_pos, dyn_obs_pred, dyn_obs_vel,
                    vmax, dt=DT_PLAN):
    """Compute J_tilde(q_{z*}) for a simulated trajectory."""
    N = len(vels)
    r_sum_dyn = 2.0 * agent_radius
    r_sum_stat = agent_radius                      # static obstacles treated as points

    # --- per-step collision probability (closest obstacle at each step) ---
    pc_arr = np.zeros(N + 1)
    for k in range(N + 1):
        p_r = poses[k, :2]
        # robot instantaneous velocity along its heading
        if k < N:
            v_r = vels[k, 0] * np.array([np.cos(poses[k, 2]), np.sin(poses[k, 2])])
        else:
            v_r = vels[-1, 0] * np.array([np.cos(poses[-1, 2]), np.sin(poses[-1, 2])])

        best = (np.inf, np.inf, SIGMA_D_STATIC)   # (pc, d_eff, sigma)
        # static
        for obs in static_obs_pos:
            d = float(np.linalg.norm(p_r - obs)) - r_sum_stat
            t = ttc_disks(p_r, v_r, obs, np.zeros(2), r_sum_stat)
            p = pc_eq5(d, t, SIGMA_D_STATIC)
            if p > best[0] or best[0] == np.inf:
                best = (p, d, SIGMA_D_STATIC)
        # dynamic (other agents, const-velocity prediction)
        for j in range(dyn_obs_pred.shape[1]):
            p_o = dyn_obs_pred[k, j]
            v_o = dyn_obs_vel[j]
            d = float(np.linalg.norm(p_r - p_o)) - r_sum_dyn
            t = ttc_disks(p_r, v_r, p_o, v_o, r_sum_dyn)
            p = pc_eq5(d, t, SIGMA_D_DYNAMIC)
            if p > best[0] or best[0] == np.inf:
                best = (p, d, SIGMA_D_DYNAMIC)
        pc_arr[k] = best[0] if np.isfinite(best[0]) else 0.0

    # --- survivability (cumulative product) ---
    ps_arr = np.zeros(N + 1)
    running = 1.0
    for k in range(N + 1):
        running *= (1.0 - pc_arr[k])
        ps_arr[k] = running

    # --- progress toward goal, weighted by ps ---
    J_progress = 0.0
    for k in range(1, N + 1):
        d_k = float(np.linalg.norm(poses[k, :2] - goal_xy))
        d_km1 = float(np.linalg.norm(poses[k - 1, :2] - goal_xy))
        J_progress += ps_arr[k] * W_PROGRESS * (d_k - d_km1)

    # --- action cost ---
    J_action = 0.0
    for k in range(N):
        J_action += (W_ACTION_V * vels[k, 0] ** 2 + W_ACTION_W * vels[k, 1] ** 2) * dt

    # --- collision cost ---
    phi_col = 1.0
    J_collision = 0.0
    for k in range(1, N + 1):
        J_collision += (1.0 - ps_arr[k]) * W_COLLISION * phi_col

    # --- terminal cost (Eq. 6) ---
    term_pose = poses[-1]
    term_v = vels[-1, 0]
    goal_vec = goal_xy - term_pose[:2]
    d_goal = float(np.linalg.norm(goal_vec))
    heading = np.array([np.cos(term_pose[2]), np.sin(term_pose[2])])

    if d_goal > 1e-6:
        v_toward = term_v * float(heading @ goal_vec) / d_goal
    else:
        v_toward = 0.0
    if v_toward > 1e-3:
        ttg = d_goal / v_toward
    else:
        ttg = np.inf

    v_term_vec = vmax * heading
    ttc_term = np.inf
    for obs in static_obs_pos:
        t = ttc_disks(term_pose[:2], v_term_vec, obs, np.zeros(2), r_sum_stat)
        if t < ttc_term:
            ttc_term = t
    for j in range(dyn_obs_pred.shape[1]):
        t = ttc_disks(term_pose[:2], v_term_vec,
                      dyn_obs_pred[-1, j], dyn_obs_vel[j], r_sum_dyn)
        if t < ttc_term:
            ttc_term = t

    inv_ttg_sq = 0.0 if np.isinf(ttg) else (1.0 / max(ttg, 1e-6)) ** 2
    inv_ttc_sq = 0.0 if np.isinf(ttc_term) else (1.0 / max(ttc_term, 1e-3)) ** 2
    C_TTG = float(np.exp(-inv_ttg_sq / (SIGMA_INV_TTG ** 2)))
    C_TTC = float(np.exp(-inv_ttc_sq / (SIGMA_INV_TTC ** 2)))
    J_terminal = -W_TERMINAL * ps_arr[-1] * C_TTG * C_TTC

    return J_progress + J_action + J_collision + J_terminal


# ======================================================================
# Planner: sampling-based minimization over z*
# ======================================================================
def plan_one_step(robot_pose, goal_xy, agent_radius,
                  static_obs_pos, dyn_obs_pred, dyn_obs_vel,
                  prev_z=None, rng=None):
    """Choose z* = (r, theta, delta, vmax) minimizing J_tilde."""
    if rng is None:
        rng = np.random.default_rng()

    goal_vec = goal_xy - robot_pose[:2]
    d_goal = float(np.linalg.norm(goal_vec))
    los_goal = float(np.arctan2(goal_vec[1], goal_vec[0]))
    delta_goal = wrap_pi(robot_pose[2] - los_goal)

    r_goal = min(max(d_goal, 0.5), 6.0)

    candidates = []
    # goal-directed: aim at goal pose with heading aligned to goal direction
    candidates.append((r_goal, -delta_goal, delta_goal, VMAX))
    # slower goal-directed
    candidates.append((r_goal, -delta_goal, delta_goal, 0.6 * VMAX))
    # stopping (null motion)
    candidates.append((0.05, 0.0, 0.0, 0.0))
    # typical soft/hard turns around the goal line
    for dturn in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
        candidates.append((r_goal, -delta_goal + dturn, delta_goal + dturn, VMAX))
    # seed with previous optimum
    if prev_z is not None:
        candidates.append(tuple(prev_z))

    # random coverage
    for _ in range(N_RANDOM_SAMPLES):
        r = rng.uniform(0.4, 6.0)
        theta = rng.uniform(-1.2, 1.2)
        delta = rng.uniform(-1.6, 1.6)
        vm = rng.uniform(0.1, VMAX)
        candidates.append((r, theta, delta, vm))

    best = None
    best_cost = np.inf
    best_poses = None
    best_vels = None

    for z in candidates:
        r, theta, delta, vmax = z
        if r < 1e-3:
            target = robot_pose.copy()
        else:
            target = ego_to_target(robot_pose, r, theta, delta)
        poses, vels = simulate_trajectory(robot_pose, target, vmax)
        cost = trajectory_cost(poses, vels, goal_xy, agent_radius,
                               static_obs_pos, dyn_obs_pred, dyn_obs_vel, vmax)
        if cost < best_cost:
            best_cost = cost
            best = z
            best_poses = poses
            best_vels = vels

    return best, best_cost, best_poses, best_vels


# ======================================================================
# Output helpers
# ======================================================================
def _logs_dirs():
    root_dir = Path(__file__).resolve().parents[3]
    anim_dir = root_dir / 'logs' / 'DS-MPEPC' / 'animations'
    traj_dir = root_dir / 'logs' / 'DS-MPEPC' / 'trajectories'
    anim_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    return anim_dir, traj_dir


def save_gif(X, G, N, env_type, agent_radius):
    anim_dir, _ = _logs_dirs()
    filename = anim_dir / f"{env_type}_{N}agents.gif"

    Kp1 = X.shape[2]
    colors = StandardizedEnvironment.AGENT_COLORS

    fig, ax = plt.subplots(figsize=StandardizedEnvironment.FIG_SIZE)
    ax.set_xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
    ax.set_ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"DS-MPEPC: {N}-robot position swap")

    # Goals
    for i in range(N):
        ax.scatter(G[i, 0], G[i, 1], marker='*', s=300,
                   color='green', edgecolor='black', zorder=4)

    # Full path traces (faint)
    for i in range(N):
        color = colors[i % len(colors)]
        ax.plot(X[i, 0, :], X[i, 1, :], color=color, alpha=0.25, linewidth=1.2)

    # Animated disks
    circles = []
    for i in range(N):
        color = colors[i % len(colors)]
        circ = patches.Circle((X[i, 0, 0], X[i, 1, 0]), radius=agent_radius,
                              facecolor=color, edgecolor='black', linewidth=1, zorder=5)
        ax.add_patch(circ)
        circles.append(circ)

    legend_handles = []
    legend_labels = []
    for i in range(N):
        color = colors[i % len(colors)]
        legend_handles.append(mlines.Line2D([], [], color=color, marker='o',
                                            linestyle='None', markersize=10,
                                            markerfacecolor=color,
                                            markeredgecolor='black'))
        legend_labels.append(f'Agent {i+1}')
    legend_handles.append(mlines.Line2D([], [], color='green', marker='*',
                                        linestyle='None', markersize=12,
                                        markerfacecolor='green',
                                        markeredgecolor='none'))
    legend_labels.append('Goal')
    ax.legend(legend_handles, legend_labels,
              loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=12, borderaxespad=0., markerscale=1.2)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    frame_step = max(1, Kp1 // 200)
    frame_indices = list(range(0, Kp1, frame_step))

    def animate(idx):
        k = frame_indices[idx]
        for i, circ in enumerate(circles):
            circ.center = (X[i, 0, k], X[i, 1, k])
        return circles

    anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                         interval=StandardizedEnvironment.ANIMATION_INTERVAL,
                         blit=True)
    anim.save(str(filename), writer='pillow',
              fps=StandardizedEnvironment.ANIMATION_FPS)
    print(f"GIF animation saved as {filename}")
    plt.close(fig)


def save_csvs(X, G, Uhist, N, K, goal_threshold=0.3):
    out_dir = Path(__file__).resolve().parent

    ttg_list = []
    all_reached = True
    for i in range(N):
        reached = False
        agent_ttg = K
        for k in range(K + 1):
            if np.linalg.norm(X[i, :, k] - G[i]) < goal_threshold:
                agent_ttg = k
                reached = True
                break
        if not reached:
            all_reached = False
        ttg_list.append((i, agent_ttg, reached))

    completion_step = max(t for _, t, _ in ttg_list) if all_reached else K

    with open(out_dir / "completion_step.txt", "w") as f:
        f.write(str(completion_step))

    with open(out_dir / "ttg_impc_dr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["robot_id", "ttg", "reached_goal"])
        for rid, ttg, reached in ttg_list:
            w.writerow([rid, ttg, reached])

    for i in range(N):
        num_steps = K + 1
        nominal_x = np.linspace(X[i, 0, 0], G[i, 0], num_steps)
        nominal_y = np.linspace(X[i, 1, 0], G[i, 1], num_steps)
        with open(out_dir / f"path_deviation_robot_{i}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["actual_x", "actual_y", "nominal_x", "nominal_y"])
            for k in range(num_steps):
                w.writerow([X[i, 0, k], X[i, 1, k], nominal_x[k], nominal_y[k]])

        with open(out_dir / f"avg_delta_velocity_robot_{i}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vx", "vy"])
            for k in range(K):
                w.writerow([Uhist[i, 0, k], Uhist[i, 1, k]])

    print(f"Trajectory CSVs saved to {out_dir}")


# ======================================================================
# 4-robot circle position-swap scenario (Figure 7 in DS-MPEPC)
# ======================================================================
def run_circle_swap(N=4, radius=3.0, T_sim=25.0, seed=0, verbose=True):
    rng = np.random.default_rng(seed)

    # Cardinal positions -> diagonally opposite
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    starts = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    goals = -starts
    headings = np.arctan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
    # tiny asymmetry to break perfectly symmetric ties
    headings = headings + rng.uniform(-0.02, 0.02, size=N)

    agent_radius = StandardizedEnvironment.DEFAULT_AGENT_RADIUS

    K_steps = int(round(T_sim / DT_SIM))
    X = np.zeros((N, 2, K_steps + 1))
    Theta = np.zeros((N, K_steps + 1))
    Uhist = np.zeros((N, 2, K_steps))

    X[:, :, 0] = starts
    Theta[:, 0] = headings
    prev_z = [None] * N

    # cadence: replan every plan_every sim steps (DT_PLAN / DT_SIM)
    plan_every = max(1, int(round(DT_PLAN / DT_SIM)))
    current_target = [None] * N
    current_vmax = [0.0] * N

    static_obs_pos = []  # Figure 7 scenario has no static obstacles

    print(f"Running DS-MPEPC 4-robot circle swap ({N} agents, radius={radius})...")

    for k in range(K_steps):
        # Current states
        poses = np.column_stack([X[:, 0, k], X[:, 1, k], Theta[:, k]])
        # Approx each agent's instantaneous (vx, vy) from last command
        if k == 0:
            vels_xy = np.zeros((N, 2))
        else:
            v_last = Uhist[:, 0, k - 1]
            psi = Theta[:, k]
            vels_xy = np.column_stack([v_last * np.cos(psi),
                                       v_last * np.sin(psi)])

        # (Re)plan for each agent that is due
        for i in range(N):
            if k % plan_every != 0 and current_target[i] is not None:
                continue

            # Predictions of other agents over horizon (const velocity)
            others_idx = [j for j in range(N) if j != i]
            M = len(others_idx)
            dyn_pred = np.zeros((N_HORIZON + 1, M, 2))
            dyn_vel = np.zeros((M, 2))
            for m, j in enumerate(others_idx):
                dyn_vel[m] = vels_xy[j]
                for t in range(N_HORIZON + 1):
                    dyn_pred[t, m] = poses[j, :2] + vels_xy[j] * t * DT_PLAN

            z_opt, _, _, _ = plan_one_step(
                poses[i], goals[i], agent_radius,
                static_obs_pos, dyn_pred, dyn_vel,
                prev_z=prev_z[i], rng=rng,
            )
            prev_z[i] = z_opt
            r_z, theta_z, delta_z, vm_z = z_opt
            if r_z < 1e-3:
                current_target[i] = poses[i].copy()
            else:
                current_target[i] = ego_to_target(poses[i], r_z, theta_z, delta_z)
            current_vmax[i] = vm_z

        # Apply one-step control
        for i in range(N):
            v_cmd, w_cmd = control_cmd(poses[i], current_target[i], current_vmax[i])
            x, y, psi = poses[i]
            x_new = x + v_cmd * np.cos(psi) * DT_SIM
            y_new = y + v_cmd * np.sin(psi) * DT_SIM
            psi_new = wrap_pi(psi + w_cmd * DT_SIM)
            X[i, 0, k + 1] = x_new
            X[i, 1, k + 1] = y_new
            Theta[i, k + 1] = psi_new
            Uhist[i, 0, k] = v_cmd
            Uhist[i, 1, k] = w_cmd

        # Early termination
        dists = np.linalg.norm(X[:, :, k + 1] - goals, axis=1)
        if np.all(dists < 0.3):
            K_steps = k + 1
            X = X[:, :, :K_steps + 1]
            Theta = Theta[:, :K_steps + 1]
            Uhist = Uhist[:, :, :K_steps]
            if verbose:
                print(f"All agents reached goals at step {K_steps} "
                      f"(t={K_steps * DT_SIM:.2f}s)")
            break

        if verbose and (k % 20 == 0):
            max_d = float(np.max(dists))
            print(f"  step {k:4d} | t={k * DT_SIM:5.2f}s | "
                  f"max dist-to-goal = {max_d:.3f}")

    return X, goals, Uhist, K_steps, agent_radius


def main():
    env_type = sys.argv[1] if len(sys.argv) > 1 else 'circle_swap'

    # The paper's 4-robot swap scenario is the only one we reproduce here.
    if env_type not in ('circle_swap', 'circle', '4robots', 'default', None):
        print(f"Note: DS-MPEPC app is configured only for the 4-robot circle "
              f"position-swap scenario. Ignoring env_type={env_type!r}.")

    X, G, Uhist, K, agent_radius = run_circle_swap(N=4, radius=3.0, T_sim=25.0)

    print("\nSaving results...")
    save_csvs(X, G, Uhist, N=4, K=K)
    save_gif(X, G, N=4, env_type='circle_swap', agent_radius=agent_radius)

    print("\nSimulation Results:")
    print(f"Number of steps: {K}   (t={K * DT_SIM:.2f}s)")
    print("Final positions:")
    for i in range(4):
        d = float(np.linalg.norm(X[i, :, -1] - G[i]))
        status = "reached goal" if d < 0.3 else f"dist to goal: {d:.3f}"
        print(f"  Agent {i+1}: ({X[i, 0, -1]:+.3f}, {X[i, 1, -1]:+.3f}) - {status}")


if __name__ == "__main__":
    main()

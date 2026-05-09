"""
DS-MPEPC implementation
from Figure 7 of:

  S. H. Arul, J. J. Park, and D. Manocha,
  "DS-MPEPC: Safe and Deadlock-Avoiding Robot Navigation in Cluttered Dynamic
  Scenes", arXiv:2303.10133, 2023.

Background (MPEPC) follows:

  J. J. Park, C. Johnson, and B. Kuipers,
  "Robot Navigation with Model Predictive Equilibrium Point Control",
  IROS 2012.

Cooperative non-communicative extension
---------------------------------------

Added an alpha_shared value to allow robots to assume shared responsibility for collision avvoidence

"""

import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.patches as patches
from pathlib import Path
from scipy.optimize import minimize

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

# # Cost weights
# W_PROGRESS = 2.0
# W_ACTION_V = 0.15
# W_ACTION_W = 0.05
# W_COLLISION = 0.4
# W_TERMINAL = 5.0

# Weights for doorway no shared responsibility
# W_PROGRESS = 4.9537
# W_ACTION_V = 0.0758
# W_ACTION_W = 0.9790
# W_COLLISION = 4.8877
# W_TERMINAL = 1.5572

# Weights for doorway shared responsibility
W_PROGRESS = 5.0
W_ACTION_V = 0.1
W_ACTION_W = 0.2
W_COLLISION = 1.0
W_TERMINAL = 1.0
# ttgs: 11.6, 15.0, 14.3

# Weights for hallway
# W_PROGRESS = 4.2314
# W_ACTION_V = 0.4040
# W_ACTION_W = 0.2516
# W_COLLISION = 2.6545
# W_TERMINAL = 2.3558
# ttgs: 13.7, 14.1, 16.0

# weights for intersection
# W_PROGRESS = 3.8147
# W_ACTION_V = 0.9022
# W_ACTION_W = 0.2844
# W_COLLISION = 0.5899
# W_TERMINAL = 1.3089
# ttgs: 13.6, 14.2, 14.1

# Cooperative non-communicative extension (see module docstring)
ALPHA_SHARED = 0.5     # responsibility share each agent carries per neighbor

# Planning / simulation
T_HORIZON = 5.0
DT_PLAN = 0.2
N_HORIZON = int(round(T_HORIZON / DT_PLAN))   # 25
DT_SIM = 0.1

# Sampling
N_RANDOM_SAMPLES = 200


# ======================================================================
# Utilities
# ======================================================================
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def get_input(prompt, default, type_cast=str):
    while True:
        user_input = input(f"{prompt} (default: {default}): ")
        if not user_input:
            return default
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input! Please enter a valid {type_cast.__name__}.")

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


def _ttc_disks_vec(dp, dv, r_sum):
    """Vectorized ttc_disks. Inputs broadcast over leading axes.
    dp, dv: arrays with last axis == 2 (relative pos / vel of robot - obstacle).
    Returns ttc with shape = broadcast of dp.shape[:-1] and dv.shape[:-1]."""
    a = np.sum(dv * dv, axis=-1)
    b = 2.0 * np.sum(dp * dv, axis=-1)
    c = np.sum(dp * dp, axis=-1) - r_sum * r_sum
    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(a < 1e-12, 1.0, a)
    t1 = (-b - sqrt_disc) / (2.0 * a_safe)
    ttc = np.where(t1 > 0.0, t1, np.inf)
    ttc = np.where((disc < 0.0) | (a < 1e-12), np.inf, ttc)
    ttc = np.where(c <= 0.0, 0.0, ttc)
    return ttc


def _pc_eq5_vec(d_eff, ttc, sigma_d):
    """Vectorized pc_eq5."""
    safe_d = np.maximum(d_eff, 0.0)
    reactive = np.exp(-(safe_d ** 2) / (sigma_d ** 2))
    finite_pos = np.isfinite(ttc) & (ttc > 0.0)
    safe_ttc = np.where(finite_pos, ttc, 1.0)
    inv_ttc_sq = np.where(finite_pos, (1.0 / safe_ttc) ** 2, 0.0)
    anticip = 1.0 - A_ANTICIP * np.exp(-inv_ttc_sq / (SIGMA_INV_TTC ** 2))
    # ttc == 0 (already touching): anticipatory factor = 1
    anticip = np.where(np.isfinite(ttc) & (ttc <= 0.0), 1.0, anticip)
    pc = reactive * anticip
    pc = np.where(d_eff <= 0.0, 1.0, pc)
    return pc


# ======================================================================
# Trajectory cost
# ======================================================================
def trajectory_cost(poses, vels, goal_xy, agent_radius,
                    static_obs_pos, dyn_obs_pred, dyn_obs_vel,
                    vmax, dt=DT_PLAN, cooperative=True):
    """Compute J_tilde(q_{z*}) for a simulated trajectory.

    cooperative=True assumes every entry of dyn_obs_pred is another instance
    of this same planner, so each agent carries only ALPHA_SHARED of the
    pairwise collision-avoidance burden. Set cooperative=False to recover
    the original non-reciprocal behavior (dynamic neighbors treated as
    adversarial constant-velocity obstacles).
    """
    N = len(vels)
    r_sum_dyn = 2.0 * agent_radius
    # Static obstacles in this codebase are visualized as disks of agent_radius,
    # so the effective collision sum is also 2 * agent_radius.
    r_sum_stat = 2.0 * agent_radius

    # --- robot pos/velocity at every step (heading-aligned linear vel) ---
    p_r = poses[:, :2]                                          # (N+1, 2)
    v_lin = np.concatenate([vels[:, 0], vels[-1:, 0]])          # (N+1,)
    psi = poses[:, 2]
    v_r = np.stack([v_lin * np.cos(psi), v_lin * np.sin(psi)], axis=1)  # (N+1, 2)

    # --- per-step collision probability ---
    # Static and dynamic contributions are computed separately and combined
    # as independent events so we can scale the dynamic part by the
    # shared-responsibility factor without diluting the static term.
    pc_stat = np.zeros(N + 1)
    pc_dyn = np.zeros(N + 1)

    if static_obs_pos is not None and len(static_obs_pos) > 0:
        static_arr = np.asarray(static_obs_pos, dtype=float)    # (S, 2)
        dp_s = p_r[:, None, :] - static_arr[None, :, :]         # (N+1, S, 2)
        dv_s = v_r[:, None, :]                                  # (N+1, 1, 2); obs vel = 0
        d_eff_s = np.linalg.norm(dp_s, axis=-1) - r_sum_stat    # (N+1, S)
        ttc_s = _ttc_disks_vec(dp_s, dv_s, r_sum_stat)
        pc_s = _pc_eq5_vec(d_eff_s, ttc_s, SIGMA_D_STATIC)
        pc_stat = pc_s.max(axis=1)

    D = dyn_obs_pred.shape[1] if dyn_obs_pred.ndim == 3 else 0
    if D > 0:
        dp_d = p_r[:, None, :] - dyn_obs_pred                   # (N+1, D, 2)
        dv_d = v_r[:, None, :] - dyn_obs_vel[None, :, :]        # (N+1, D, 2)
        d_eff_d = np.linalg.norm(dp_d, axis=-1) - r_sum_dyn
        ttc_d = _ttc_disks_vec(dp_d, dv_d, r_sum_dyn)
        pc_d = _pc_eq5_vec(d_eff_d, ttc_d, SIGMA_D_DYNAMIC)
        share = ALPHA_SHARED if cooperative else 1.0
        pc_dyn = share * pc_d.max(axis=1)

    # Combine static + dynamic pc as independent events. Equivalent to the
    # original max() formulation when one source dominates, but additive on
    # the survivability product 1 - pc, which is what we actually use.
    pc_arr = 1.0 - (1.0 - pc_stat) * (1.0 - pc_dyn)

    # --- survivability (cumulative product) ---
    ps_arr = np.cumprod(1.0 - pc_arr)

    # --- progress toward goal, weighted by ps ---
    d_to_goal = np.linalg.norm(p_r - goal_xy, axis=1)           # (N+1,)
    delta_d = d_to_goal[1:] - d_to_goal[:-1]                    # (N,)
    J_progress = float(np.sum(ps_arr[1:] * W_PROGRESS * delta_d))

    # --- action cost ---
    J_action = float(np.sum(W_ACTION_V * vels[:, 0] ** 2
                            + W_ACTION_W * vels[:, 1] ** 2) * dt)

    # --- collision cost ---
    phi_col = 1.0
    J_collision = float(np.sum((1.0 - ps_arr[1:]) * W_COLLISION * phi_col))

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
    ttg = d_goal / v_toward if v_toward > 1e-3 else np.inf

    v_term_vec = vmax * heading
    ttc_term = np.inf
    if static_obs_pos is not None and len(static_obs_pos) > 0:
        static_arr = np.asarray(static_obs_pos, dtype=float)
        dp_t = term_pose[:2] - static_arr                        # (S, 2)
        ttc_s_term = _ttc_disks_vec(dp_t, v_term_vec, r_sum_stat)
        ttc_term = min(ttc_term, float(ttc_s_term.min()))
    if D > 0:
        dp_t = term_pose[:2] - dyn_obs_pred[-1]                  # (D, 2)
        dv_t = v_term_vec - dyn_obs_vel                          # (D, 2)
        ttc_d_term = _ttc_disks_vec(dp_t, dv_t, r_sum_dyn)
        ttc_term = min(ttc_term, float(ttc_d_term.min()))

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
                  prev_z=None, rng=None, cooperative=True):
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

    # random coverage — widened to allow lateral and rear-aimed targets so the
    # planner can find escape maneuvers from symmetric deadlocks.
    for _ in range(N_RANDOM_SAMPLES):
        r = rng.uniform(0.2, 6.0)
        theta = rng.uniform(-np.pi, np.pi)
        delta = rng.uniform(-np.pi, np.pi)
        vm = rng.uniform(0.0, VMAX)
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
                               static_obs_pos, dyn_obs_pred, dyn_obs_vel, vmax,
                               cooperative=cooperative)
        if cost < best_cost:
            best_cost = cost
            best = z
            best_poses = poses
            best_vels = vels

    # gradient-based refinement from best candidate
    def cost_fn(z):
        r, theta, delta, vmax = z
        target = ego_to_target(robot_pose, r, theta, delta)
        poses, vels = simulate_trajectory(robot_pose, target, vmax)
        return trajectory_cost(poses, vels, goal_xy, agent_radius,
                               static_obs_pos, dyn_obs_pred, dyn_obs_vel, vmax)

    bounds = [(0.05, 8.0), (-1.2, 1.2), (-1.8, 1.8), (0.0, VMAX)]
    result = minimize(cost_fn, list(best), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-4})
    if result.fun < best_cost:
        best       = result.x
        best_cost  = result.fun
        target     = ego_to_target(robot_pose, *best[:3])
        best_poses, best_vels = simulate_trajectory(robot_pose, target, best[3])

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


def save_gif(X, G, N, env_type, agent_radius, static_obs_pos=None):
    anim_dir, _ = _logs_dirs()
    filename = anim_dir / f"{env_type}_{N}agents.gif"

    Kp1 = X.shape[2]
    colors = StandardizedEnvironment.AGENT_COLORS

    fig, ax = plt.subplots(figsize=StandardizedEnvironment.FIG_SIZE)
    ax.set_xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
    ax.set_ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"DS-MPEPC: {N}-robot {env_type or 'doorway'}")

    # Static obstacles
    for obs in (static_obs_pos or []):
        ax.add_patch(patches.Circle((float(obs[0]), float(obs[1])),
                                    radius=agent_radius,
                                    facecolor='dimgray', edgecolor='black',
                                    linewidth=0.5, zorder=2))

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


def save_csvs(X, G, Uhist, N, K, goal_threshold=0.4):
    ds_mpepc_dir = Path(__file__).resolve().parents[3] / 'logs' / 'DS-MPEPC' / 'trajectories'
    ds_mpepc_dir.mkdir(parents=True, exist_ok=True)

    ttg_list = []
    all_reached = True
    for i in range(N):
        reached = False
        agent_ttg = -1  # sentinel: did not reach goal
        for k in range(K + 1):
            if np.linalg.norm(X[i, :, k] - G[i]) < goal_threshold:
                agent_ttg = k
                reached = True
                break
        if not reached:
            all_reached = False
        ttg_list.append((i, agent_ttg, reached))

    # -1 means at least one robot failed to reach its goal within the sim window.
    completion_step = max(t for _, t, _ in ttg_list) if all_reached else -1

    with open(ds_mpepc_dir / "completion_step.txt", "w") as f:
        f.write(str(completion_step))

    with open(ds_mpepc_dir / "ttg_dsmpepc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["robot_id", "ttg", "reached_goal"])
        for rid, ttg, reached in ttg_list:
            w.writerow([rid, ttg, reached])

    for i in range(N):
        num_steps = K + 1
        nominal_x = np.linspace(X[i, 0, 0], G[i, 0], num_steps)
        nominal_y = np.linspace(X[i, 1, 0], G[i, 1], num_steps)
        with open(ds_mpepc_dir / f"path_deviation_robot_{i}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["actual_x", "actual_y", "nominal_x", "nominal_y"])
            for k in range(num_steps):
                w.writerow([X[i, 0, k], X[i, 1, k], nominal_x[k], nominal_y[k]])

        with open(ds_mpepc_dir / f"avg_delta_velocity_robot_{i}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vx", "vy"])
            for k in range(K):
                w.writerow([Uhist[i, 0, k], Uhist[i, 1, k]])

    print(f"Trajectory CSVs saved to {ds_mpepc_dir}")



def run_dsmpepc(N=4, radius=3.0, T_sim=25.0, seed=0, verbose=True,
                    targets=None, initials=None, env='doorway', obstacles=None):
    rng = np.random.default_rng(seed)
    # Independent per-agent RNGs so sample streams don't interleave across agents.
    agent_rngs = [np.random.default_rng(seed + 1000 + i) for i in range(N)]

    user_supplied = initials is not None and targets is not None
    if not user_supplied:
        # Cardinal positions -> diagonally opposite
        angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        starts = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        goals = -starts
    else:
        goals = np.stack(targets, axis=0)
        starts = np.stack(initials, axis=0)

    headings = np.arctan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
    if not user_supplied:
        # Break perfectly symmetric ties only in the auto-generated circle scenario.
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

    static_obs_pos = [np.asarray(o, dtype=float) for o in (obstacles or [])]

    print(f"Running DS-MPEPC {env} ({N} agents, radius={radius}, "
          f"static_obstacles={len(static_obs_pos)})...")

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
                prev_z=prev_z[i], rng=agent_rngs[i],
            )
            if z_opt is None:
                # Planner found no usable sample; hold the previous target if any,
                # otherwise stop in place.
                if current_target[i] is None:
                    current_target[i] = poses[i].copy()
                    current_vmax[i] = 0.0
                continue
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

def setup_doorway_scenario():
    """Static-obstacle positions for the doorway scenario."""
    print("Setting up Doorway Environment using standardized configuration...")
    return StandardizedEnvironment.get_doorway_obstacles()

def setup_hallway_scenario():
    """Static-obstacle positions for the hallway scenario."""
    print("Setting up Hallway Environment using standardized configuration...")
    return StandardizedEnvironment.get_hallway_obstacles()

def setup_intersection_scenario():
    """Static-obstacle positions for the intersection scenario."""
    print("Setting up Intersection Environment using standardized configuration...")
    return StandardizedEnvironment.get_intersection_obstacles()

def main():
    env_type = None
    verbose_mode = True  # Default to verbose for backwards compatibility

    if len(sys.argv) > 1:
        env_type = sys.argv[1]

    if len(sys.argv) > 2:
        verbose_arg = sys.argv[2]
        verbose_mode = (verbose_arg == '--verbose')

    obstacle_agents_x = []
    if env_type == 'doorway':
        obstacle_agents_x = setup_doorway_scenario()
    elif env_type == 'hallway':
        obstacle_agents_x = setup_hallway_scenario()
    elif env_type == 'intersection':
        obstacle_agents_x = setup_intersection_scenario()

    # --- Get User Input for Simulation ---

    # Get parameters for the moving drones
    num_moving_drones = get_input("Enter number of moving drones", 2, int)

    # Get simulation parameters from user - optimized per environment
    min_radius = get_input("Enter minimum distance between drones", StandardizedEnvironment.DEFAULT_COLLISION_DISTANCE, float)

    print("\nConfigure moving drones:")

    # Print environment-specific instructions using standardized coordinates
    if env_type == 'doorway':
        print("\nDoorway Configuration:")
        print("- The doorway has a vertical wall at x=0 with a gap between y=-2 and y=2")
        print("- X coordinates should be between -5 and 5")
        print("- Y coordinates should be between -7 and 7")
    elif env_type == 'hallway':
        print("\nHallway Configuration:")
        print("- The hallway has walls at y=-2 and y=2")
        print("- Robots should stay between y=-1.5 and y=1.5 (middle of hallway)")
        print("- X coordinates should be between -5 and 5")
    elif env_type == 'intersection':
        print("\nIntersection Configuration:")
        print("- The intersection has corridors with center at (0, 0)")
        print("- Corridor width extends from -2 to 2 in both directions")
        print("- X and Y coordinates should be between -5 and 5")

    # Get drone positions in ORCA-style individual configuration
    ini_x_moving = []
    target_moving = []

    # Get standardized default positions
    standard_positions = StandardizedEnvironment.get_standard_agent_positions(env_type, num_moving_drones)

    # Convert to the format expected by the rest of the code
    default_positions = []
    for pos in standard_positions:
        default_positions.append({
            'start_x': pos['start'][0],
            'start_y': pos['start'][1],
            'goal_x': pos['goal'][0],
            'goal_y': pos['goal'][1]
        })

    for i in range(num_moving_drones):
        print(f"\n--- Agent {i+1} Parameters ---")

        # Get default values for this drone (cycle through available defaults)
        default_idx = i % len(default_positions)
        defaults = default_positions[default_idx]

        # Get start position
        start_x = get_input(f"Start X position (default: {defaults['start_x']})", defaults['start_x'], float)
        start_y = get_input(f"Start Y position (default: {defaults['start_y']})", defaults['start_y'], float)

        # Get goal position
        goal_x = get_input(f"Goal X position (default: {defaults['goal_x']})", defaults['goal_x'], float)
        goal_y = get_input(f"Goal Y position (default: {defaults['goal_y']})", defaults['goal_y'], float)

        # Store positions
        ini_x_moving.append(np.array([start_x, start_y]))
        target_moving.append(np.array([goal_x, goal_y]))

        print(f"Agent {i+1} configured: Start=({start_x}, {start_y}), Goal=({goal_x}, {goal_y})")

    X, G, Uhist, K, agent_radius = run_dsmpepc(
        N=num_moving_drones, radius=3.0, T_sim=25.0,
        targets=target_moving, initials=ini_x_moving,
        verbose=verbose_mode, obstacles=obstacle_agents_x,
        env=env_type or 'circle swap',
    )

    print("\nSaving results...")
    save_csvs(X, G, Uhist, N=num_moving_drones, K=K)
    save_gif(X, G, N=num_moving_drones, env_type=env_type,
             agent_radius=agent_radius, static_obs_pos=obstacle_agents_x)

    print("\nSimulation Results:")
    print(f"Number of steps: {K}   (t={K * DT_SIM:.2f}s)")
    print("Final positions:")
    for i in range(num_moving_drones):
        d = float(np.linalg.norm(X[i, :, -1] - G[i]))
        status = "reached goal" if d < 0.3 else f"dist to goal: {d:.3f}"
        print(f"  Agent {i+1}: ({X[i, 0, -1]:+.3f}, {X[i, 1, -1]:+.3f}) - {status}")


if __name__ == "__main__":
    main()

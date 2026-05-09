import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.patches as patches
from pathlib import Path
from scipy.optimize import minimize

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[3] / 'src'))
from utils import StandardizedEnvironment

# -------------------- Parameters --------------------
K1 = 1.5
K2 = 3.0
BETA = 0.4
LAMBDA = 2.0
R_THRESH = 1.2      # user config w/ 1.2 as default

VMAX = 1.0          # user config w/ 1.0 as default

# Uncertainity parameters
SIGMA_D_STATIC = 0.1         
SIGMA_D_DYNAMIC = 0.2          

# Cost weights
W_PROGRESS = 1.0
W_ACTION_V = 0.15
W_ACTION_W = 0.05
W_COLLISION = 0.4

# MPEPC paper cost weigths -->leads to deadlock
# W_PROGRESS = 0.2
# W_ACTION_V = 1
# W_ACTION_W = 0.2
# W_COLLISION = 0.1

# Planning / simulation
T_HORIZON = 5.0
DT_PLAN = 0.2
N_HORIZON = int(round(T_HORIZON / DT_PLAN))   # 25
DT_SIM = 0.1

N_RANDOM_SAMPLES = 180



# -------------------- Utilities --------------------
def wrap_pi(a: float) -> float:
    """Ensures all angles are between [-π, π]"""
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





# -------------------- Collision Probability --------------------
def collision_prob(d_eff, sigma):
    """Eq.(22): Probability of collision given distance to nearest obstacle"""
    if d_eff <= 0:
        return 1.0
    return float(np.exp(-(d_eff **2)  / (sigma ** 2)))





# -------------------- Navigation Function --------------------
def nav_func(pos, goal):
    """Approx. of NF(·) from Eq.(24) -- approxed w/ euclidean dist"""
    return float(np.linalg.norm(pos - goal))





# -------------------- Trajectory Cost --------------------
def trajectory_cost(poses, vels, goal_xy, agent_radius, static_obs_pos, dyn_obs_pred, dt=DT_PLAN):
    """Compute MPEPC trajectory cost J(q_{z*}), Eq.(28)."""
    N = len(vels)
    r_sum_dyn = 2.0 * agent_radius
    r_sum_stat = agent_radius                      # static obstacles treated as points

    # --- per-step collision probability (closest obstacle at each step) ---
    pc_arr = np.zeros(N + 1)
    for k in range(N + 1):
        p_r = poses[k, :2]

        best = 0.0
        # static
        for obs in static_obs_pos:
            d = float(np.linalg.norm(p_r - obs)) - r_sum_stat
            p = collision_prob(d, SIGMA_D_STATIC)
            if p > best:
                best = p
        # dynamic (other agents, const-velocity prediction)
        for j in range(dyn_obs_pred.shape[1]):
            p_o = dyn_obs_pred[k, j]
            d = float(np.linalg.norm(p_r - p_o)) - r_sum_dyn
            p = collision_prob(d, SIGMA_D_DYNAMIC)
            if p > best:
                best = p
        pc_arr[k] = best

    # --- survivability (cumulative product) ---
    ps_arr = np.zeros(N + 1)
    running = 1.0
    for k in range(N + 1):
        running *= (1.0 - pc_arr[k])
        ps_arr[k] = running

    # --- progress toward goal, weighted by ps ---
    J_progress = 0.0
    for k in range(1, N + 1):
        d_k = float(nav_func(poses[k, :2], goal_xy))
        d_km1 = float(nav_func(poses[k - 1, :2], goal_xy))
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

    return J_progress + J_action + J_collision
    




# -------------------- Planner --------------------
def plan_one_step(robot_pose, goal_xy, agent_radius, static_obs_pos, dyn_obs_pred, prev_z=None, rng=None):
    """Choose z* = (r, theta, delta, vmax) minimizing J, Eq.(28)."""
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
        cost = trajectory_cost(poses, vels, goal_xy, agent_radius, static_obs_pos, dyn_obs_pred)
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
                               static_obs_pos, dyn_obs_pred)

    bounds = [(0.05, 8.0), (-1.2, 1.2), (-1.8, 1.8), (0.0, VMAX)]
    result = minimize(cost_fn, list(best), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-4})
    if result.fun < best_cost:
        best       = result.x
        best_cost  = result.fun
        target     = ego_to_target(robot_pose, *best[:3])
        best_poses, best_vels = simulate_trajectory(robot_pose, target, best[3])

    return best, best_cost, best_poses, best_vels





# -------------------- Helpers --------------------
def get_input(prompt, default, type_cast=str):
    while True:
        user_input = input(f"{prompt} (default: {default}): ")
        if not user_input:
            return default
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input! Please enter a valid {type_cast.__name__}.")


def _mpepc_logs_dirs():
    root_dir = Path(__file__).resolve().parents[3]
    anim_dir = root_dir / 'logs' / 'MPEPC' / 'animations'
    traj_dir = root_dir / 'logs' / 'MPEPC' / 'trajectories'
    anim_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    return anim_dir, traj_dir


def save_gif(X, G, N, obstacles, env_type):
    """Save standardized GIF animation of the simulation."""
    anim_dir, _ = _mpepc_logs_dirs()
    filename = anim_dir / f"{env_type}_{N}agents.gif"

    Kp1 = X.shape[2]
    colors = StandardizedEnvironment.AGENT_COLORS

    fig, ax = plt.subplots(figsize=StandardizedEnvironment.FIG_SIZE)
    ax.set_xlim(StandardizedEnvironment.GRID_X_MIN, StandardizedEnvironment.GRID_X_MAX)
    ax.set_ylim(StandardizedEnvironment.GRID_Y_MIN, StandardizedEnvironment.GRID_Y_MAX)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw obstacles as gray circles
    for obs_pos in obstacles:
        circle = patches.Circle(obs_pos, radius=StandardizedEnvironment.DEFAULT_AGENT_RADIUS,
                                facecolor='gray', edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(circle)

    # Goals as green stars
    for i in range(N):
        ax.scatter(G[i, 0], G[i, 1], marker='*', s=300, color='green', edgecolor='black', zorder=4)

    dyn_scatter = ax.scatter([], [], c=[], s=200, edgecolors='black', linewidths=1, label='Agent')

    # Full path traces (faint)
    for i in range(N):
        color = colors[i % len(colors)]
        ax.plot(X[i, 0, :], X[i, 1, :], color=color, alpha=0.25, linewidth=1.2)

    # Legend
    legend_handles = []
    legend_labels = []
    legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                        markersize=10, markerfacecolor='gray', markeredgecolor='black'))
    legend_labels.append('Obstacle')
    for i in range(N):
        color = colors[i % len(colors)]
        legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                            markersize=10, markerfacecolor=color, markeredgecolor='black'))
        legend_labels.append(f'Agent {i+1}')
    legend_handles.append(mlines.Line2D([], [], color='green', marker='*', linestyle='None',
                                        markersize=12, markerfacecolor='green', markeredgecolor='none'))
    legend_labels.append('Goal')

    ax.legend(legend_handles, legend_labels,
              loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, borderaxespad=0., markerscale=1.2)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    # Sample frames for reasonable GIF size
    frame_step = max(1, Kp1 // 200)
    frame_indices = list(range(0, Kp1, frame_step))

    def animate(idx):
        k = frame_indices[idx]
        positions = []
        dyn_colors = []
        for i in range(N):
            positions.append([X[i, 0, k], X[i, 1, k]])
            dyn_colors.append(colors[i % len(colors)])
        dyn_scatter.set_offsets(np.array(positions).reshape(-1, 2))
        dyn_scatter.set_color(dyn_colors)
        return [dyn_scatter]

    anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                         interval=StandardizedEnvironment.ANIMATION_INTERVAL, blit=True)
    anim.save(str(filename), writer='pillow', fps=StandardizedEnvironment.ANIMATION_FPS)
    print(f"GIF animation saved as {filename}")
    plt.close(fig)


def save_csvs(X, G, Uhist, N, K, goal_threshold=0.3):
    """Save trajectory CSVs in the format expected by evaluate_impc_trajectories."""
    mpepc_dir = Path(__file__).resolve().parents[3] / 'logs' / 'MPEPC' / 'trajectories'
    mpepc_dir.mkdir(parents=True, exist_ok=True)

    # Determine goal completion per agent
    ttg_list = []
    completion_step = K  # default: simulation ended without all reaching goals
    all_reached = True

    for i in range(N):
        reached = False
        agent_ttg = K
        for k in range(K + 1):
            dist = np.linalg.norm(X[i, :, k] - G[i])
            if dist < goal_threshold:
                agent_ttg = k
                reached = True
                break
        if not reached:
            all_reached = False
        ttg_list.append((i, agent_ttg, reached))

    # Completion step = when the last agent reached its goal (if all did)
    if all_reached:
        completion_step = max(ttg for _, ttg, _ in ttg_list)
    else:
        completion_step = K

    # Save completion_step.txt
    with open(mpepc_dir / "completion_step.txt", "w") as f:
        f.write(str(completion_step))

    # Save ttg_impc_dr.csv (reuses the same filename convention)
    with open(mpepc_dir / "ttg_impc_dr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["robot_id", "ttg", "reached_goal"])
        for robot_id, ttg, reached in ttg_list:
            writer.writerow([robot_id, ttg, reached])

    # Save path_deviation_robot_*.csv for each agent
    for i in range(N):
        num_steps = K + 1
        # Nominal path: straight line from start to goal
        nominal_x = np.linspace(X[i, 0, 0], G[i, 0], num_steps)
        nominal_y = np.linspace(X[i, 1, 0], G[i, 1], num_steps)

        csv_path = mpepc_dir / f"path_deviation_robot_{i}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actual_x", "actual_y", "nominal_x", "nominal_y"])
            for k in range(num_steps):
                writer.writerow([X[i, 0, k], X[i, 1, k], nominal_x[k], nominal_y[k]])

    # Save avg_delta_velocity_robot_*.csv for each agent
    for i in range(N):
        csv_path = mpepc_dir / f"avg_delta_velocity_robot_{i}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vx", "vy"])
            for k in range(K):
                writer.writerow([Uhist[i, 0, k], Uhist[i, 1, k]])

    print(f"Trajectory CSVs saved to {mpepc_dir}")







def run_mpepc_simulation(
    scenario,
    N,
    X0,
    G,
    vmax=None,
    r_thresh=None,
    weights=None,
    T_sim=30.0,
    dt=DT_SIM,
    goal_threshold=0.3,
    verbose_mode=True,
):
    """Run one MPEPC rollout and return histories plus time-to-goal metrics.

    Tunable parameters (vmax, r_thresh, weights) are written to module globals
    while the rollout runs because the planner reads them from there.
    """
    global VMAX, R_THRESH, W_PROGRESS, W_ACTION_V, W_ACTION_W, W_COLLISION

    if vmax is not None:
        VMAX = float(vmax)
    if r_thresh is not None:
        R_THRESH = float(r_thresh)
    if weights:
        W_PROGRESS = float(weights.get('W_PROGRESS', W_PROGRESS))
        W_ACTION_V = float(weights.get('W_ACTION_V', W_ACTION_V))
        W_ACTION_W = float(weights.get('W_ACTION_W', W_ACTION_W))
        W_COLLISION = float(weights.get('W_COLLISION', W_COLLISION))

    obstacles = []
    if scenario == 'doorway':
        obstacles = StandardizedEnvironment.get_doorway_obstacles()
    elif scenario == 'hallway':
        obstacles = StandardizedEnvironment.get_hallway_obstacles()
    elif scenario == 'intersection':
        obstacles = StandardizedEnvironment.get_intersection_obstacles()
    else:
        print(f"Warning: unknown scenario '{scenario}', using no static obstacles.")

    K = int(round(T_sim / dt))
    agent_radius = StandardizedEnvironment.DEFAULT_AGENT_RADIUS
    plan_every = max(1, int(round(DT_PLAN / dt)))

    static_obs_pos = [np.array(o[:2], float) for o in obstacles]

    X = np.zeros((N, 2, K + 1))
    Theta = np.zeros((N, K + 1))
    Uhist = np.zeros((N, 2, K))

    X[:, :, 0] = X0
    Theta[:, 0] = np.arctan2(G[:, 1] - X0[:, 1], G[:, 0] - X0[:, 0])

    prev_z = [None] * N
    current_target = [None] * N
    current_vmax = [VMAX] * N
    infeasible_count = np.zeros(N, dtype=int)

    print("\nStarting MPEPC simulation...")

    for k in range(K):
        poses = np.column_stack([X[:, 0, k], X[:, 1, k], Theta[:, k]])

        if k == 0:
            vels_xy = np.zeros((N, 2))
        else:
            v_last = np.sqrt(Uhist[:, 0, k - 1] ** 2 + Uhist[:, 1, k - 1] ** 2)
            psi = Theta[:, k]
            vels_xy = np.column_stack([v_last * np.cos(psi), v_last * np.sin(psi)])

        if k % plan_every == 0:
            for i in range(N):
                others_idx = [j for j in range(N) if j != i]
                M = len(others_idx)
                dyn_pred = np.zeros((N_HORIZON + 1, M, 2))
                dyn_vel = np.zeros((M, 2))
                for m, j in enumerate(others_idx):
                    dyn_vel[m] = vels_xy[j]
                    for t in range(N_HORIZON + 1):
                        dyn_pred[t, m] = poses[j, :2] + vels_xy[j] * t * DT_PLAN

                z_opt, _, _, _ = plan_one_step(
                    poses[i], G[i], agent_radius,
                    static_obs_pos, dyn_pred,
                    prev_z=prev_z[i],
                )
                prev_z[i] = z_opt
                r_z, theta_z, delta_z, vm_z = z_opt
                current_target[i] = ego_to_target(poses[i], r_z, theta_z, delta_z)
                current_vmax[i] = vm_z

        for i in range(N):
            v_cmd, w_cmd = control_cmd(poses[i], current_target[i], current_vmax[i])
            x, y, psi = poses[i]
            X[i, 0, k + 1] = x + v_cmd * np.cos(psi) * dt
            X[i, 1, k + 1] = y + v_cmd * np.sin(psi) * dt
            Theta[i, k + 1] = wrap_pi(psi + w_cmd * dt)
            Uhist[i, 0, k] = v_cmd * np.cos(psi)
            Uhist[i, 1, k] = v_cmd * np.sin(psi)

        dists = np.linalg.norm(X[:, :, k + 1] - G, axis=1)
        if np.all(dists < goal_threshold):
            K = k + 1
            X = X[:, :, :K + 1]
            Theta = Theta[:, :K + 1]
            Uhist = Uhist[:, :, :K]
            if verbose_mode:
                print(f"All agents reached goals at step {K} (t={K*dt:.2f}s)")
            break

    if verbose_mode:
        print(f"Infeasible fallback counts per agent: {infeasible_count.tolist()}")

    ttg_steps = np.full(N, K, dtype=int)
    reached_goal = np.zeros(N, dtype=bool)
    for i in range(N):
        for step in range(K + 1):
            if np.linalg.norm(X[i, :, step] - G[i]) <= goal_threshold:
                ttg_steps[i] = step
                reached_goal[i] = True
                break
    ttg_seconds = ttg_steps.astype(float) * dt

    return (
        X,
        Theta,
        Uhist,
        K,
        ttg_steps,
        ttg_seconds,
        reached_goal,
        obstacles,
    )


def main():
    env_type = None
    verbose_mode = True

    if len(sys.argv) > 1:
        env_type = sys.argv[1]
    if len(sys.argv) > 2:
        verbose_mode = (sys.argv[2] == '--verbose')

    # Get obstacles from standardized environment
    obstacles = []
    if env_type == 'doorway':
        obstacles = StandardizedEnvironment.get_doorway_obstacles()
    elif env_type == 'hallway':
        obstacles = StandardizedEnvironment.get_hallway_obstacles()
    elif env_type == 'intersection':
        obstacles = StandardizedEnvironment.get_intersection_obstacles()

    # --- User input for moving agents (matching IMPC-DR / ORCA format) ---
    num_moving = get_input("Enter number of moving agents", 2, int)

    print("\nConfigure moving agents:")

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

    standard_positions = StandardizedEnvironment.get_standard_agent_positions(env_type, num_moving)

    default_positions = []
    for pos in standard_positions:
        default_positions.append({
            'start_x': pos['start'][0],
            'start_y': pos['start'][1],
            'goal_x': pos['goal'][0],
            'goal_y': pos['goal'][1]
        })

    X0_list = []
    G_list = []
    for i in range(num_moving):
        print(f"\n--- Agent {i+1} Parameters ---")
        default_idx = i % len(default_positions)
        defaults = default_positions[default_idx]

        start_x = get_input(f"Start X position", defaults['start_x'], float)
        start_y = get_input(f"Start Y position", defaults['start_y'], float)
        goal_x = get_input(f"Goal X position", defaults['goal_x'], float)
        goal_y = get_input(f"Goal Y position", defaults['goal_y'], float)

        X0_list.append([start_x, start_y])
        G_list.append([goal_x, goal_y])
        print(f"Agent {i+1} configured: Start=({start_x}, {start_y}), Goal=({goal_x}, {goal_y})")

    print("\n--- MPEPC Parameters ---")
    global VMAX, R_THRESH
    VMAX = get_input("Maximum velocity (vmax)", VMAX, float)
    R_THRESH = get_input("Slowdown threshold (r_thresh)", R_THRESH, float)

    N = num_moving
    X0 = np.array(X0_list)
    G = np.array(G_list)


    dt = DT_SIM
    T = 30.0
    K = int(round(T / dt))
    agent_radius = StandardizedEnvironment.DEFAULT_AGENT_RADIUS
    plan_every = max(1, int(round(DT_PLAN / DT_SIM)))

    # format obstacles as 2D points for distance calculations
    static_obs_pos = [np.array(o[:2], float) for o in obstacles]

    X = np.zeros((N, 2, K + 1))
    Theta = np.zeros((N, K + 1))
    Uhist = np.zeros((N, 2, K))

    X[:, :, 0] = X0
    Theta[:, 0] = np.arctan2(G[:, 1] - X0[:, 1], G[:, 0] - X0[:, 0])

    prev_z = [None] * N
    current_target = [None] * N
    current_vmax = [VMAX]  * N
    infeasible_count = np.zeros(N, dtype=int)

    print("\nStarting MPEPC simulation...")

    for k in range(K):
        poses = np.column_stack([X[:, 0, k], X[:, 1, k], Theta[:, k]])

        # estimate current velocities from last command
        if k == 0:
            vels_xy = np.zeros((N, 2))
        else:
            v_last = np.sqrt(Uhist[:, 0, k-1]**2 + Uhist[:, 1, k-1]**2)
            psi    = Theta[:, k]
            vels_xy = np.column_stack([v_last * np.cos(psi), v_last * np.sin(psi)])

        # replan every plan_every steps
        if k % plan_every == 0:
            for i in range(N):
                # build constant velocity predictions for other agents
                others_idx = [j for j in range(N) if j != i]
                M          = len(others_idx)
                dyn_pred   = np.zeros((N_HORIZON + 1, M, 2))
                dyn_vel    = np.zeros((M, 2))
                for m, j in enumerate(others_idx):
                    dyn_vel[m] = vels_xy[j]
                    for t in range(N_HORIZON + 1):
                        dyn_pred[t, m] = poses[j, :2] + vels_xy[j] * t * DT_PLAN

                z_opt, _, _, _ = plan_one_step(
                    poses[i], G[i], agent_radius,
                    static_obs_pos, dyn_pred,
                    prev_z=prev_z[i])

                prev_z[i]         = z_opt
                r_z, theta_z, delta_z, vm_z = z_opt
                current_target[i] = ego_to_target(poses[i], r_z, theta_z, delta_z)
                current_vmax[i]   = vm_z

        # apply one integration step
        for i in range(N):
            v_cmd, w_cmd = control_cmd(poses[i], current_target[i], current_vmax[i])
            x, y, psi    = poses[i]
            X[i, 0, k+1] = x   + v_cmd * np.cos(psi) * dt
            X[i, 1, k+1] = y   + v_cmd * np.sin(psi) * dt
            Theta[i, k+1] = wrap_pi(psi + w_cmd * dt)

            # store as world-frame vx, vy to match save_csvs format
            Uhist[i, 0, k] = v_cmd * np.cos(psi)
            Uhist[i, 1, k] = v_cmd * np.sin(psi)

        # early termination
        dists    = np.linalg.norm(X[:, :, k+1] - G, axis=1)
        all_done = np.all(dists < 0.3)
        if all_done:
            K      = k + 1
            X      = X[:,    :, :K+1]
            Theta  = Theta[:,    :K+1]
            Uhist  = Uhist[:, :, :K]
            print(f"All agents reached goals at step {K} (t={K*dt:.2f}s)")
            break


    if verbose_mode:
        print(f"Infeasible fallback counts per agent: {infeasible_count.tolist()}")

    # Save results
    print("\nSaving results...")
    save_csvs(X, G, Uhist, N, K)
    save_gif(X, G, N, obstacles, env_type or 'default')

    # Print final positions
    print("\nSimulation Results:")
    print(f"Number of steps taken: {K}")
    print("Final positions:")
    for i in range(N):
        dist = np.linalg.norm(X[i, :, -1] - G[i])
        status = "reached goal" if dist < 0.3 else f"dist to goal: {dist:.3f}"
        print(f"  Agent {i+1}: ({X[i, 0, -1]:.3f}, {X[i, 1, -1]:.3f}) - {status}")


if __name__ == "__main__":
    main()
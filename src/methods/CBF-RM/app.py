import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.patches as patches
from pathlib import Path

# Import standardized environment configuration
sys.path.append(str(Path(__file__).resolve().parents[3] / 'src'))
from utils import StandardizedEnvironment


# ============================================================
# Exact single-integrator implementation of Eqs. (9)-(20)
# from:
#   Zhang et al., "Adaptive Deadlock Avoidance for Decentralized
#   Multi-Agent Systems via CBF-Inspired Risk Measurement", ICRA 2025.
#
# What is exact here:
#   - Risk indicator zeta_i(R_i) from Eqs. (9)-(10)
#   - CLF/rotated-CLF constraint Eq. (19a)
#   - Collision CBF constraint Eq. (19b)
#   - Auxiliary deadlock CBF h_Dij and its analytic derivatives
#     for the single-integrator case of Eqs. (13)-(19c)
#   - Practical infeasibility fallback from Remark 4
#
# Important note:
#   The paper text does not provide all numerical gains used in the plots.
#   So the equations are implemented exactly, but the scalar gains below
#   are user-set simulation parameters.
# ============================================================


def rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def d_rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[-s, -c], [c, -s]], dtype=float)


def d_rot2_T(theta: float) -> np.ndarray:
    return d_rot2(theta).T


def wrap_to_pi(theta: float) -> float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


# -------------------- CLF and rotated CLF --------------------
def V_base(x: np.ndarray, g: np.ndarray) -> float:
    e = x - g
    return float(e @ e)


def grad_V_base(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return 2.0 * (x - g)


def Vq_and_grads(x: np.ndarray, g: np.ndarray, theta: float):
    """
    Exact single-integrator version of Eq. (11):
        V_q(x,Q) = ||Qx - g||^2 = ||x - Q^T g||^2

    Returns
    -------
    Vq : float
    grad_x : ndarray shape (2,)
    grad_theta : float
    c = Q^T g : rotated virtual goal in x coordinates
    dc_dtheta : derivative of c wrt theta
    """
    Q = rot2(theta)
    QT = Q.T
    c = QT @ g
    dc_dtheta = d_rot2_T(theta) @ g

    e = x - c
    Vq = e @ e
    grad_x = 2.0 * e
    grad_theta = float(-2.0 * dc_dtheta @ e)
    return float(Vq), grad_x, grad_theta, c, dc_dtheta


# -------------------- Barrier functions --------------------
def hij_and_grad(xi: np.ndarray, xj: np.ndarray, d_safe: float):
    p = xi - xj
    h = float(p @ p - d_safe**2)
    grad_h = 2.0 * p
    return h, grad_h


def psi(h: float, k_psi: float) -> float:
    """
    Smooth decreasing positive function satisfying
    psi(h) > 0, psi'(0)=0, lim_{h->infty} psi(h)=0.
    """
    return float(np.exp(-k_psi * h * h))


def dpsi(h: float, k_psi: float) -> float:
    ph = psi(h, k_psi)
    return float(-2.0 * k_psi * h * ph)


def projection_matrix(v: np.ndarray) -> np.ndarray:
    return float(v @ v) * np.eye(v.size) - np.outer(v, v)


def D_and_grads_single_integrator(
    xi: np.ndarray,
    xj: np.ndarray,
    gi: np.ndarray,
    theta_i: float,
):
    """
    Exact analytic single-integrator reduction of Eqs. (13)-(17).

    For xdot = u, we have f = 0, g = I, G = I.
    Hence
        D_ij = 0.5 * grad(V_q)^T * P_{grad h_ij} * grad(V_q)

    Let
        a = grad(V_q)
        b = grad(h_ij)
    Then
        D = 0.5 * a^T ( ||b||^2 I - b b^T ) a
          = 0.5 * ( ||b||^2 ||a||^2 - (a^T b)^2 )

    Analytic gradients:
        grad_x D = 2||b||^2 a + 2||a||^2 b - 2(a^T b)(a+b)
        dD/dtheta = (da/dtheta)^T P_b a
    """
    _, grad_h = hij_and_grad(xi, xj, d_safe=0.0)
    _, a, _, _, _ = Vq_and_grads(xi, gi, theta_i)
    _, _, _, _, dc_dtheta = Vq_and_grads(xi, gi, theta_i)

    b = grad_h
    Pb = projection_matrix(b)
    D = 0.5 * a @ Pb @ a

    a_norm_sq = float(a @ a)
    b_norm_sq = float(b @ b)
    ab = float(a @ b)
    grad_x_D = 2.0 * b_norm_sq * a + 2.0 * a_norm_sq * b - 2.0 * ab * (a + b)

    da_dtheta = -2.0 * dc_dtheta
    grad_theta_D = float(da_dtheta @ (Pb @ a))
    return float(D), grad_x_D, grad_theta_D


def hD_and_grads(
    xi: np.ndarray,
    xj: np.ndarray,
    gi: np.ndarray,
    theta_i: float,
    d_safe: float,
    eps_D: float,
    k_psi: float,
):
    """
    Exact single-integrator implementation of Eqs. (13)-(15):
        h_Dij = psi(h_ij) ( D_ij - eps )
    """
    h, grad_h = hij_and_grad(xi, xj, d_safe)
    D, grad_x_D, grad_theta_D = D_and_grads_single_integrator(xi, xj, gi, theta_i)

    ph = psi(h, k_psi)
    dph = dpsi(h, k_psi)
    hD = ph * (D - eps_D)

    grad_x_hD = ph * grad_x_D + dph * (D - eps_D) * grad_h
    grad_theta_hD = ph * grad_theta_D
    return float(hD), grad_x_hD, float(grad_theta_hD), float(D)


# -------------------- Risk indicator --------------------
def compute_risk_indicator(
    Xk: np.ndarray,
    Uprev: np.ndarray,
    d_safe: float,
    alpha_gain: float,
    phi_risk: float,
    c_risk: float,
    t_risk: float,
):
    """
    Eq. (9) + Eq. (10)

    To match the paper sentence saying other agents' states and velocities
    are considered in the risk evaluation, we use the pairwise derivative
        hdot_ij = 2 (x_i - x_j)^T (u_i - u_j)
    for single integrators.
    """
    N = Xk.shape[0]
    Ri = np.zeros(N)

    for i in range(N):
        acc = 0.0
        xi = Xk[i]
        ui = Uprev[i]
        for j in range(N):
            if j == i:
                continue
            xj = Xk[j]
            uj = Uprev[j]
            p = xi - xj
            hij = float(p @ p - d_safe**2)
            hdotij = float(2.0 * p @ (ui - uj))
            acc += (-hdotij - alpha_gain * hij)
        Ri[i] = acc / (N - 1) + phi_risk

    s = np.clip(t_risk * (Ri - c_risk), -60.0, 60.0)
    zeta = 1.0 / (1.0 + np.exp(-s))
    return Ri, zeta


# -------------------- QP solver --------------------
def solve_qp(H: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-9):
    """
    Solve the convex QP exactly for this small problem size using an
    active-set / KKT enumeration method:

        min 0.5 y^T H y + c^T y
        s.t. A y <= b

    This is a true QP solve for linear inequality constraints, unlike a
    generic nonlinear optimizer such as SLSQP.

    Notes
    -----
    - H must be positive definite.
    - Because the decision dimension is only 4, enumerating active sets is
      practical and robust here.
    """
    import itertools

    n = H.shape[0]
    m = A.shape[0] if A.ndim == 2 else 0

    def obj(y: np.ndarray) -> float:
        return float(0.5 * y @ H @ y + c @ y)

    candidates = []

    # Unconstrained minimizer.
    try:
        y_unc = -np.linalg.solve(H, c)
        if m == 0 or np.all(A @ y_unc <= b + 1e-8):
            candidates.append((obj(y_unc), y_unc))
    except np.linalg.LinAlgError:
        pass

    if m == 0:
        if candidates:
            return min(candidates, key=lambda t: t[0])[1], True
        return None, False

    # Enumerate active sets up to size n.
    idx = range(m)
    for r in range(1, min(n, m) + 1):
        for active in itertools.combinations(idx, r):
            AI = A[list(active), :]
            bI = b[list(active)]

            KKT = np.block([
                [H, AI.T],
                [AI, np.zeros((r, r))],
            ])
            rhs = np.concatenate([-c, bI])

            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                # Skip linearly dependent or singular active sets.
                continue

            y = sol[:n]
            lam = sol[n:]

            # Primal feasibility for all inequalities.
            if np.any(A @ y > b + 1e-7):
                continue

            # Dual feasibility for active inequalities.
            if np.any(lam < -1e-7):
                continue

            candidates.append((obj(y), y))

    if not candidates:
        return None, False

    y_best = min(candidates, key=lambda t: t[0])[1]
    return y_best, True


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


def _cbfrm_logs_dirs():
    root_dir = Path(__file__).resolve().parents[3]
    anim_dir = root_dir / 'logs' / 'CBF-RM' / 'animations'
    traj_dir = root_dir / 'logs' / 'CBF-RM' / 'trajectories'
    anim_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    return anim_dir, traj_dir


def save_gif(X, G, N, obstacles, env_type):
    """Save standardized GIF animation of the simulation."""
    anim_dir, _ = _cbfrm_logs_dirs()
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
    cbf_rm_dir = Path(__file__).resolve().parents[3] / 'logs' / 'CBF-RM' / 'trajectories'
    cbf_rm_dir.mkdir(parents=True, exist_ok=True)

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
    with open(cbf_rm_dir / "completion_step.txt", "w") as f:
        f.write(str(completion_step))

    # Save ttg_impc_dr.csv (reuses the same filename convention)
    with open(cbf_rm_dir / "ttg_impc_dr.csv", "w", newline="") as f:
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

        csv_path = cbf_rm_dir / f"path_deviation_robot_{i}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actual_x", "actual_y", "nominal_x", "nominal_y"])
            for k in range(num_steps):
                writer.writerow([X[i, 0, k], X[i, 1, k], nominal_x[k], nominal_y[k]])

    # Save avg_delta_velocity_robot_*.csv for each agent
    for i in range(N):
        csv_path = cbf_rm_dir / f"avg_delta_velocity_robot_{i}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vx", "vy"])
            for k in range(K):
                writer.writerow([Uhist[i, 0, k], Uhist[i, 1, k]])

    print(f"Trajectory CSVs saved to {cbf_rm_dir}")


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

    N = num_moving
    X0 = np.array(X0_list)
    G = np.array(G_list)

    # ========================================================
    # Simulation parameters
    # ========================================================
    dt = 0.1  # match step_size used by evaluate_impc_trajectories
    T = 30.0
    K = int(round(T / dt))

    agent_radius = StandardizedEnvironment.DEFAULT_AGENT_RADIUS
    d_safe = 2.0 * agent_radius + 0.08

    # Obstacle sensing range (only add CBF constraints for nearby obstacles)
    obs_sense_range = 3.0

    # QP gains
    gamma_gain = 1.2
    alpha_gain = 5.7
    beta_gain = 1.5
    p_weight = 12.0
    q_weight = 0.24

    phi_risk = 1
    c_risk = 0.3
    t_risk = 0.8

    eps_D = 0.01
    k_psi = 2.5
    omega_c = 0.4

    clip_u = 1.0  # increased for standardized grid scale
    clip_omega = 2.0

    # ========================================================
    # Histories
    # ========================================================
    X = np.zeros((N, 2, K + 1))
    Theta = np.zeros((N, K + 1))
    Uhist = np.zeros((N, 2, K))
    Ohist = np.zeros((N, K))
    Rhist = np.zeros((N, K))
    Zhist = np.zeros((N, K))
    Dmin_hist = np.zeros((N, K))
    infeasible_count = np.zeros(N, dtype=int)

    X[:, :, 0] = X0
    Theta[:, 0] = np.arctan2(G[:, 1] - X0[:, 1], G[:, 0] - X0[:, 0])

    print("\nStarting CBF-RM simulation...")

    for k in range(K):
        Xk = X[:, :, k]
        Tk = Theta[:, k]

        if k == 0:
            Uprev = np.zeros((N, 2))
        else:
            Uprev = Uhist[:, :, k - 1]

        Ri, zeta = compute_risk_indicator(
            Xk, Uprev, d_safe, alpha_gain, phi_risk, c_risk, t_risk
        )
        Rhist[:, k] = Ri
        Zhist[:, k] = zeta

        Uk = np.zeros((N, 2))
        Ok = np.zeros(N)

        for i in range(N):
            xi = Xk[i]
            gi = G[i]
            theta_i = Tk[i]
            z = float(zeta[i])

            # Decision y = [u_x, u_y, omega, delta]
            H = np.diag([2.0, 2.0, 2.0 * q_weight, 2.0 * p_weight])
            c_vec = np.zeros(4)
            A_rows = []
            b_rows = []

            # ---------- Eq. (19a): CLF + rotated CLF ----------
            V_i = V_base(xi, gi)
            gradV = grad_V_base(xi, gi)
            Vq_i, gradVq_x, gradVq_theta, _, _ = Vq_and_grads(xi, gi, theta_i)

            A_clf = np.array([
                (1.0 - z) * gradV[0] + z * gradVq_x[0],
                (1.0 - z) * gradV[1] + z * gradVq_x[1],
                z * gradVq_theta,
                -1.0,
            ])
            b_clf = -gamma_gain * ((1.0 - z) * V_i + z * Vq_i)
            A_rows.append(A_clf)
            b_rows.append(b_clf)

            # ---------- Eq. (19b): pairwise collision CBF (other agents) ----------
            for j in range(N):
                if j == i:
                    continue
                xj = Xk[j]
                h_ij, grad_h = hij_and_grad(xi, xj, d_safe)
                Wi = 0.5
                A_cbf = np.array([-grad_h[0], -grad_h[1], 0.0, 0.0])
                b_cbf = Wi * alpha_gain * h_ij
                A_rows.append(A_cbf)
                b_rows.append(b_cbf)

            # ---------- Collision CBF for nearby obstacles ----------
            for obs_pos in obstacles:
                obs = np.asarray(obs_pos, dtype=float)
                dist_to_obs = np.linalg.norm(xi - obs)
                if dist_to_obs < obs_sense_range:
                    h_obs, grad_h_obs = hij_and_grad(xi, obs, d_safe)
                    Wi = 0.5
                    A_obs = np.array([-grad_h_obs[0], -grad_h_obs[1], 0.0, 0.0])
                    b_obs = Wi * alpha_gain * h_obs
                    A_rows.append(A_obs)
                    b_rows.append(b_obs)

            # ---------- Eq. (19c): auxiliary deadlock CBF (other agents only) ----------
            D_values = []
            for j in range(N):
                if j == i:
                    continue
                xj = Xk[j]
                hD_ij, grad_hD_x, grad_hD_theta, Dij = hD_and_grads(
                    xi, xj, gi, theta_i, d_safe, eps_D, k_psi
                )
                D_values.append(Dij)

                A_hd = np.array([
                    -z * grad_hD_x[0],
                    -z * grad_hD_x[1],
                    -z * grad_hD_theta,
                    0.0,
                ])
                b_hd = z * beta_gain * hD_ij
                A_rows.append(A_hd)
                b_rows.append(b_hd)

            Dmin_hist[i, k] = np.min(D_values) if D_values else np.nan

            A_mat = np.vstack(A_rows) if A_rows else np.zeros((0, 4))
            b_vec = np.array(b_rows) if b_rows else np.zeros(0)

            y, ok = solve_qp(H, c_vec, A_mat, b_vec)

            if ok and y is not None:
                u_star = y[:2]
                omega_star = y[2]
            else:
                # Remark 4 practical fallback
                infeasible_count[i] += 1
                u_star = np.zeros(2)
                omega_star = omega_c

            # Numerical clipping only after optimization
            nu = np.linalg.norm(u_star)
            if nu > clip_u:
                u_star = clip_u * u_star / nu
            omega_star = float(np.clip(omega_star, -clip_omega, clip_omega))

            Uk[i] = u_star
            Ok[i] = omega_star

        X[:, :, k + 1] = Xk + dt * Uk
        Theta[:, k + 1] = np.array([wrap_to_pi(Tk[i] + dt * Ok[i]) for i in range(N)])
        Uhist[:, :, k] = Uk
        Ohist[:, k] = Ok

        # Early termination: check if all agents reached goals
        all_done = True
        for i in range(N):
            if np.linalg.norm(X[i, :, k + 1] - G[i]) > 0.3:
                all_done = False
                break
        if all_done:
            # Truncate histories to actual length
            K = k + 1
            X = X[:, :, :K + 1]
            Theta = Theta[:, :K + 1]
            Uhist = Uhist[:, :, :K]
            Ohist = Ohist[:, :K]
            Rhist = Rhist[:, :K]
            Zhist = Zhist[:, :K]
            Dmin_hist = Dmin_hist[:, :K]
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

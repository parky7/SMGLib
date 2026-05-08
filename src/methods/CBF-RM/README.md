# CBF-RM

CBF-RM (Control-Barrier-Function Risk Measurement) is a decentralized method for adaptive deadlock avoidance in multi-agent systems. Each agent solves a small QP at every step that combines a Control Lyapunov Function (CLF) for goal convergence, a collision Control Barrier Function (CBF) for safety against neighbors and static obstacles, and an *auxiliary deadlock CBF* that activates in proportion to a CBF-inspired risk indicator. When the risk indicator detects that an agent is approaching a stalled, symmetric configuration, the rotated-CLF formulation lets it pick a perturbed virtual goal — breaking symmetry locally without inter-agent communication.

This implementation follows the equations from the ICRA 2025 paper exactly for the single-integrator case. The risk indicator $\zeta_i(R_i)$ (Eqs. 9–10), the CLF / rotated-CLF constraint (Eq. 19a), the collision CBF constraint (Eq. 19b), and the auxiliary deadlock CBF $h_{D_{ij}}$ with its analytic derivatives (Eqs. 13–19c) are all implemented as written. The numerical gains used in the paper plots are not all reported in the text, so the scalar gains in [app.py](app.py) are user-set simulation parameters.

## Reference

Zhang et al., *"Adaptive Deadlock Avoidance for Decentralized Multi-Agent Systems via CBF-Inspired Risk Measurement,"* ICRA 2025.

## Running

CBF-RM is wired into the top-level `run_simulation.py` menu — the easiest way to run it is from the repo root and pick the CBF-RM option for the desired scenario.

To run it directly:

```
cd ./src/methods/CBF-RM/
python app.py {doorway|hallway|intersection} [--verbose]
```

You will be prompted for the number of moving agents and, for each agent, start/goal coordinates. Pressing Enter at any prompt accepts the standardized default for that scenario (from `src/utils.py::StandardizedEnvironment`).

After the simulation completes, the following are written to `./src/methods/CBF-RM/`:
- `cbf_rm.gif` — animation of the run
- `path_deviation_robot_*.csv` — per-step deviation from the straight-line path for each agent
- `avg_delta_velocity_robot_*.csv` — per-step velocity-change magnitudes for each agent

## Tunable parameters

The QP gains, risk-measure parameters, and saturation limits are set in `main()` of [app.py](app.py).

- `dt`, `T` — simulation step and total time
- `agent_radius`, `d_safe` — agent radius and pairwise safety distance used by the collision CBF
- `obs_sense_range` — range within which static obstacles contribute CBF constraints
- `gamma_gain`, `alpha_gain`, `beta_gain` — CLF / collision-CBF / deadlock-CBF class-K gains
- `p_weight`, `q_weight` — QP slack penalties on the CLF constraint and the rotation control input
- `phi_risk`, `c_risk`, `t_risk` — risk-indicator $\zeta_i(R_i)$ shape parameters
- `eps_D`, `k_psi`, `omega_c` — auxiliary deadlock CBF parameters (smoothing and rotation cap)
- `clip_u`, `clip_omega` — saturation limits on translational and rotational control

## Setting up a new environment

New scenarios should be added through `StandardizedEnvironment` in [src/utils.py](../../utils.py) so the obstacle layout and default agent positions stay consistent across all methods in SMGLib. Once registered there, the new scenario name can be passed as the first positional argument to `app.py`.

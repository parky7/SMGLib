# DS-MPEPC

DS-MPEPC (Deadlock-aware Sampling-based Model Predictive Equilibrium Point Control) is a sampling-based MPC method for safe, deadlock-avoiding navigation in cluttered dynamic scenes. Each agent rolls out the smooth equilibrium-point control law of Park & Kuipers over a finite horizon, then selects the candidate trajectory that minimizes a cost combining progress-to-goal, control effort, and a probabilistic collision term built from inverse time-to-collision and inverse time-to-goal. The deadlock-avoidance behavior emerges from the anticipation term in the cost — agents that would otherwise stall in symmetric encounters break symmetry by trading a small amount of progress for lower expected collision risk.

This implementation also includes a **cooperative non-communicative extension**: an `ALPHA_SHARED` parameter that lets each agent assume a configurable share of mutual collision-avoidance responsibility for its neighbors. With `ALPHA_SHARED = 0.5` the agents split responsibility evenly, which we found gives smoother joint behavior in the symmetric mini-game scenarios (doorway / hallway / intersection) than the original single-agent formulation.

## Reference

S. H. Arul, J. J. Park, and D. Manocha,
*"DS-MPEPC: Safe and Deadlock-Avoiding Robot Navigation in Cluttered Dynamic Scenes,"* arXiv:2303.10133, 2023.

Background:
J. J. Park, C. Johnson, and B. Kuipers, *"Robot Navigation with Model Predictive Equilibrium Point Control,"* IROS 2012.

## Running

DS-MPEPC is wired into the top-level `run_simulation.py` menu — the easiest way to run it is from the repo root and pick the DS-MPEPC option for the desired scenario.

To run it directly:

```
cd ./src/methods/DS-MPEPC/
python DSMPEPC.py {doorway|hallway|intersection} [--verbose]
```

You will be prompted for the number of moving agents and, for each agent, start/goal coordinates. Pressing Enter at any prompt accepts the standardized default for that scenario (from `src/utils.py::StandardizedEnvironment`).

After the simulation completes, the following are written to `./src/methods/DS-MPEPC/`:
- `dsmpepc.gif` — animation of the run
- `path_deviation_robot_*.csv` — per-step deviation from the straight-line path for each agent
- `avg_delta_velocity_robot_*.csv` — per-step velocity-change magnitudes for each agent
- `ttg_dsmpepc.csv` — time-to-goal for each agent

## Tunable parameters

The cost weights and gains are defined as module-level constants near the top of [DSMPEPC.py](DSMPEPC.py). The defaults checked in are tuned for the doorway scenario with shared responsibility; commented blocks for hallway and intersection are kept inline for reference.

- `K1`, `K2`, `BETA`, `LAMBDA` — smooth control-law gains (Park & Kuipers)
- `R_THRESH` — switching radius for the smooth control law
- `VMAX` — maximum forward speed
- `W_PROGRESS`, `W_ACTION_V`, `W_ACTION_W`, `W_COLLISION`, `W_TERMINAL` — cost weights for the sampled rollouts
- `A_ANTICIP`, `SIGMA_INV_TTC`, `SIGMA_INV_TTG` — anticipation / time-to-collision / time-to-goal scales (paper values)
- `SIGMA_D_STATIC`, `SIGMA_D_DYNAMIC` — collision-cost spreads for static vs. dynamic obstacles
- `ALPHA_SHARED` — cooperative-responsibility share per neighbor (cooperative non-communicative extension)
- `T_HORIZON`, `DT_PLAN`, `DT_SIM` — planning horizon, planning step, simulation step
- `N_RANDOM_SAMPLES` — number of candidate trajectories sampled per agent per planning step

## Setting up a new environment

New scenarios should be added through `StandardizedEnvironment` in [src/utils.py](../../utils.py) so the obstacle layout and default agent positions stay consistent across all methods in SMGLib. Once registered there, the new scenario name can be passed as the first positional argument to `DSMPEPC.py`.

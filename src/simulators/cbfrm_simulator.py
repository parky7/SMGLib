"""
CBF-RM (Control Barrier Function - Risk Measurement) simulator.
"""

import sys
from pathlib import Path

# Import from the original run_simulation.py for now (maintains exact functionality)
sys.path.append(str(Path(__file__).parent.parent.parent))
import run_simulation


def run_cbf_rm_simulation(env_type: str, verbose: bool = False):
    """
    Run CBF-RM simulation.

    Args:
        env_type: Environment type ('hallway', 'doorway', 'intersection')
        verbose: Whether to show detailed output

    Returns:
        dict: Simulation results including makespan, flow_rate, completion_data
    """
    try:
        return run_simulation.run_cbf_rm(env_type, verbose=verbose)
    except Exception as e:
        print(f"CBF-RM simulation error: {e}")
        return None

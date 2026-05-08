"""
DS-MPEPC (Decentralized Social Model Predictive Enhanced Priority Control) simulator.
"""

import sys
import os
from pathlib import Path

# Import from the original run_simulation.py for now (maintains exact functionality)
sys.path.append(str(Path(__file__).parent.parent.parent))
import run_simulation

def run_impc_simulation(num_robots: int, env_type: str):
    """
    Run DS-MPEPC simulation.
    
    Args:
        num_robots: Number of robots to simulate
        env_type: Environment type ('hallway', 'doorway', 'intersection')
    
    Returns:
        dict: Simulation results including makespan, flow_rate, completion_data
    """
    try:
        # Use the original run_dsmpepc function
        return run_simulation.run_dsmpepc(num_robots, env_type)
    except Exception as e:
        print(f"DS-MPEPC simulation error: {e}")
        return None 
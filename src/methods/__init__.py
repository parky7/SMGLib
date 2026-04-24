"""
Methods package containing different social navigation algorithms.

Available algorithms:
- CBF-RM: Adaptive Deadlock Avoidance for Decentralized Multi-Agent Systems via CBF-Inspired Risk Measurement
- MPEPC: Model Predictive Equilibrium Point Control
- DS-MPEPC: Safe and Deadlock-Avoiding Robot Navigation in Cluttered Dynamic Scenes
- Social-ORCA: Optimal Reciprocal Collision Avoidance
- Social-CADRL: Collision Avoidance with Deep Reinforcement Learning  
- Social-IMPC-DR: Integrated Model Predictive Control with Deadlock Resolution
"""

AVAILABLE_METHODS = [
    "CBF-RM",
    "DS-MPEPC",
    "MPEPC",
    "Social-ORCA",
    "Social-CADRL", 
    "Social-IMPC-DR"
] 
"""
===============================================================================
Module:       base_agent.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Parent class for all Reinforcement Learning (RL) agents.

    Defines the shared infrastructure for loading pre-trained Stable Baselines3
    models and running inference against the current facility state:
        1. Model Loading:            Loads saved neural network weights (.zip files).
        2. Observation Normalization: Scales state inputs to the [-1, 1] range
                                     required by neural network policies.
        3. Inference:                Passes the normalized state to the policy network
                                     to predict the next control action.

    This tier is not yet implemented. The class skeleton is provided to define
    the interface contract and document the intended architecture.

Inputs:
    - Path to trained model file (.zip), raw state dictionary
Outputs:
    - Action dictionary (predicted control signals)
Dependencies:
    - stable_baselines3, numpy, ..interface
===============================================================================
"""

# Not yet implemented.
# The RL tier is scoped as a stretch goal for this capstone. The file structure
# and docstrings are provided to define the intended architecture and interface
# contract. Implementation requires a trained Stable Baselines3 policy and a
# gymnasium-compatible environment wrapping the relevant digital twin.

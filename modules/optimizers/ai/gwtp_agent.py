"""
===============================================================================
Module:       gwtp_agent.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    RL agent for the Groundwater Treatment Plant (stub).

    Intended to learn a pump scheduling policy using Proximal Policy
    Optimization (PPO) or Deep Q-Network (DQN) trained in a gym environment
    wrapping the GWTP digital twin.

    Observation space: [GST level, Hydro level, time of day, price, demand forecast]
    Action space:      Discrete pump combinations (well on/off, booster count)
    Reward:            Negative energy cost, penalized for constraint violations

    This tier is not yet implemented. See base_agent.py for the intended
    loading and inference architecture.

Inputs:
    - Raw state dictionary from get_system_state()
Outputs:
    - Control dictionary matching the facility's solve() return format
Dependencies:
    - stable_baselines3, gymnasium, .base_agent
===============================================================================
"""

# Not yet implemented.
# The RL tier is scoped as a stretch goal for this capstone. The file structure
# and docstrings are provided to define the intended architecture and interface
# contract. Implementation requires a trained Stable Baselines3 policy and a
# gymnasium-compatible environment wrapping the relevant digital twin.

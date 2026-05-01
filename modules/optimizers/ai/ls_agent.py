"""
===============================================================================
Module:       ls_agent.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    RL agent for the Main Influent Lift Station (stub).

    Intended to learn a pre-pumping strategy (draining the wet well before
    anticipated inflow events) using PPO or Multi-Discrete DQN.

    Observation space: [wet well level, inflow rate, rain forecast, price]
    Action space:      Multi-Discrete pump on/off per pump
    Reward:            Negative energy cost, large penalty for SSO (overflow)

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

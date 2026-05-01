"""
===============================================================================
Module:       interface.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Abstract Base Class (ABC) that defines the control contract every
    optimizer in this framework must satisfy.

    I use the Strategy Pattern here: main.py holds a reference to an
    OptimizerInterface and calls solve() without knowing whether the
    underlying implementation is MILP, NMPC, or an RL agent. Swapping
    strategies at runtime requires no changes to the orchestration layer.

    All three optimizer tiers (milp/, nmpc/, ai/) inherit from this class.
    The solve() signature is identical across tiers; differences in physics
    fidelity and solve time are internal to each implementation.

Inputs:       None (abstract definition only)
Outputs:      None (abstract definition only)
Dependencies: abc
===============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class OptimizerInterface(ABC):
    """
    Abstract Base Class for all optimization strategies.

    Enforces a single public entry point -- solve() -- so the Orchestration
    Layer (main.py) can execute any controller without knowing its internal
    logic (MILP, NMPC, RL, etc.).
    """

    @abstractmethod
    def solve(self,
              initial_state:   Dict[str, Any],
              price_forecast:  List[float],
              demand_forecast: List[float]) -> Dict[str, Any]:
        """
        Receding-horizon control solve from the current facility state.

        Args:
            initial_state:   Current physical state (tank levels, DO, etc.).
                             Keys are facility-specific (see each subclass).
            price_forecast:  Electricity prices ($/kWh) for the optimization
                             horizon, one entry per 15-minute timestep.
            demand_forecast: Water demand or influent flow for the horizon,
                             in units appropriate to the facility (ft^3/hr
                             for GWTP, MGD for WWTP and LS).

        Returns:
            Control dictionary whose keys match what main.py expects per
            facility. Only the first-step values are dispatched; the rest
            of the horizon is discarded (receding-horizon principle).
        """
        pass

"""
===============================================================================
Module:       base_milp.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Abstract parent class for all Mixed-Integer Linear Programming (MILP)
    controllers. Wraps the Pyomo modeling environment and GLPK solver with
    shared infrastructure used by all three facility optimizers.

    Core responsibilities:
        1. Solver Initialization: Connects to GLPK with performance tuning
           (2% MIP gap, 60-second time limit) applied uniformly so each
           facility optimizer does not need to manage solver options.
        2. Interface Adherence: Inherits from OptimizerInterface so MILP
           controllers are interchangeable with NMPC and AI tiers in main.py.
        3. Model Scaffolding: Provides _setup_base_model() to construct a
           Pyomo ConcreteModel with a standard time index, reducing boilerplate
           in each facility subclass.
        4. Timing Infrastructure: Records wall-clock solve time per call via
           self.solve_times for the comparative solver timing table in main.py.

Inputs:
    - config: Facility configuration dictionary (gwtp_config.yaml, etc.)
Outputs:
    - Base class for GWTP_MILP, WWTP_MILP, LiftStationMILP
Dependencies:
    - pyomo, numpy, logging, .interface
===============================================================================
"""

import time
import numpy as np
import pyomo.environ as pyo
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from modules.optimizers.interface import OptimizerInterface


class BaseMILP(OptimizerInterface, ABC):
    """
    Abstract Base Class for MILP-based control strategies.

    Provides the GLPK solver engine, Pyomo model scaffolding, and timing
    infrastructure. Child classes implement solve() with facility-specific
    variables, constraints, and objectives.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MILP infrastructure with GLPK performance tuning.

        I set mipgap=0.02 (stop within 2% of optimal) and tmlim=60 (hard
        60-second cutoff) to prevent individual solve calls from blocking the
        672-step simulation loop. For lift station problems the child class
        overrides tmlim to 5 seconds since the 8-hour horizon is simpler.

        Args:
            config: Facility configuration dictionary.
        """
        self.config         = config
        self.solve_times:   List[float] = []
        self.demand_penalty = 0.0

        try:
            self.solver = pyo.SolverFactory('glpk')
            if not self.solver.available():
                raise ValueError("GLPK solver not found in system path.")

            # 2% MIP gap: accept solutions within 2% of optimal to reduce solve time.
            # 60-second wall-clock limit: hard cutoff to prevent simulation hangs.
            self.solver.options['mipgap'] = 0.02
            self.solver.options['tmlim']  = 60

            logging.info("GLPK solver initialized (mipgap=2%, tmlim=60s).")

        except Exception as e:
            logging.error(f"Failed to initialize GLPK solver: {e}")
            raise

    @abstractmethod
    def solve(self,
              initial_state:   Dict[str, Any],
              price_forecast:  List[float],
              demand_forecast: List[float]) -> Dict[str, Any]:
        """Implemented by facility subclasses (gwtp_milp, wwtp_milp, ls_milp)."""
        pass

    def _setup_base_model(self, horizon: int) -> pyo.ConcreteModel:
        """
        Create a Pyomo ConcreteModel with a zero-based time index of length horizon.

        Args:
            horizon: Number of timesteps in the optimization horizon.

        Returns:
            pyo.ConcreteModel with model.T = RangeSet(0, horizon - 1).
        """
        model   = pyo.ConcreteModel()
        model.T = pyo.RangeSet(0, horizon - 1)
        return model

    def solver_stats(self) -> Dict[str, float]:
        """
        Return descriptive timing statistics over all recorded solve calls.

        Returns:
            Dict with keys: mean, p95, max (wall-clock seconds), n (call count).
        """
        if not self.solve_times:
            return {'mean': 0.0, 'p95': 0.0, 'max': 0.0, 'n': 0}
        arr = np.array(self.solve_times)
        return {
            'mean': float(arr.mean()),
            'p95':  float(np.percentile(arr, 95)),
            'max':  float(arr.max()),
            'n':    len(arr),
        }

    def calculate_energy_cost(self,
                               power_draw_kw: List[float],
                               prices:        List[float],
                               dt:            float = 0.25) -> float:
        """
        Post-solve energy cost calculation for financial analysis.

        Args:
            power_draw_kw: Instantaneous power draw per timestep (kW).
            prices:        Electricity price per timestep ($/kWh).
            dt:            Timestep duration in hours (default: 0.25 for 15-min).

        Returns:
            Total energy cost in dollars.
        """
        return sum(p * cost * dt for p, cost in zip(power_draw_kw, prices))

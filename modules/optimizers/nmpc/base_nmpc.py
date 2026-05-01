"""
===============================================================================
Module:       base_nmpc.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Abstract parent class for all Nonlinear Model Predictive Control (NMPC)
    controllers. Wraps the do-mpc framework (v5.1.1) backed by CasADi
    automatic differentiation and the Ipopt interior-point NLP solver.

    Core responsibilities:
        1. Solver Configuration: Standard Ipopt options (tolerance, iteration
           limit, warm-start, output suppression) applied uniformly.
        2. Interface Adherence: Inherits OptimizerInterface; all child classes
           expose the same solve() contract as the MILP tier.
        3. Horizon Management: Prediction horizon N and timestep dt_hr are
           set here and overridable per-facility.
        4. Forecast Injection: Stores the current price/demand forecast in
           instance variables so the TVP callback can access them; resets
           do-mpc's internal clock (_t0) to 0 before each solve call so
           the TVP always indexes from step 0.
        5. Timing Infrastructure: Every solve() call records wall-clock time
           via time.perf_counter() for the comparative solver timing table.

    Nonlinear Physics Summary (vs. MILP linearizations):
        GWTP: Power = P_rated * u^3  (VFD affinity law; MILP uses P_rated * u)
              Flow = Q_rated * u     (affinity law 1, same as MILP)
        WWTP: gain = KLa * u * (DO_sat - DO(t))  (exact; MILP fixes deficit)
        LS:   Continuous pump count relaxation (MILP uses integer variables)

Inputs:       None (Abstract Definition)
Outputs:      None
Dependencies: do_mpc, casadi, numpy, abc, ..interface
===============================================================================
"""

import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from modules.optimizers.interface import OptimizerInterface

# ---------------------------------------------------------------------------
# Ipopt solver options applied uniformly to every NMPC instance.
# Tolerances are relaxed relative to the Ipopt default (1e-8) to balance
# solution accuracy with the throughput demands of a 672-step receding-
# horizon loop (7 days x 96 steps/day).
# ---------------------------------------------------------------------------
_IPOPT_OPTS = {
    'ipopt.print_level':           0,
    'ipopt.sb':                    'yes',
    'ipopt.max_iter':              300,
    'ipopt.tol':                   1e-5,
    'ipopt.dual_inf_tol':          1e-4,
    'ipopt.constr_viol_tol':       1e-5,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.warm_start_bound_push': 1e-6,
    'ipopt.warm_start_mult_bound_push': 1e-6,
    'print_time':                  0,
}


class BaseNMPC(OptimizerInterface, ABC):
    """
    Abstract Base Class for do-mpc / CasADi / Ipopt control strategies.

    Child classes (GWTP_NMPC, WWTP_NMPC, LiftStation_NMPC) implement:
        _build_model()  -> do_mpc.model.Model  (states, inputs, TVP, dynamics)
        _build_mpc()    -> do_mpc.controller.MPC  (objective, constraints)
        solve()         -> Dict[str, Any]  (OptimizerInterface contract)
    """

    N_HORIZON: int   = 96     # Default: 24-hr prediction horizon at 15-min res
    DT_HR:     float = 0.25   # 15-minute timestep in hours

    def __init__(self, config: Dict[str, Any], demand_penalty: float = 0.0):
        self.config         = config
        self.demand_penalty = demand_penalty

        # Forecast storage: populated by solve() before each MPC step so the
        # TVP callback (which receives only t_now, not the forecast directly)
        # can index into the current prediction window.
        self._price_fcst:  List[float] = [0.05] * (self.N_HORIZON + 1)
        self._demand_fcst: List[float] = [1.0]  * (self.N_HORIZON + 1)

        # Solve-time telemetry (wall-clock seconds per call)
        self.solve_times: List[float] = []

        try:
            self._model = self._build_model()
            self._mpc   = self._build_mpc()
            logging.info(
                f"{self.__class__.__name__} NMPC initialized "
                f"(N={self.N_HORIZON}, dt={self.DT_HR:.2f}hr, Ipopt)."
            )
        except Exception as exc:
            logging.error(f"{self.__class__.__name__} init failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Abstract: child classes must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self):
        """Return a fully configured do_mpc.model.Model (after model.setup())."""
        pass

    @abstractmethod
    def _build_mpc(self):
        """Return a fully configured do_mpc.controller.MPC (after mpc.setup())."""
        pass

    @abstractmethod
    def solve(self,
              initial_state:   Dict[str, Any],
              price_forecast:  List[float],
              demand_forecast: List[float]) -> Dict[str, Any]:
        """Receding-horizon solve: return the first-step optimal control action."""
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _prepare_solve(self,
                       price_forecast:  List[float],
                       demand_forecast: List[float]) -> None:
        """
        Store the current forecast window and reset do-mpc's internal clock
        to t=0. Must be called at the top of every solve() implementation.

        The TVP function is registered once in __init__ and references
        self._price_fcst / self._demand_fcst by closure, so updating them
        here is sufficient to feed the optimizer fresh forecast data.
        """
        n = self.N_HORIZON + 1       # TVP needs entries for steps 0 .. N
        def pad(lst):
            lst = list(lst)
            return (lst + [lst[-1]] * max(0, n - len(lst)))[:n]

        self._price_fcst  = pad(price_forecast)
        self._demand_fcst = pad(demand_forecast)
        self._mpc._t0     = np.array([0.0])   # Reset horizon clock

    def _timed_make_step(self, x0: np.ndarray) -> np.ndarray:
        """
        Calls mpc.make_step(x0), records wall-clock time, and returns u0.
        """
        t0 = time.perf_counter()
        u0 = self._mpc.make_step(x0)
        self.solve_times.append(time.perf_counter() - t0)
        return u0

    def solver_stats(self) -> Dict[str, float]:
        """Descriptive timing statistics over all recorded solve calls."""
        if not self.solve_times:
            return {'mean': 0.0, 'p95': 0.0, 'max': 0.0, 'n': 0}
        arr = np.array(self.solve_times)
        return {
            'mean': float(arr.mean()),
            'p95':  float(np.percentile(arr, 95)),
            'max':  float(arr.max()),
            'n':    len(arr),
        }

    @staticmethod
    def _mpc_params(n_horizon: int, dt_hr: float) -> dict:
        """Standard do-mpc MPC parameter dict."""
        return {
            'n_horizon':            n_horizon,
            't_step':               dt_hr,
            'state_discretization': 'discrete',
            'store_full_solution':  False,
            'nlpsol_opts':          _IPOPT_OPTS,
        }

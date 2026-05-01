"""
===============================================================================
Module:       ls_nmpc.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    NMPC controller for the Main Influent Lift Station (LS) wet well.

    Integer Relaxation:
    -------------------
    Ipopt is a continuous NLP solver; integer pump counts are relaxed to
    u_count ∈ [0, max_pumps] ⊂ ℝ and rounded to the nearest integer for
    dispatch.  This is standard practice for pump scheduling NMPCs
    (Kernan et al., 2017) and introduces at most one pump step of
    suboptimality per timestep.

    Level-Dependent Pump Efficiency with VFD Affinity Law:
    -------------------------------------------------------
    Consistent with the GWTP NMPC, the lift station is modeled as VFD-equipped.
    Pump power follows the affinity law (P ∝ n³), combined with the level-dependent
    TDH correction:
        u_norm  = u_count / max_pumps          (normalized speed [0, 1])
        P_eff   = P_rated_total * u_norm³ * (1 - alpha * (level - level_ref) / level_range)
    where P_rated_total = pump_kw * max_pumps.
    This replaces the prior linear-in-u_count formulation and ensures architectural
    consistency across all three NMPC facility models.

    State: ww_level [min_level, overflow_level]
    Control: u_count [0, max_pumps]  (continuous, rounded for dispatch)
    TVPs: price ($/kWh), influent_mgd (MGD)

Inputs:  ls_config.yaml dict, demand_penalty ($/kW)
Outputs: {'ls_count': [int_list]}
===============================================================================
"""

import numpy as np
import casadi as ca
import do_mpc

from .base_nmpc import BaseNMPC
from typing import Dict, Any, List


class LiftStation_NMPC(BaseNMPC):
    N_HORIZON  = 32
    DT_HR      = 0.25
    SAFETY_PEN = 5000.0
    ALPHA_HEAD = 0.08

    def __init__(self, config: Dict[str, Any], demand_penalty: float = 0.0):
        pumps   = config['pumps']['influent_pumps']
        storage = config['storage']

        self.pump_kw   = float(pumps['rated_kw'])
        self.pump_gpm  = float(pumps['design_flow_gpm'])
        self.max_pumps = int(pumps['count'])
        self.area      = float(storage['wet_well_area_sqft'])
        self.min_level = float(storage['min_level_ft'])
        self.max_level = float(storage['overflow_level_ft'])
        self.soft_ceil = 13.5
        self.lvl_ref   = (self.min_level + self.soft_ceil) / 2.0
        self.lvl_range = self.soft_ceil - self.min_level

        super().__init__(config, demand_penalty)

    def _build_model(self) -> do_mpc.model.Model:
        model = do_mpc.model.Model('discrete', symvar_type='MX')
        model.set_variable('_x', 'ww_level')
        model.set_variable('_u', 'u_count')
        model.set_variable('_tvp', 'price')
        model.set_variable('_tvp', 'influent_mgd')

        x   = model.x
        u   = model.u
        tvp = model.tvp

        ww_level = x['ww_level']
        u_count  = u['u_count']
        inf_mgd  = tvp['influent_mgd']

        dt_min     = self.DT_HR * 60.0
        gal_per_ft = self.area * 7.48
        inflow_gpm = (inf_mgd * 1_000_000.0) / 1440.0
        level_next = ww_level + ((inflow_gpm - u_count * self.pump_gpm) / gal_per_ft) * dt_min

        model.set_rhs('ww_level', level_next)
        model.setup()
        return model

    def _build_mpc(self) -> do_mpc.controller.MPC:
        mpc = do_mpc.controller.MPC(self._model)
        mpc.set_param(**self._mpc_params(self.N_HORIZON, self.DT_HR))

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            for k in range(self.N_HORIZON + 1):
                tvp_template['_tvp', k, 'price']        = self._price_fcst[k]
                tvp_template['_tvp', k, 'influent_mgd'] = self._demand_fcst[k]
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)

        x   = self._model.x
        u   = self._model.u
        tvp = self._model.tvp

        ww_level = x['ww_level']
        u_count  = u['u_count']
        price    = tvp['price']

        # VFD affinity law: normalize u_count to speed fraction, apply cubic
        # P_eff = P_rated_total * u_norm³ * (1 - alpha * lvl_frac)
        # Consistent with GWTP NMPC (P ∝ n³ for VFD-equipped pumps)
        lvl_frac    = (ww_level - self.lvl_ref) / self.lvl_range
        u_norm      = u_count / float(self.max_pumps)
        P_rated_tot = self.pump_kw * float(self.max_pumps)
        P_eff       = P_rated_tot * u_norm ** 3 * (1.0 - self.ALPHA_HEAD * lvl_frac)

        # Wet well target tracking — symmetric to GWTP GST approach but mirrored:
        # the cubic VFD law makes partial-speed operation cheap, so the optimizer
        # drains the wet well to the floor. Two complementary penalties fix this:
        #   1. One-sided quadratic: penalizes ww_level below 50% target each step
        #   2. Bilinear: suppresses u_count when well is already low
        #      (∂lterm/∂u_count = ww_deficit × 500/max_pumps → strong gradient
        #       toward u_count=0 when wet well is depleted)
        ww_target  = self.min_level + 0.50 * (self.soft_ceil - self.min_level)
        ww_range   = self.soft_ceil - self.min_level
        ww_dev     = (ww_level - ca.DM(ww_target)) / ca.DM(ww_range)
        ww_deficit = ca.fmax(0.0, ca.DM(ww_target) - ww_level) / ca.DM(ww_range)

        overflow   = ca.fmax(0.0, ww_level - self.soft_ceil)
        lterm      = (P_eff * self.DT_HR * price
                      + self.demand_penalty * P_rated_tot * u_norm ** 3
                      + self.SAFETY_PEN * overflow
                      + ww_deficit ** 2 * 200.0
                      + u_norm * ww_deficit * 500.0)

        mpc.set_objective(lterm=lterm, mterm=ww_dev ** 2 * 2000.0)
        mpc.set_rterm(u_count=1e-3)

        mpc.bounds['lower', '_u', 'u_count']  = 0.0
        mpc.bounds['upper', '_u', 'u_count']  = float(self.max_pumps)
        mpc.bounds['lower', '_x', 'ww_level'] = self.min_level
        mpc.bounds['upper', '_x', 'ww_level'] = self.max_level

        mpc.setup()
        return mpc

    def solve(self,
              initial_state:   Dict[str, Any],
              price_forecast:  List[float],
              demand_forecast: List[float]) -> Dict[str, Any]:
        HORIZON = min(len(price_forecast), self.N_HORIZON)
        self._prepare_solve(price_forecast[:HORIZON], demand_forecast[:HORIZON])

        ww_0 = float(np.clip(initial_state['wet_well_ft'], self.min_level, self.max_level))
        x0   = np.array([[ww_0]])
        self._mpc.x0 = x0
        self._mpc.set_initial_guess()
        u0 = self._timed_make_step(x0)

        u_int = int(round(float(np.clip(u0[0, 0], 0.0, self.max_pumps))))
        N = len(price_forecast)
        return {'ls_count': [u_int] * N}

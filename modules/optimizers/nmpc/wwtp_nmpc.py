"""
===============================================================================
Module:       wwtp_nmpc.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    NMPC controller for the WWTP aeration basin Dissolved Oxygen (DO) system.

    Key Nonlinear Physics (vs. MILP linearization):
    -----------------------------------------------
    DO dynamics: dDO/dt = KLa(t) * (DO_sat - DO(t)) - OUR(t)

    The gain term KLa * (DO_sat - DO(t)) is bilinear in control and state.
    As DO rises toward saturation, the oxygen transfer driving force naturally
    decreases, allowing the optimizer to reduce blower duty without risking
    DO deficit — an efficiency gain unavailable to the MILP.

    MILP: fixes deficit at 2.5 mg/L → gain = KLa_max * u * 5.7  [constant]
    NMPC: exact deficit = DO_sat - DO(t)                          [nonlinear]

    The blower is a constant-speed rotary lobe unit (no VFD), so
    P = P_rated * u_blower is linear. The nonlinearity is exclusively in DO.

    State: do_mg_l [0.1, do_sat]  |  Control: u_blower [0, 1]
    TVPs:  price ($/kWh), influent_mgd (MGD)

Inputs:  wwtp_config.yaml dict, demand_penalty ($/kW)
Outputs: {'blower_duty': [list], 'predicted_do': [list]}
===============================================================================
"""

import numpy as np
import casadi as ca
import do_mpc

from .base_nmpc import BaseNMPC
from typing import Dict, Any, List


class WWTP_NMPC(BaseNMPC):
    N_HORIZON    = 96
    DT_HR        = 0.25
    DO_FLOOR_PEN = 1000.0

    def __init__(self, config: Dict[str, Any], demand_penalty: float = 0.0):
        bio    = config['biology']
        blower = config['blowers']['aeration_blower']

        self.blower_kw = float(blower['rated_kw'])
        self.do_sat    = float(bio['do_saturation_limit'])
        self.kla_max   = float(bio['kla_max'])
        self.our_base  = float(bio['our_base'])
        self.do_min    = float(bio.get('do_min_mg_l', 1.5))

        super().__init__(config, demand_penalty)

    def _build_model(self) -> do_mpc.model.Model:
        model = do_mpc.model.Model('discrete', symvar_type='MX')
        model.set_variable('_x', 'do_mg_l')
        model.set_variable('_u', 'u_blower')
        model.set_variable('_tvp', 'price')
        model.set_variable('_tvp', 'influent_mgd')

        x   = model.x
        u   = model.u
        tvp = model.tvp

        do_mg_l  = x['do_mg_l']
        u_blower = u['u_blower']
        inf_mgd  = tvp['influent_mgd']

        # Exact nonlinear DO dynamics (bilinear in u_blower and do_mg_l)
        load_factor   = inf_mgd / 1.5
        our           = self.our_base * load_factor
        aeration_gain = u_blower * self.kla_max * (self.do_sat - do_mg_l)
        do_next       = do_mg_l + (aeration_gain - our) * self.DT_HR

        model.set_rhs('do_mg_l', do_next)
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

        do_mg_l  = x['do_mg_l']
        u_blower = u['u_blower']
        price    = tvp['price']

        P_blower    = self.blower_kw * u_blower
        energy_cost = P_blower * self.DT_HR * price
        demand_prox = self.demand_penalty * P_blower
        do_deficit  = ca.fmax(0.0, self.do_min - do_mg_l)
        safety_pen  = self.DO_FLOOR_PEN * do_deficit

        mpc.set_objective(lterm=energy_cost + demand_prox + safety_pen, mterm=ca.DM(0))
        mpc.set_rterm(u_blower=1e-4)

        mpc.bounds['lower', '_u', 'u_blower'] = 0.0
        mpc.bounds['upper', '_u', 'u_blower'] = 1.0
        # Hard lower bound at regulatory floor (1.5 mg/L) rather than 0.1.
        # This makes DO compliance a hard Ipopt constraint, not just a soft penalty,
        # preventing the receding horizon from allowing sustained sub-floor operation.
        mpc.bounds['lower', '_x', 'do_mg_l']  = self.do_min
        mpc.bounds['upper', '_x', 'do_mg_l']  = self.do_sat

        mpc.setup()
        return mpc

    def solve(self,
              initial_state:   Dict[str, Any],
              price_forecast:  List[float],
              demand_forecast: List[float]) -> Dict[str, Any]:
        self._prepare_solve(price_forecast, demand_forecast)

        do_0 = float(np.clip(initial_state['do_mg_l'], 0.1, self.do_sat))
        x0   = np.array([[do_0]])
        self._mpc.x0 = x0
        self._mpc.set_initial_guess()
        u0 = self._timed_make_step(x0)

        u_blower = float(np.clip(u0[0, 0], 0.0, 1.0))

        # Propagate DO trajectory for diagnostic output
        N = len(price_forecast)
        do_traj, do_cur = [do_0], do_0
        for t in range(N):
            inf_mgd = self._demand_fcst[t]
            our     = self.our_base * (inf_mgd / 1.5)
            gain    = u_blower * self.kla_max * (self.do_sat - do_cur)
            do_cur  = float(np.clip(do_cur + (gain - our) * self.DT_HR, 0.1, self.do_sat))
            do_traj.append(do_cur)

        return {
            'blower_duty':  [u_blower] * N,
            'predicted_do': do_traj[:N],
        }

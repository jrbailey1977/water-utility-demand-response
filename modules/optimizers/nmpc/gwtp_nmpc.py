"""
===============================================================================
Module:       gwtp_nmpc.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    NMPC controller for the two-stage Groundwater Treatment Plant (GWTP).

    Key Nonlinear Physics (vs. MILP linearization):
    -----------------------------------------------
    MILP models pump power as P = P_rated * u (linear in fractional runtime).
    This module models pumps with Variable Frequency Drives (VFDs) using the
    centrifugal pump affinity laws:
        Law 1 (Flow):   Q(u) = Q_rated * u           [same as MILP]
        Law 2 (Head):   H(u) = H_rated * u^2
        Law 3 (Power):  P(u) = P_rated * u^3         [NONLINEAR — key difference]

    The cubic power-speed relationship means a pump at 50% speed delivers
    50% of rated flow at only 12.5% of rated power (vs. 50% under the MILP
    linear model), opening a far wider range of partial-load operating points.

    State Variables:
        v_gst   (ft^3): GST volume
        v_hydro (ft^3): Hydropneumatic tank volume

    Control Variables:
        u_well  [0, 1]: Well pump speed ratio (VFD)
        u_boost [0, 1]: Aggregate booster speed ratio (VFD, 3-pump bank)

    TVPs:
        price       ($/kWh): ERCOT day-ahead price
        demand_fhr  (ft^3/hr): Community water demand

    Stage cost:
        lterm = (P_well * u_well^3 + P_boost * u_boost^3) * dt * price
              + demand_penalty * (P_well * u_well^3 + P_boost * u_boost^3)

    Terminal cost (mterm): penalizes ending GST volume below initial value.

Inputs:  gwtp_config.yaml dict, demand_penalty ($/kW)
Outputs: {'well_pump': [list], 'booster_pumps': {1:[], 2:[], 3:[]}}
===============================================================================
"""

import math
import numpy as np
import casadi as ca
import do_mpc

from .base_nmpc import BaseNMPC
from typing import Dict, Any, List


class GWTP_NMPC(BaseNMPC):
    """NMPC controller for the GWTP with VFD centrifugal pump affinity laws."""

    N_HORIZON = 96
    DT_HR     = 0.25

    def __init__(self, config: Dict[str, Any], demand_penalty: float = 0.0):
        gst_cfg   = config['storage']['gst']
        hydro_cfg = config['storage']['hydro_tank']
        well_cfg  = config['pumps']['well_pump']
        boost_cfg = config['pumps']['booster_pumps']

        self.A_gst    = math.pi * (gst_cfg['diameter_ft'] / 2) ** 2
        self.A_hydro  = math.pi * (hydro_cfg['effective_diameter_ft'] / 2) ** 2

        self.v_gst_min   = gst_cfg['min_level_ft']   * self.A_gst
        self.v_gst_max   = gst_cfg['max_level_ft']   * self.A_gst
        self.v_hydro_min = hydro_cfg['min_level_ft'] * self.A_hydro
        self.v_hydro_max = hydro_cfg['max_level_ft'] * self.A_hydro

        # Rated flows: GPM → ft^3/hr
        self.Q_well_rated  = well_cfg['design_flow_gpm']  * 60.0 / 7.48
        self.Q_boost_rated = boost_cfg['design_flow_gpm'] * 60.0 / 7.48 * 3

        # Rated electrical power (kW), shaft HP / motor efficiency
        well_eff  = well_cfg.get('motor_efficiency', 0.93)
        boost_eff = boost_cfg.get('motor_efficiency', 0.917)
        self.P_well_rated  = well_cfg['motor_hp']  * 0.7457 / well_eff
        self.P_boost_rated = boost_cfg['motor_hp'] * 0.7457 / boost_eff * 3

        # Initial GST volume used for terminal cost reference (updated per solve)
        self._v_gst_init_val = (self.v_gst_min + self.v_gst_max) * 0.5

        super().__init__(config, demand_penalty)

    # ------------------------------------------------------------------

    def _build_model(self) -> do_mpc.model.Model:
        model = do_mpc.model.Model('discrete', symvar_type='MX')

        model.set_variable('_x', 'v_gst')
        model.set_variable('_x', 'v_hydro')
        model.set_variable('_u', 'u_well')
        model.set_variable('_u', 'u_boost')
        model.set_variable('_tvp', 'price')
        model.set_variable('_tvp', 'demand_fhr')

        # Use the CasADi MX symbols returned by set_variable directly
        x     = model.x
        u     = model.u
        tvp   = model.tvp

        v_gst      = x['v_gst']
        v_hydro    = x['v_hydro']
        u_well     = u['u_well']
        u_boost    = u['u_boost']
        demand_fhr = tvp['demand_fhr']

        # Mass balance — flow is linear in speed (affinity law 1)
        Q_in     = self.Q_well_rated  * u_well   # ft^3/hr
        Q_boost  = self.Q_boost_rated * u_boost  # ft^3/hr
        dem_vol  = demand_fhr * self.DT_HR       # ft^3 per step

        model.set_rhs('v_gst',   v_gst   + (Q_in   - Q_boost) * self.DT_HR)
        model.set_rhs('v_hydro', v_hydro + (Q_boost * self.DT_HR - dem_vol))

        model.setup()
        return model

    # ------------------------------------------------------------------

    def _build_mpc(self) -> do_mpc.controller.MPC:
        mpc = do_mpc.controller.MPC(self._model)
        mpc.set_param(**self._mpc_params(self.N_HORIZON, self.DT_HR))

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            for k in range(self.N_HORIZON + 1):
                tvp_template['_tvp', k, 'price']      = self._price_fcst[k]
                tvp_template['_tvp', k, 'demand_fhr'] = self._demand_fcst[k]
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)

        x   = self._model.x
        u   = self._model.u
        tvp = self._model.tvp

        u_well  = u['u_well']
        u_boost = u['u_boost']
        price   = tvp['price']
        v_gst   = x['v_gst']

        # ---- Nonlinear stage cost: affinity law — power ∝ speed^3 ----
        P_well_kw  = self.P_well_rated  * u_well  ** 3
        P_boost_kw = self.P_boost_rated * u_boost ** 3
        P_total    = P_well_kw + P_boost_kw

        lterm = P_total * self.DT_HR * price + self.demand_penalty * P_total

        # Stage cost: GST tracking — two complementary penalties:
        #   1. Quadratic state penalty: discourages GST from deviating above target
        #   2. Bilinear control penalty: directly suppresses u_well when GST > target
        #      (∂lterm/∂u_well = gst_excess × 1000 → strong gradient toward u_well=0
        #       when tank is high, regardless of Ipopt warm-start anchoring)
        v_gst_target = self.v_gst_min + 0.70 * (self.v_gst_max - self.v_gst_min)
        v_gst_range  = self.v_gst_max - self.v_gst_min
        gst_dev      = (v_gst - ca.DM(v_gst_target)) / ca.DM(v_gst_range)
        gst_excess   = ca.fmax(0.0, v_gst - ca.DM(v_gst_target)) / ca.DM(v_gst_range)
        lterm        = lterm + gst_excess ** 2 * 500.0 + u_well * gst_excess * 1000.0

        # Terminal cost: same quadratic tracking, higher weight to ensure end-of-horizon
        # GST is restored toward target level for the next receding-horizon solve.
        mterm = gst_dev ** 2 * 2000.0

        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(u_well=1e-3, u_boost=1e-3)

        mpc.bounds['lower', '_u', 'u_well']  = 0.0
        mpc.bounds['upper', '_u', 'u_well']  = 1.0
        mpc.bounds['lower', '_u', 'u_boost'] = 0.0
        mpc.bounds['upper', '_u', 'u_boost'] = 1.0
        mpc.bounds['lower', '_x', 'v_gst']   = self.v_gst_min
        mpc.bounds['upper', '_x', 'v_gst']   = self.v_gst_max
        mpc.bounds['lower', '_x', 'v_hydro'] = self.v_hydro_min
        mpc.bounds['upper', '_x', 'v_hydro'] = self.v_hydro_max

        mpc.setup()
        return mpc

    # ------------------------------------------------------------------

    def solve(self,
              initial_state:   Dict[str, Any],
              price_forecast:  List[float],
              demand_forecast: List[float]) -> Dict[str, Any]:
        """
        Receding-horizon NMPC solve from current GWTP state.

        Args:
            initial_state:   {'gst_level_ft', 'hydro_level_ft'}
            price_forecast:  $/kWh for N steps
            demand_forecast: ft^3/hr demand for N steps

        Returns:
            main.py-compatible dict with well_pump and booster_pumps lists
        """
        # Update terminal cost reference to current GST fill level
        self._v_gst_init_val = initial_state['gst_level_ft'] * self.A_gst

        self._prepare_solve(price_forecast, demand_forecast)

        v_gst_0   = initial_state['gst_level_ft']   * self.A_gst
        v_hydro_0 = initial_state['hydro_level_ft'] * self.A_hydro
        x0 = np.array([[v_gst_0], [v_hydro_0]])

        self._mpc.x0 = x0
        self._mpc.set_initial_guess()
        u0 = self._timed_make_step(x0)

        u_well  = float(np.clip(u0[0, 0], 0.0, 1.0))
        u_boost = float(np.clip(u0[1, 0], 0.0, 1.0))

        # Split aggregate booster into 3 equal pumps for main.py interface
        booster_each = u_boost / 3.0
        N = len(price_forecast)
        return {
            'well_pump': [u_well] * N,
            'booster_pumps': {
                1: [booster_each] * N,
                2: [booster_each] * N,
                3: [booster_each] * N,
            }
        }

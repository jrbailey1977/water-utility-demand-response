"""
===============================================================================
Module:       wwtp_milp.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    MILP scheduler for the WWTP aeration basin. Optimizes blower energy
    usage against a 24-hour ERCOT day-ahead price forecast while maintaining
    dissolved oxygen (DO) within regulatory and process limits.
    Inherits from BaseMILP for solver management and timing infrastructure.

    Key Design Decisions:
        1. Linearized DO Mass Balance: The exact DO gain term is bilinear in
           blower duty and DO level (KLa * u * (DO_sat - DO)), which GLPK
           cannot solve. I fix the oxygen deficit at a nominal operating point
           of 2.5 mg/L (deficit = DO_sat - 2.5 = 5.7 mg/L at 20 deg C) to
           linearize the gain term. This approximates realistic mass transfer
           across the full horizon rather than overstating early gains when
           the basin is well below saturation.
        2. Soft Safety Floor: Slack variables absorb DO deficit below 1.5 mg/L
           at a $1,000/unit penalty. The soft form maintains GLPK feasibility
           during sustained high-price periods without hard constraint violation.
        3. Efficiency Cap: DO is bounded at 4.0 mg/L. Operating above this
           level reduces KLa mass transfer efficiency (smaller deficit term)
           without process benefit, so the optimizer naturally avoids it.
        4. Terminal Constraint: Commented out by design. A hard terminal DO
           target biases the optimizer toward end-of-horizon aeration regardless
           of price, undermining demand response. The soft safety floor provides
           sufficient process protection without this bias.

Inputs:
    - initial_state:          {'do_mg_l'}
    - price_forecast:         $/kWh for 96 steps (24 hr at 15-min resolution)
    - influent_mgd_forecast:  Influent flow (MGD) for 96 steps
Outputs:
    - Control dict: {'blower_duty': [list], 'predicted_do': [list]}
Dependencies:
    - pyomo, glpk solver, .base_milp
===============================================================================
"""

import time
import pyomo.environ as aml
from modules.optimizers.milp.base_milp import BaseMILP


class WWTP_MILP(BaseMILP):

    def __init__(self, config, demand_penalty=0.0):
        """
        Initialize the WWTP MILP optimizer from wwtp_config.yaml.

        Args:
            config:         Parsed wwtp_config.yaml dictionary.
            demand_penalty: Peak demand charge proxy ($/kW) from main.py.
        """
        super().__init__(config)
        self.demand_penalty  = demand_penalty

        self.blower_kw       = config['blowers']['aeration_blower']['rated_kw']
        self.do_sat          = config['biology']['do_saturation_limit']
        self.kla_max         = config['biology']['kla_max']
        self.our_base        = config['biology']['our_base']

        self.do_floor_penalty   = 1000.0   # $/unit slack -- high enough to strongly discourage sub-floor DO
        self.terminal_do_target = 2.0      # mg/L -- retained for reference if terminal constraint is re-enabled

    def solve(self, initial_state, price_forecast, influent_mgd_forecast):
        """
        Solve the WWTP MILP over a 24-hour receding horizon.

        Builds and solves a Pyomo ConcreteModel with a linearized DO mass
        balance, soft safety floor, efficiency cap, and peak demand variable.
        Only the first-step blower duty is dispatched.

        Args:
            initial_state:          Dict with 'do_mg_l'.
            price_forecast:         $/kWh for each horizon step.
            influent_mgd_forecast:  Influent flow (MGD) for each horizon step.

        Returns:
            Dict with keys:
                'blower_duty'  -- continuous duty fraction [0, 1] per step
                'predicted_do' -- linearized DO trajectory (mg/L) per step
        """
        steps = len(price_forecast)
        dt    = 0.25  # 15-minute intervals in hours
        model = self._setup_base_model(steps)

        model.blower_duty = aml.Var(model.T, bounds=(0, 1))
        model.do_level    = aml.Var(model.T, bounds=(0.1, 4.0))  # 4.0 mg/L efficiency cap
        model.do_slack    = aml.Var(model.T, bounds=(0, None))
        model.peak_kw     = aml.Var(bounds=(0, None))

        def obj_rule(m):
            energy_cost    = sum(m.blower_duty[t] * self.blower_kw * dt * price_forecast[t] for t in m.T)
            safety_penalty = sum(m.do_slack[t] * self.do_floor_penalty for t in m.T)
            demand_charge  = m.peak_kw * self.demand_penalty
            return energy_cost + safety_penalty + demand_charge
        model.obj = aml.Objective(rule=obj_rule, sense=aml.minimize)

        def do_dynamics_rule(m, t):
            load_factor   = influent_mgd_forecast[t] / 1.5
            our           = self.our_base * load_factor
            prev_val      = float(initial_state['do_mg_l']) if t == 0 else m.do_level[t - 1]

            # Linearized gain: deficit fixed at nominal operating point (2.5 mg/L).
            # Exact form is KLa * u * (DO_sat - DO(t)), which is bilinear and not
            # solvable by GLPK. Fixing the deficit at 5.7 mg/L (DO_sat - 2.5) gives
            # a representative transfer rate across the full horizon. The NMPC tier
            # uses the exact bilinear form for higher-fidelity scheduling.
            nominal_deficit = float(self.do_sat - 2.5)
            gain            = m.blower_duty[t] * self.kla_max * nominal_deficit
            return m.do_level[t] == prev_val + (gain - our) * dt
        model.con_do = aml.Constraint(model.T, rule=do_dynamics_rule)

        def safety_floor_rule(m, t):
            return m.do_level[t] + m.do_slack[t] >= 1.5
        model.con_safety = aml.Constraint(model.T, rule=safety_floor_rule)

        def peak_rule(m, t):
            return m.peak_kw >= m.blower_duty[t] * self.blower_kw
        model.con_peak = aml.Constraint(model.T, rule=peak_rule)

        # Terminal DO constraint is intentionally disabled. A hard end-of-horizon
        # target biases the optimizer to aerate regardless of price at the last
        # few steps, introducing a systematic cost floor that undermines demand
        # response. The soft safety floor (con_safety) provides adequate process
        # protection without this directional bias.
        # def terminal_do_rule(m):
        #     return m.do_level[steps - 1] >= self.terminal_do_target
        # model.con_terminal = aml.Constraint(rule=terminal_do_rule)

        _t0 = time.perf_counter()
        self.solver.solve(model)
        self.solve_times.append(time.perf_counter() - _t0)

        return {
            'blower_duty':  [aml.value(model.blower_duty[t]) for t in model.T],
            'predicted_do': [aml.value(model.do_level[t]) for t in model.T],
        }

"""
===============================================================================
Module:       gwtp_milp.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    MILP scheduler for the two-zone Groundwater Treatment Plant (GWTP).
    Optimizes both the Ground Storage Tank (GST) and Hydropneumatic Tank
    (Hydro) pressure zones simultaneously over a 24-hour horizon.
    Inherits from BaseMILP for solver management and timing infrastructure.

    Key Design Decisions:
        1. Dynamic Configuration: All physical constants (HP, area, flow) are
           read from gwtp_config.yaml so the optimizer stays consistent with
           the Digital Twin physics in gwtp.py.
        2. Price-Dominant Control: Storage incentive weights are set to 0.0 so
           ERCOT day-ahead prices are the sole driver of pumping decisions.
           Non-zero weights caused sawtooth oscillations where the optimizer
           filled tanks to avoid penalty regardless of price signals.
        3. Soft Terminal Constraint: A slack variable with a $1,000/unit penalty
           ensures the horizon ends with at least as much GST storage as it
           started. The soft form (vs. a hard equality) maintains feasibility
           during periods of sustained high prices.
        4. Symmetry Breaking: A priority constraint (booster_1 >= booster_2 >=
           booster_3) eliminates equivalent permutations and reduces the MILP
           search space.

Inputs:
    - initial_state:   {'gst_level_ft', 'hydro_level_ft'}
    - price_forecast:  $/kWh for 96 steps (24 hr at 15-min resolution)
    - demand_forecast: ft^3/hr community water demand for 96 steps
Outputs:
    - Control dict: {'well_pump': [list], 'booster_pumps': {1:[], 2:[], 3:[]}}
Dependencies:
    - pyomo, glpk solver, .base_milp
===============================================================================
"""

import time
import math
import pyomo.environ as aml
from modules.optimizers.milp.base_milp import BaseMILP


class GWTP_MILP(BaseMILP):

    def __init__(self, config, demand_penalty=0.0):
        """
        Initialize the GWTP MILP optimizer from gwtp_config.yaml.

        Precomputes tank areas and volume bounds so the solve() method avoids
        redundant geometry calculations at each of the 672 receding-horizon calls.

        Args:
            config:         Parsed gwtp_config.yaml dictionary.
            demand_penalty: Peak demand charge proxy ($/kW) passed from main.py.
                            Propagated into the MILP objective to penalize
                            instantaneous power peaks within the horizon.
        """
        super().__init__(config)
        self.demand_penalty = demand_penalty

        gst_cfg   = config['storage']['gst']
        hydro_cfg = config['storage']['hydro_tank']

        self.gst_area   = math.pi * (gst_cfg['diameter_ft'] / 2) ** 2
        self.hydro_area = math.pi * (hydro_cfg['effective_diameter_ft'] / 2) ** 2

        self.gst_min_vol   = gst_cfg['min_level_ft']   * self.gst_area
        self.gst_max_vol   = gst_cfg['max_level_ft']   * self.gst_area
        self.hydro_min_vol = hydro_cfg['min_level_ft'] * self.hydro_area
        self.hydro_max_vol = hydro_cfg['max_level_ft'] * self.hydro_area

        well_cfg    = config['pumps']['well_pump']
        booster_cfg = config['pumps']['booster_pumps']

        # Convert shaft HP to electrical kW (1 HP = 0.7457 kW) for the
        # MILP objective. Motor efficiency is not applied here because the
        # MILP uses linear power -- efficiency correction is applied in
        # main.py's financial audit, which uses nameplate data directly.
        self.well_kw    = well_cfg['motor_hp']    * 0.7457
        self.booster_kw = booster_cfg['motor_hp'] * 0.7457

        self.well_flow    = well_cfg['design_flow_gpm']
        self.booster_flow = booster_cfg['design_flow_gpm']

        # Storage incentive weights are zeroed intentionally. Non-zero weights
        # drove sawtooth behavior where the optimizer would fill tanks to collect
        # incentive rewards regardless of price signals, defeating the demand
        # response objective. ERCOT prices alone provide sufficient gradient.
        self.gst_incentive   = 0.0
        self.hydro_incentive = 0.0

    def solve(self, initial_state, price_forecast, demand_forecast):
        """
        Solve the GWTP MILP over a 24-hour receding horizon.

        Builds and solves a Pyomo ConcreteModel with GST and Hydro Tank mass
        balance constraints, booster symmetry breaking, a peak demand variable,
        and a soft terminal storage constraint. Only the first-step control
        actions are dispatched; the remainder of the horizon is discarded
        (receding-horizon principle).

        Args:
            initial_state:   Dict with 'gst_level_ft' and 'hydro_level_ft'.
            price_forecast:  $/kWh for each of the 96 horizon steps.
            demand_forecast: ft^3/hr community demand for each horizon step.

        Returns:
            Dict with keys:
                'well_pump'     -- list of well pump duty [0, 1] per step
                'booster_pumps' -- dict {1: [], 2: [], 3: []} per step
        """
        steps = len(price_forecast)
        dt    = 0.25  # 15-minute intervals in hours
        model = self._setup_base_model(steps)
        model.P = aml.RangeSet(1, 3)

        model.well          = aml.Var(model.T, bounds=(0, 1))
        model.boosters      = aml.Var(model.P, model.T, bounds=(0, 1))
        model.v_gst         = aml.Var(model.T, bounds=(self.gst_min_vol,   self.gst_max_vol))
        model.v_hydro       = aml.Var(model.T, bounds=(self.hydro_min_vol, self.hydro_max_vol))
        model.terminal_slack = aml.Var(bounds=(0, None))
        model.peak_kw       = aml.Var(bounds=(0, None))

        def obj_rule(m):
            energy_cost      = sum(
                (m.well[t] * self.well_kw +
                 sum(m.boosters[p, t] for p in m.P) * self.booster_kw)
                * dt * price_forecast[t]
                for t in m.T
            )
            terminal_penalty = m.terminal_slack * 1000.0
            demand_charge    = m.peak_kw * self.demand_penalty
            return energy_cost + terminal_penalty + demand_charge
        model.obj = aml.Objective(rule=obj_rule, sense=aml.minimize)

        def mass_balance_gst(m, t):
            prev_vol = initial_state['gst_level_ft'] * self.gst_area if t == 0 else m.v_gst[t - 1]
            inflow   = m.well[t] * self.well_flow * 60 / 7.48
            outflow  = sum(m.boosters[p, t] for p in m.P) * self.booster_flow * 60 / 7.48
            return m.v_gst[t] == prev_vol + (inflow - outflow) * dt
        model.con_gst = aml.Constraint(model.T, rule=mass_balance_gst)

        def mass_balance_hydro(m, t):
            prev_vol = initial_state['hydro_level_ft'] * self.hydro_area if t == 0 else m.v_hydro[t - 1]
            inflow   = sum(m.boosters[p, t] for p in m.P) * self.booster_flow * 60 / 7.48
            outflow  = demand_forecast[t]
            return m.v_hydro[t] == prev_vol + (inflow - outflow) * dt
        model.con_hydro = aml.Constraint(model.T, rule=mass_balance_hydro)

        # Symmetry breaking: enforce pump priority ordering to reduce search space.
        def priority_rule(m, p, t):
            if p < 3:
                return m.boosters[p, t] >= m.boosters[p + 1, t]
            return aml.Constraint.Skip
        model.con_priority = aml.Constraint(model.P, model.T, rule=priority_rule)

        def peak_rule(m, t):
            power_t = (m.well[t] * self.well_kw +
                       sum(m.boosters[p, t] for p in m.P) * self.booster_kw)
            return m.peak_kw >= power_t
        model.con_peak = aml.Constraint(model.T, rule=peak_rule)

        # Soft terminal constraint: penalize ending below initial GST fill.
        # Soft form prevents infeasibility during high-price periods where
        # meeting the terminal target would require pumping at any price.
        def terminal_gst_rule(m):
            target = initial_state['gst_level_ft'] * self.gst_area
            return m.v_gst[steps - 1] + m.terminal_slack >= target
        model.con_terminal_gst = aml.Constraint(rule=terminal_gst_rule)

        _t0 = time.perf_counter()
        self.solver.solve(model)
        self.solve_times.append(time.perf_counter() - _t0)

        return {
            'well_pump':     [aml.value(model.well[t]) for t in model.T],
            'booster_pumps': {p: [aml.value(model.boosters[p, t]) for t in model.T] for p in model.P},
        }

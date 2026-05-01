"""
===============================================================================
Module:       ls_milp.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:  
    Hydraulic MILP Scheduler for the Influent Lift Station.
    Manages wet well level as a hydraulic buffer to avoid peak-price pumping.
    Inherits from BaseMILP to leverage standardized solver orchestration.

    Key Logic:
        1. Dynamic Configuration: All physical constants (kW, Area, Flow) are 
           sourced from ls_config.yaml to maintain Digital Twin fidelity.
        2. Performance Tuning: Truncated 8-hour horizon (32 steps) ensures 
           execution speed within high-frequency simulation loops.
        3. Soft Safety Constraints: High-penalty slack variables prevent 
           solver infeasibility while allowing use of the 13.5ft surcharge zone.
        4. Integer Pump Sequencing: Manages 3 discrete submersible pumps.

Inputs:       
    - Current State (Wet Well Level ft)
    - 8-Hour Price Forecast ($/kWh)
    - 8-Hour Influent Flow Forecast (MGD)
Outputs:      
    - Control Dict: {'ls_count': [list]}
Dependencies: 
    - pyomo, glpk solver, .base_milp
===============================================================================
"""

import time
import pyomo.environ as pyo
from modules.optimizers.milp.base_milp import BaseMILP

class LiftStationMILP(BaseMILP):
    def __init__(self, config, demand_penalty=0.0):
        """Standardized initialization via BaseMILP."""
        super().__init__(config)
        self.demand_penalty = demand_penalty
        
        # Sourced from config to ensure Digital Twin fidelity
        pumps = config['pumps']['influent_pumps']
        storage = config['storage']
        
        self.pump_kw = pumps['rated_kw']
        self.pump_gpm = pumps['design_flow_gpm']
        self.max_pumps = pumps['count']
        
        self.area = storage['wet_well_area_sqft']
        self.max_physical_level = storage['overflow_level_ft'] # 14.0ft

    def solve(self, initial_state, price_forecast, influent_mgd_forecast):
        """
        Solves hydraulic mass balance with a truncated 8-hour lookahead 
        to maintain simulation throughput.
        """
        HORIZON = min(len(price_forecast), 32) 
        prices = price_forecast[:HORIZON]
        flows = influent_mgd_forecast[:HORIZON]
        
        dt_min = 15.0 
        gal_per_ft = self.area * 7.48
        model = self._setup_base_model(HORIZON)

        # Variables
        model.pumps_on = pyo.Var(model.T, domain=pyo.Integers, bounds=(0, self.max_pumps))
        model.level = pyo.Var(model.T, bounds=(2.0, self.max_physical_level)) 
        model.ls_slack = pyo.Var(model.T, bounds=(0, None))
        model.peak_kw = pyo.Var(bounds=(0, None)) 

        # Objective: Minimize Energy Cost + Penalty
        def obj_rule(m):
            energy_cost = sum(m.pumps_on[t] * self.pump_kw * (dt_min/60) * prices[t] for t in m.T)
            safety_penalty = sum(m.ls_slack[t] * 5000.0 for t in m.T)
            demand_charge = m.peak_kw * self.demand_penalty
            return energy_cost + safety_penalty + demand_charge
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Mass Balance
        def mass_rule(m, t):
            prev = initial_state['wet_well_ft'] if t == 0 else m.level[t-1]
            inflow = (flows[t] * 1000000.0) / 1440.0
            outflow = m.pumps_on[t] * self.pump_gpm
            return m.level[t] == prev + ((inflow - outflow) / gal_per_ft) * dt_min
        model.con_mass = pyo.Constraint(model.T, rule=mass_rule)

        # Safety Ceiling: Set to 13.5ft to exploit the surcharge "Sewer Battery"
        def ceiling_rule(m, t):
            return m.level[t] <= 13.5 + m.ls_slack[t]
        model.con_ceiling = pyo.Constraint(model.T, rule=ceiling_rule)

        # Peak power constraint
        def peak_rule(m, t):
            return m.peak_kw >= m.pumps_on[t] * self.pump_kw
        model.con_peak = pyo.Constraint(model.T, rule=peak_rule)

        # Explicit Solver Timeout (in seconds)
        self.solver.options['tmlim'] = 5  # LS gets a tighter timeout - 8hr horizon is simpler
        _t0 = time.perf_counter()
        self.solver.solve(model, load_solutions=True)
        self.solve_times.append(time.perf_counter() - _t0)
        
        return {
            'ls_count': [int(pyo.value(model.pumps_on[t])) for t in model.T]
        }
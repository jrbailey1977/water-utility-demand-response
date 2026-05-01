"""
===============================================================================
Module:       lift_station.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Physics engine for the Main Influent Lift Station (LS).
    Models the wet well as the "pipeline battery" - the third storage
    buffer in the Triple-Battery framework alongside elevation storage
    (GWTP) and biological oxygen (WWTP).

    Wet well level evolves via mass balance:
        dV/dt = Q_in - Q_out
    where Q_in is influent flow and Q_out is the total pump discharge.

Inputs:
    - config: Dictionary from ls_config.yaml
    - dt_min: Simulation timestep in minutes
    - lift_pump_count: Control signal (pumps running, [0 - max_pumps])
    - influent_mgd: Influent flow rate (Million Gallons per Day)
Outputs:
    - State Dictionary: {'wet_well_ft'}
===============================================================================
"""


class LiftStation:
    """
    Digital Twin of the Main Influent Lift Station wet well.

    Tracks wet well level via first-order mass balance (Euler integration).
    Asset specifications are read from ls_config.yaml.
    """

    def __init__(self, config):
        """
        Initialize the LiftStation physics engine from ls_config.yaml.
        """
        self.config = config

        ls_pumps = config['pumps']['influent_pumps']
        self.ls_pump_kw = ls_pumps['rated_kw']
        self.ls_pump_gpm = ls_pumps['design_flow_gpm']

        storage = config['storage']
        self.wet_well_area = storage['wet_well_area_sqft']
        self.min_level = storage['min_level_ft']
        self.max_level = storage['max_level_ft']
        self.wet_well_level_ft = storage['initial_level_ft']

    def update(self, dt_min, lift_pump_count, influent_mgd):
        """
        Advance wet well level by dt_min minutes via mass balance.
        """
        inflow_gpm = (influent_mgd * 1_000_000) / 1440.0
        outflow_gpm = lift_pump_count * self.ls_pump_gpm
        net_gpm = inflow_gpm - outflow_gpm

        gal_per_ft = self.wet_well_area * 7.48
        self.wet_well_level_ft += (net_gpm / gal_per_ft) * dt_min
        self.wet_well_level_ft = max(self.min_level,
                                     min(self.wet_well_level_ft, self.max_level))

    def get_system_state(self):
        return {
            'wet_well_ft': round(self.wet_well_level_ft, 2),
            'ls_pump_kw_active': self.ls_pump_kw
        }
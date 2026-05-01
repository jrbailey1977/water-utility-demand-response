"""
===============================================================================
Module:       gwtp.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Physics engine (Digital Twin) for the Groundwater Treatment Plant (GWTP).
    Models the elevation storage battery -- the first storage buffer in the
    Triple-Battery framework alongside biological oxygen (WWTP) and pipeline
    storage (Lift Station).

    System Topology (Two-Stage):
        1. Supply:       Aquifer -> Well Pump (30HP) -> Ground Storage Tank (GST)
        2. Distribution: GST -> Booster Pumps (3x 25HP) -> Hydro Tank -> Community
        3. Auxiliary:    Air Compressor (5HP) for Hydro Tank pressure maintenance

    State Management -- WNTR vs. Euler Integration:
        I use WNTR solely to construct the hydraulic network topology (nodes,
        pipes, tanks, junctions). I do not call wn.simulate() or any WNTR
        hydraulic solver. State evolution is handled by explicit first-order
        Euler integration in update(), which advances GST and Hydro Tank levels
        via mass balance at each 15-minute timestep.

        I chose Euler integration over WNTR's iterative pressure solver because
        the receding-horizon MILP/NMPC loop requires deterministic, low-latency
        state transitions at every optimization step. WNTR's solver introduces
        convergence overhead that would make the 672-step (7-day) simulation
        intractable at 15-minute resolution.

        The WNTR model (self.wn) is retained as a topological reference artifact
        and provides a foundation for future validation: comparing Euler-integrated
        levels against a full EPANET pressure-driven simulation over a
        representative 24-hour period.

        This deviation is documented in the capstone report under Assumptions
        and Limitations.

Inputs:
    - config: Dictionary from gwtp_config.yaml
Outputs:
    - State Dictionary: {'gst_level_ft', 'gst_volume_gal',
                         'hydro_level_ft', 'system_pressure_psi'}
Dependencies:
    - wntr, logging, numpy
===============================================================================
"""

import wntr
import logging
import numpy as np
from typing import Dict, Any


class GroundwaterPlant:
    """
    Digital Twin of the Groundwater Treatment Plant.

    Tracks two hydraulic state variables via Euler integration:
        gst_level   (ft): Ground Storage Tank fill level
        hydro_level (ft): Hydropneumatic Tank fill level

    The WNTR network (self.wn) is built once at initialization for topological
    reference only. See module docstring for the rationale on why I do not
    invoke the WNTR hydraulic solver.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GWTP Digital Twin from gwtp_config.yaml.

        Reads storage geometry, pump nameplate data, and initial conditions.
        Computes tank cross-sectional areas once here so update() avoids
        redundant math at every timestep.

        Args:
            config: Parsed YAML configuration dictionary (gwtp_config.yaml).
        """
        self.config    = config
        self.gst_cfg   = config['storage']['gst']
        self.hydro_cfg = config['storage']['hydro_tank']
        self.pumps_cfg = config['pumps']

        # Internal state variables -- tracked here rather than in the WNTR
        # model object, which exposes tank levels as read-only properties.
        self.gst_level   = float(self.gst_cfg['initial_level_ft'])
        self.hydro_level = float(self.hydro_cfg['initial_level_ft'])

        # Cross-sectional areas (ft^2) for level-to-volume conversion.
        # Hydro Tank uses effective_diameter_ft to model horizontal vessel volume.
        self.A_gst   = np.pi * (self.gst_cfg['diameter_ft'] / 2) ** 2
        self.A_hydro = np.pi * (self.hydro_cfg['effective_diameter_ft'] / 2) ** 2

        # Build topology reference (no hydraulic solve is run; see module docstring).
        self.wn = self._build_network()

        logging.info(
            f"GWTP Digital Twin initialized:\n"
            f"   - Nodes: {self.wn.num_nodes} (GST + Hydro Tank)\n"
            f"   - Pumps: 1x Well (30HP), 3x Boosters (25HP)\n"
            f"   - Aux:   1x Compressor (5HP)"
        )

    def _build_network(self) -> wntr.network.WaterNetworkModel:
        """
        Build the GWTP hydraulic network topology using WNTR.

        Constructs all nodes (reservoirs, tanks, junctions) and links (pipes)
        that describe the two-stage supply system. The returned model is stored
        as self.wn for topological reference. No WNTR solver method is called
        here or elsewhere in the simulation loop.
        """
        wn = wntr.network.WaterNetworkModel()

        # --- Nodes ---
        wn.add_reservoir('Aquifer', base_head=0.0)

        wn.add_tank(
            name='GST',
            elevation=0.0,
            init_level=self.gst_level,
            min_level=self.gst_cfg['min_level_ft'],
            max_level=self.gst_cfg['max_level_ft'],
            diameter=self.gst_cfg['diameter_ft']
        )

        wn.add_junction('Suction_Header',  elevation=0.0)
        wn.add_junction('Discharge_Header', elevation=0.0)

        wn.add_tank(
            name='Hydro_Tank',
            elevation=0.0,
            init_level=self.hydro_level,
            min_level=self.hydro_cfg['min_level_ft'],
            max_level=self.hydro_cfg['max_level_ft'],
            diameter=self.hydro_cfg['effective_diameter_ft']
        )

        wn.add_junction('Community_Demand', elevation=5.0, base_demand=0.0)

        # --- Links ---
        wn.add_pipe('Well_Pipe',   'Aquifer',          'GST',              length=100,  diameter=8,  roughness=100)
        wn.add_pipe('GST_Outlet',  'GST',              'Suction_Header',   length=20,   diameter=12, roughness=120)

        for i in range(1, 4):
            wn.add_pipe(f'Booster_{i}', 'Suction_Header', 'Discharge_Header', length=10, diameter=6, roughness=120)

        wn.add_pipe('Main_1', 'Discharge_Header', 'Hydro_Tank',        length=50,   diameter=12, roughness=120)
        wn.add_pipe('Main_2', 'Hydro_Tank',       'Community_Demand',  length=2000, diameter=12, roughness=120)

        return wn

    def update(self, dt_min: int, well_duty: float, booster_duty: float, demand_gpm: float):
        """
        Advance the GWTP Digital Twin by dt_min minutes via Euler integration.

        Applies two sequential mass balances:
            1. GST:   dV = (Q_well_in - Q_boost_out) * dt
            2. Hydro: dV = (Q_boost_out - Q_demand)  * dt

        Flow rates (GPM) are converted to ft^3/hr before integration using
        the factor GPM * 60 / 7.48. Level changes (ft) follow from dV / area.

        Args:
            dt_min:       Timestep duration in minutes.
            well_duty:    Well pump fractional command [0.0, 1.0].
            booster_duty: Booster pump aggregate fractional command [0.0, 3.0]
                          (sum of three binary or continuous pump signals).
            demand_gpm:   Community water demand (gallons per minute).
        """
        dt_hr = dt_min / 60.0

        # Convert design flows from GPM to ft^3/hr (multiply by 60 min/hr / 7.48 gal/ft^3).
        q_well_max  = (self.pumps_cfg['well_pump']['design_flow_gpm']    * 60) / 7.48
        q_boost_max = (self.pumps_cfg['booster_pumps']['design_flow_gpm'] * 60) / 7.48

        q_well_in   = q_well_max  * well_duty
        q_boost_out = q_boost_max * booster_duty
        q_demand    = (demand_gpm * 60) / 7.48

        # GST mass balance: well pump fills, boosters draw down.
        dv_gst = (q_well_in - q_boost_out) * dt_hr
        self.gst_level += dv_gst / self.A_gst

        # Enforce physical bounds -- GST cannot overflow or go below dead storage.
        self.gst_level = max(float(self.gst_cfg['min_level_ft']),
                             min(self.gst_level, float(self.gst_cfg['max_level_ft'])))

        # Hydro Tank mass balance: boosters fill, community demand draws down.
        dv_hydro = (q_boost_out - q_demand) * dt_hr
        self.hydro_level += dv_hydro / self.A_hydro

        # Enforce physical bounds -- Hydro Tank cannot drain below minimum pressure head.
        self.hydro_level = max(float(self.hydro_cfg['min_level_ft']),
                               min(self.hydro_level, float(self.hydro_cfg['max_level_ft'])))

    def get_system_state(self) -> Dict[str, float]:
        """
        Return current GWTP state for logging and optimizer seeding.

        Pressure is derived from Hydro Tank level using the hydrostatic
        approximation (0.433 PSI/ft) plus a 40 PSI static base pressure
        representing the distribution system set-point.

        Returns:
            dict with keys:
                gst_level_ft         -- GST fill level (ft)
                gst_volume_gal       -- GST stored volume (gallons)
                hydro_level_ft       -- Hydro Tank fill level (ft)
                system_pressure_psi  -- Estimated distribution pressure (PSI)
        """
        gst_vol    = (self.A_gst * self.gst_level) * 7.48
        hydro_psi  = (self.hydro_level * 0.433) + 40.0

        return {
            'gst_level_ft':        self.gst_level,
            'gst_volume_gal':      gst_vol,
            'hydro_level_ft':      self.hydro_level,
            'system_pressure_psi': hydro_psi,
        }

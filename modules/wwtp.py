"""
===============================================================================
Module:       wwtp.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Physics engine (Digital Twin) for the Wastewater Treatment Plant (WWTP).
    Models the biological oxygen battery -- the second storage buffer in the
    Triple-Battery framework alongside elevation storage (GWTP) and pipeline
    storage (Lift Station).

    I track dissolved oxygen (DO) in the aeration basin via a first-order
    ODE driven by the oxygen transfer coefficient KLa and a fixed oxygen
    uptake rate (OUR). The blower duty cycle is the primary control variable.
    Wet well hydraulics are handled separately by LiftStation (lift_station.py).

    Architecture Note -- QSDsan Substitution:
        The original project scope included QSDsan for process-level wastewater
        modeling (stoichiometry, costing, LCA). I removed QSDsan during scope
        refinement to keep the framework focused on energy scheduling rather
        than treatment process design.

        In its place, I use a first-order ODE for the biological oxygen
        battery: DO dynamics are governed by a KLa transfer term and a fixed
        OUR, both parameterized from nameplate blower data and standard
        activated-sludge design values. This is sufficient for evaluating
        blower scheduling strategies.

        This deviation is documented in the capstone report under Scope
        Changes and Limitations.

Inputs:
    - config:        Dictionary from wwtp_config.yaml
    - dt_min:        Simulation timestep in minutes
    - blower_duty:   Blower control signal [0.0, 1.0]
    - influent_mgd:  Influent flow rate (Million Gallons per Day)
Outputs:
    - State Dictionary: {'do_mg_l', 'blower_kw_active'}
Dependencies:
    - None (pure Python, no external solvers)
===============================================================================
"""


class WastewaterPlant:
    """
    Digital Twin of the Wastewater Treatment Plant aeration basin.

    Models dissolved oxygen (DO) dynamics as a first-order ODE:
        dDO/dt = KLa(t) * (DO_sat - DO(t)) - OUR(t)

    where KLa is proportional to blower duty and OUR scales with influent
    load. Both MILP and NMPC optimizers reference these same parameters to
    ensure the optimizer and the digital twin are physically consistent.
    """

    def __init__(self, config):
        """
        Initialize the WWTP physics engine from wwtp_config.yaml.

        Reads blower nameplate data and activated-sludge biology parameters.
        All constants are sourced from config to maintain Digital Twin fidelity
        with the optimizer tier.
        """
        self.config = config

        # Biological state -- aeration basin dissolved oxygen
        bio = config['biology']
        self.do_mg_l  = bio['do_initial_mg_l']
        self.do_sat   = bio['do_saturation_limit']
        self.kla_max  = bio['kla_max']
        self.our_base = bio['our_base']

        # Asset specifications -- electrical power for cost accounting
        self.blower_kw = config['blowers']['aeration_blower']['rated_kw']

    def update(self, dt_min, blower_duty, influent_mgd):
        """
        Advance the biological oxygen battery state by dt_min minutes.

        Uses first-order Euler integration of the DO mass balance ODE.
        Load factor scales OUR linearly with influent flow relative to the
        design capacity of 1.5 MGD -- a standard activated-sludge approximation
        that avoids the nonlinear ASM1 stoichiometry not needed at this scope.

        Args:
            dt_min:       Timestep in minutes.
            blower_duty:  Blower control fraction [0.0, 1.0].
            influent_mgd: Influent flow rate (Million Gallons per Day).
        """
        dt_hr = dt_min / 60.0

        # Scale OUR by influent load relative to 1.5 MGD design capacity.
        # Linear scaling is a simplification of ASM1 kinetics; acceptable here
        # because the optimizer is evaluated on energy cost, not effluent quality.
        load_factor = influent_mgd / 1.5
        current_our = self.our_base * load_factor

        aeration_gain = (blower_duty * self.kla_max) * (self.do_sat - self.do_mg_l)
        self.do_mg_l += (aeration_gain - current_our) * dt_hr

        # Clamp to physical bounds: 0.1 mg/L floor prevents numerical singularities;
        # do_sat ceiling prevents supersaturation artifacts.
        self.do_mg_l = max(0.1, min(self.do_mg_l, self.do_sat))

    def get_system_state(self):
        """
        Return current WWTP state for logging and optimizer seeding.

        Returns:
            dict with keys:
                do_mg_l          -- current dissolved oxygen (mg/L)
                blower_kw_active -- nameplate blower power for cost accounting (kW)
        """
        return {
            'do_mg_l':          round(self.do_mg_l, 3),
            'blower_kw_active': self.blower_kw
        }

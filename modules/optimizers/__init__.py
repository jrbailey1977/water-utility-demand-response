"""
===============================================================================
Module:       __init__.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Optimizer Factory implementing the Strategy Pattern.
    Reads 'active_strategy' from global_settings.yaml and returns the
    appropriate optimizer instances for GWTP, WWTP, and LS systems.

    Supported strategies:
        milp  - Mixed-Integer Linear Program (GLPK solver via Pyomo)
        nmpc  - Nonlinear MPC (do-mpc / CasADi / Ipopt)
        ai    - Reinforcement Learning agent (stub, not yet implemented)
===============================================================================
"""


def create_optimizers(strategy: str, g_cfg, w_cfg, ls_cfg, demand_penalty: float):
    """
    Factory function that returns a (gwtp_opt, wwtp_opt, ls_opt) tuple
    for the requested strategy. Raises ValueError for unknown strategies.
    """
    strategy = strategy.lower().strip()

    if strategy == 'milp':
        from modules.optimizers.milp.gwtp_milp import GWTP_MILP
        from modules.optimizers.milp.wwtp_milp import WWTP_MILP
        from modules.optimizers.milp.ls_milp   import LiftStationMILP
        return (GWTP_MILP(g_cfg, demand_penalty),
                WWTP_MILP(w_cfg, demand_penalty),
                LiftStationMILP(ls_cfg, demand_penalty))

    elif strategy == 'nmpc':
        from modules.optimizers.nmpc.gwtp_nmpc import GWTP_NMPC
        from modules.optimizers.nmpc.wwtp_nmpc import WWTP_NMPC
        from modules.optimizers.nmpc.ls_nmpc   import LiftStation_NMPC
        return (GWTP_NMPC(g_cfg, demand_penalty),
                WWTP_NMPC(w_cfg, demand_penalty),
                LiftStation_NMPC(ls_cfg, demand_penalty))

    elif strategy == 'ai':
        raise NotImplementedError("AI agent strategy is defined but not yet implemented.")

    else:
        raise ValueError(
            f"Unknown active_strategy '{strategy}'. Valid options: milp, nmpc, ai."
        )

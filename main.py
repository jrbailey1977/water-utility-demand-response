"""
===============================================================================
Module:       main.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:  
    Unified Orchestration Layer for Water/Wastewater Digital Twins. 
    Simulates integrated municipal Demand Response vs. Hysteresis Baseline.
    Performs independent financial auditing for GWTP, WWTP, and LS systems.

    Implements the Strategy Pattern via modules/optimizers/__init__.py, allowing
    the optimizer tier to be selected at runtime without modifying simulation logic.
    Supported strategies: milp (Pyomo/GLPK), nmpc (do-mpc/CasADi/Ipopt).
    
    Key Logic:
        1. 15-Minute Resolution: Solves optimizer at every hydraulic timestep
           using a receding horizon to capture maximum energy price avoidance.
        2. Breakout Logging: Provides separate status updates for Water and 
           Wastewater systems to monitor physical stability (GST/DO levels).
        3. Triple-Battery Audit: Quantifies savings for Elevation, Biological, 
           and Pipeline storage independent of one another.
        4. Side-by-Side Comparison: --strategy both runs MILP and NMPC against
           the same baseline and prints a unified 3-column audit table.

Inputs:       
    - System Configurations (gwtp_config.yaml, wwtp_config.yaml, ls_config.yaml)
    - ERCOT Real-Time/Day-Ahead Price Forecasts ($/kWh)
    - Municipal Water Demand Forecasts (GPM)
Outputs:      
    - Performance Audit Table (Cost comparison and % avoided cost)
CLI Args:
    --strategy {milp, nmpc, both}  Override active_strategy in global_settings.yaml.
                                   Use 'both' for a side-by-side MILP vs NMPC comparison.
Dependencies: 
    - pandas, yaml, argparse, pyomo, do-mpc, casadi, logging, modules.data_manager
    Note: QSDsan was removed from scope; wwtp.py uses a first-order ODE
    model in its place. See wwtp.py class docstring for full rationale.
===============================================================================
"""

import os
import yaml
import logging
import argparse
import pandas as pd
from datetime import datetime
from modules.data_manager import DataManager
from modules.gwtp import GroundwaterPlant
from modules.wwtp import WastewaterPlant
from modules.lift_station import LiftStation
from modules.optimizers import create_optimizers
from modules.utils import SimLogger, write_sensitivity_table

def setup_environment():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # Suppress duplicate console output from gridstatusio's own log handler.
    logging.getLogger('gridstatusio').setLevel(logging.WARNING)
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def prepare_forecast_data(df):
    df['demand_ft3_hr'] = (df['demand_gpm'] * 60) / 7.48
    
    # Shift by 8 timesteps (2 hrs) to model wastewater lag behind potable demand.
    # Fill leading NaNs with the series mean rather than bfill() to avoid
    # incorrectly seeding the first 2 hours with hour-3 values.
    influent_raw = df['demand_gpm'].shift(8) * 1440 / 1000000
    df['influent_mgd'] = influent_raw.fillna(influent_raw.mean())
    return df

def get_baseline_commands(state_w, state_s, state_ls, g_cfg, l_cfg):
    # GST: Forces the baseline to maintain at least 50% storage (10ft)
    well_cmd = 1.0 if state_w['gst_level_ft'] < 10.0 else 0.0

    # Hydro: Keeps pressure in the mid-range (4ft)
    booster_cmd = 1.0 if state_w['hydro_level_ft'] < 4.0 else 0.0

    # Wastewater: Maintains a standard 2.0 mg/L DO baseline
    blower_cmd = 1.0 if state_s['do_mg_l'] < 2.0 else 0.0

    # LS: Hysteresis band control - lead pump on at 7ft, off at 4ft
    ls_count = 1.0 if state_ls['wet_well_ft'] >= 7.0 else 0.0

    return well_cmd, booster_cmd, blower_cmd, ls_count

def run_simulation(mode, dm, forecast_df, g_cfg, w_cfg, ls_cfg, sim_params, demand_penalty=0.0, strategy=None, demand_rate=15.0, period_fraction=1.0):
    timestep = sim_params['hydraulic_timestep_min']
    horizon = (24 * 60) // timestep
    total_steps = len(forecast_df) - horizon
    physics_w = GroundwaterPlant(g_cfg)
    physics_s = WastewaterPlant(w_cfg)
    physics_ls = LiftStation(ls_cfg)

    if mode == 'optimized':
        if strategy is None:
            strategy = sim_params.get('active_strategy', 'milp')
        opt_g, opt_w, opt_ls = create_optimizers(strategy, g_cfg, w_cfg, ls_cfg, demand_penalty)
    else:
        opt_g = opt_w = opt_ls = None

    sim_start_date = sim_params.get('start_date', '2026-01-01 00:00:00')
    log_strategy   = strategy if mode == 'optimized' else 'baseline'
    sim_logger     = SimLogger(log_strategy, str(sim_start_date), demand_rate, period_fraction)

    stats = {'gwtp': 0.0, 'wwtp': 0.0, 'ls': 0.0,
             'peak_kw': 0.0, 'peak_kw_gwtp': 0.0, 'peak_kw_wwtp': 0.0, 'peak_kw_ls': 0.0}

    for t in range(total_steps):
        state_w = physics_w.get_system_state()
        state_s = physics_s.get_system_state()
        state_ls = physics_ls.get_system_state()
        price = forecast_df['price_kwh'].iloc[t]
        
        if mode == 'optimized':
            start_idx, end_idx = t, t + horizon
            p_prices = forecast_df['price_kwh'].iloc[start_idx:end_idx].tolist()
            p_dem = forecast_df['demand_ft3_hr'].iloc[start_idx:end_idx].tolist()
            p_inf = forecast_df['influent_mgd'].iloc[start_idx:end_idx].tolist()

            # Receding Horizon Control: dispatch index only
            res_g = opt_g.solve(state_w, p_prices, p_dem)
            well_cmd = float(res_g['well_pump'][0])
            booster_cmd = float(res_g['booster_pumps'][1][0] + res_g['booster_pumps'][2][0] + res_g['booster_pumps'][3][0])

            res_w = opt_w.solve(state_s, p_prices, p_inf)
            blower_cmd = float(res_w['blower_duty'][0])

            res_ls = opt_ls.solve(state_ls, p_prices, p_inf)
            ls_count = float(res_ls['ls_count'][0])
        else:
            well_cmd, booster_cmd, blower_cmd, ls_count = get_baseline_commands(state_w, state_s, state_ls, g_cfg, ls_cfg)

        # Update Digital Twin Physics
        physics_w.update(timestep, well_cmd, booster_cmd, forecast_df['demand_gpm'].iloc[t])
        physics_s.update(timestep, blower_cmd, forecast_df['influent_mgd'].iloc[t])
        physics_ls.update(timestep, ls_count, forecast_df['influent_mgd'].iloc[t])
        
        # Financial Auditing
        hr_share = timestep / 60.0
        
        # GWTP: Convert HP to electrical kW (shaft power / motor efficiency)
        well_eff = g_cfg['pumps']['well_pump'].get('motor_efficiency', 0.93)
        well_kw = g_cfg['pumps']['well_pump'].get('motor_hp', 30.0) * 0.7457 / well_eff
        booster_eff = g_cfg['pumps']['booster_pumps'].get('motor_efficiency', 0.917)
        booster_kw = g_cfg['pumps']['booster_pumps'].get('motor_hp', 25.0) * 0.7457 / booster_eff
        stats['gwtp'] += ((well_cmd * well_kw) + (booster_cmd * booster_kw)) * hr_share * price

        # WWTP: Blower rated_kw is electrical input - no efficiency correction needed
        stats['wwtp'] += (blower_cmd * 45.0) * hr_share * price

        # Lift Station: Convert HP to electrical kW (shaft power / motor efficiency)
        ls_hp = ls_cfg['pumps']['influent_pumps'].get('motor_hp', 50.0)
        ls_eff = ls_cfg['pumps']['influent_pumps'].get('motor_efficiency', 0.91)
        ls_kw = ls_hp * 0.7457 / ls_eff
        stats['ls'] += (ls_count * ls_kw) * hr_share * price

        # Peak Demand Tracking: record highest instantaneous kW per subsystem
        instant_kw_gwtp = (well_cmd * well_kw) + (booster_cmd * booster_kw)
        instant_kw_wwtp = blower_cmd * 45.0
        instant_kw_ls   = ls_count * ls_kw
        instant_kw      = instant_kw_gwtp + instant_kw_wwtp + instant_kw_ls

        stats['peak_kw']      = max(stats['peak_kw'],      instant_kw)
        stats['peak_kw_gwtp'] = max(stats['peak_kw_gwtp'], instant_kw_gwtp)
        stats['peak_kw_wwtp'] = max(stats['peak_kw_wwtp'], instant_kw_wwtp)
        stats['peak_kw_ls']   = max(stats['peak_kw_ls'],   instant_kw_ls)

        # Per-step solver times for CSV (last solve call times from each optimizer)
        step_solver_times = None
        if mode == 'optimized':
            step_solver_times = {
                'gwtp': opt_g.solve_times[-1] if opt_g.solve_times else 0.0,
                'wwtp': opt_w.solve_times[-1] if opt_w.solve_times else 0.0,
                'ls':   opt_ls.solve_times[-1] if opt_ls.solve_times else 0.0,
            }

        # Log this timestep to CSV
        sim_logger.log_step(
            timestamp   = forecast_df.index[t],
            price       = price,
            kw_gwtp     = instant_kw_gwtp,
            kw_wwtp     = instant_kw_wwtp,
            kw_ls       = instant_kw_ls,
            cost_gwtp   = ((well_cmd * well_kw) + (booster_cmd * booster_kw)) * hr_share * price,
            cost_wwtp   = (blower_cmd * 45.0) * hr_share * price,
            cost_ls     = (ls_count * ls_kw) * hr_share * price,
            state_w     = state_w,
            state_s     = state_s,
            state_ls    = state_ls,
            well_cmd    = well_cmd,
            booster_cmd = booster_cmd,
            blower_cmd  = blower_cmd,
            ls_count    = ls_count,
            solver_times= step_solver_times,
        )

        if t % 32 == 0:
            ts = forecast_df.index[t]
            log_line = (
                f"\n--- {mode.upper()} SNAPSHOT: {ts} ---\n"
                f"  WATER: GST {state_w['gst_level_ft']:.1f}ft | Hydro {state_w['hydro_level_ft']:.1f}ft | Well: {well_cmd}\n"
                f"  WASTE: DO {state_s['do_mg_l']:.1f}mg/L | WetWell {state_ls['wet_well_ft']:.1f}ft | LS Pumps: {ls_count}\n"
                f"  MARKET: Price ${price:.4f}/kWh"
            )
            logging.info(log_line)

    sim_logger.save()

    if mode == 'optimized':
        stats['timing'] = {
            'gwtp': opt_g.solver_stats(),
            'wwtp': opt_w.solver_stats(),
            'ls':   opt_ls.solver_stats(),
        }
    else:
        stats['timing'] = None

    return stats

def print_timing_table(optimized_results):
    """Print wall-clock solver timing statistics per facility and strategy."""
    facilities = [('gwtp', 'Water (GWTP)'), ('wwtp', 'Wastewater (WWTP)'), ('ls', 'Lift Station')]
    strategies = [(lbl, res['timing']) for lbl, res in optimized_results if res['timing']]
    if not strategies:
        return

    print("\n" + "=" * 72)
    print("SOLVER TIMING — wall-clock seconds per optimization call")
    print(f"{'FACILITY':<20}", end="")
    for lbl, _ in strategies:
        print(f"  {lbl:>6} mean  {lbl:>6} P95   {lbl:>6} max   {lbl:>6} n  ", end="")
    print()
    print("-" * 72)
    for key, name in facilities:
        print(f"{name:<20}", end="")
        for lbl, timing in strategies:
            t = timing[key]
            print(f"  {t['mean']:>9.3f}  {t['p95']:>9.3f}  {t['max']:>9.3f}  {t['n']:>6}  ", end="")
        print()
    print("=" * 72 + "\n")


def print_audit_table(sim_days, demand_charge_rate, period_fraction, res_base, results):
    """
    Print the Triple-Battery audit table.
    results: list of (label, res_dict) tuples — supports 1 or 2 optimized strategies.

    Column layout per strategy: COST | SAV$ | SAV%
    """
    # Build header row matching actual data column layout
    col_width = 12   # width of each cost/savings column
    hdr  = f"{'SYSTEM':<14}| {'BASELINE':>10} "
    sep  = f"{'-'*14}+-{'-'*11}"
    for lbl, _ in results:
        hdr += f"| {lbl+' COST':>10} | {lbl+' SAV$':>10} | {lbl+' SAV%':>9} "
        sep += f"+-{'-'*11}+-{'-'*11}+-{'-'*10}"
    W = len(sep)

    print("\n" + "=" * W)
    print(f"  TRIPLE-BATTERY MUNICIPAL ENERGY AUDIT — {sim_days} DAYS")
    print("=" * W)
    print(hdr)
    print(sep)

    # --- Energy cost rows ---
    t_energy_base = sum(res_base[k] for k in ['gwtp', 'wwtp', 'ls'])
    for key, name in [('gwtp', 'Water'), ('wwtp', 'Wastewater'), ('ls', 'Lift Station'), (None, 'Energy Total')]:
        base_val = sum(res_base[k] for k in ['gwtp','wwtp','ls']) if key is None else res_base[key]
        row = f"{name:<14}| ${base_val:>9.2f} "
        for lbl, res in results:
            opt_val = sum(res[k] for k in ['gwtp','wwtp','ls']) if key is None else res[key]
            s = base_val - opt_val
            p = (s / base_val * 100) if base_val else 0
            row += f"| ${opt_val:>9.2f} | ${s:>9.2f} | {p:>8.1f}% "
        print(row)
        if key is None:
            print(sep)

    # --- Peak demand rows ---
    print(f"  {'PEAK DEMAND':}")
    for key, name in [('peak_kw_gwtp', 'Water'), ('peak_kw_wwtp', 'Wastewater'), ('peak_kw_ls', 'Lift Station'), ('peak_kw', 'TOTAL PEAK kW')]:
        base_val = res_base[key]
        row = f"{name:<14}|  {base_val:>9.1f} "
        for lbl, res in results:
            opt_val = res[key]
            s = base_val - opt_val
            p = (s / base_val * 100) if base_val else 0
            row += f"|  {opt_val:>9.1f} |  {s:>9.1f} | {p:>8.1f}% "
        print(row)

    print(sep)

    # --- Demand charge row ---
    dc_base = res_base['peak_kw'] * demand_charge_rate * period_fraction
    row = f"{'Demand Charge':<14}| ${dc_base:>9.2f} "
    for lbl, res in results:
        dc = res['peak_kw'] * demand_charge_rate * period_fraction
        s = dc_base - dc
        p = (s / dc_base * 100) if dc_base else 0
        row += f"| ${dc:>9.2f} | ${s:>9.2f} | {p:>8.1f}% "
    print(row)

    print(sep)

    # --- Total bill row ---
    t_base = t_energy_base + dc_base
    row = f"{'TOTAL BILL':<14}| ${t_base:>9.2f} "
    for lbl, res in results:
        t_e = sum(res[k] for k in ['gwtp', 'wwtp', 'ls'])
        dc  = res['peak_kw'] * demand_charge_rate * period_fraction
        t_opt = t_e + dc
        s = t_base - t_opt
        p = (s / t_base * 100) if t_base else 0
        row += f"| ${t_opt:>9.2f} | ${s:>9.2f} | {p:>8.1f}% "
    print(row)
    print("=" * W + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Triple-Battery Water Utility Energy Audit')
    parser.add_argument(
        '--strategy',
        choices=['milp', 'nmpc', 'both'],
        default=None,
        help='Optimizer strategy to run. Defaults to active_strategy in global_settings.yaml. '
             'Use "both" for a side-by-side MILP vs NMPC comparison.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment()
    g_cfg = load_config('config/gwtp_config.yaml')
    w_cfg = load_config('config/wwtp_config.yaml')
    l_cfg = load_config('config/ls_config.yaml')
    global_cfg = load_config('config/global_settings.yaml')
    sim_params = global_cfg['simulation']
    fin_cfg    = global_cfg['financial']

    dm = DataManager({
        'simulation_days': sim_params['duration_days'],
        'timestep_minutes': sim_params['hydraulic_timestep_min'],
        'start_date': sim_params.get('start_date', '2026-01-01'),
        'demand_charge_per_kw': fin_cfg.get('demand_charge_per_kw', 15.0)
    })
    forecast_df = prepare_forecast_data(dm.get_forecast('groundwater', use_real_ercot=True))

    demand_rate     = dm.demand_charge_rate
    period_fraction = sim_params['duration_days'] / 30.0
    demand_penalty  = demand_rate * period_fraction

    logging.info("--- Starting Triple-Battery Performance Audit ---")
    res_base = run_simulation('baseline', dm, forecast_df, g_cfg, w_cfg, l_cfg, sim_params, demand_penalty,
                              demand_rate=demand_rate, period_fraction=period_fraction)

    # Determine which strategies to run
    strategy_arg = args.strategy or sim_params.get('active_strategy', 'milp')

    if strategy_arg == 'both':
        logging.info("Running MILP optimizer...")
        res_milp = run_simulation('optimized', dm, forecast_df, g_cfg, w_cfg, l_cfg, sim_params, demand_penalty,
                                  strategy='milp', demand_rate=demand_rate, period_fraction=period_fraction)
        logging.info("Running NMPC optimizer...")
        res_nmpc = run_simulation('optimized', dm, forecast_df, g_cfg, w_cfg, l_cfg, sim_params, demand_penalty,
                                  strategy='nmpc', demand_rate=demand_rate, period_fraction=period_fraction)
        optimized_results = [('MILP', res_milp), ('NMPC', res_nmpc)]
    else:
        logging.info(f"Running {strategy_arg.upper()} optimizer...")
        res_opt = run_simulation('optimized', dm, forecast_df, g_cfg, w_cfg, l_cfg, sim_params, demand_penalty,
                                 strategy=strategy_arg, demand_rate=demand_rate, period_fraction=period_fraction)
        optimized_results = [(strategy_arg.upper(), res_opt)]

    print_audit_table(sim_params['duration_days'], demand_rate, period_fraction, res_base, optimized_results)
    print_timing_table(optimized_results)

    # Write sensitivity table
    sim_start = sim_params.get('start_date', '2026-01-01 00:00:00')
    write_sensitivity_table(str(sim_start), res_base, optimized_results, demand_rate, period_fraction)

if __name__ == "__main__":
    main()
"""
===============================================================================
Module:       utils.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Shared helper functions and classes used across multiple modules to
    ensure consistency and reduce code duplication.

    Sections:
        1. Unit Conversions: Standardized US/SI helpers (GPM to ft^3/hr,
           MGD to GPM, area to gal/ft, ft to PSI, HP to kW). Inline
           equivalents remain in the physics modules for locality; these
           functions are available for future refactoring.

        2. Forecast / Matrix Helpers: Array padding and clipping utilities
           shared between optimizer tiers.

        3. Python Logging Setup: Consistent console + file handler
           configuration for use in main.py.

        4. SimLogger: Per-timestep CSV data capture class. Instantiated
           once per run_simulation() call. Captures physical state, control
           actions, market data, power draw, cumulative financials, and
           solver metadata. Writes one CSV per run to
           logs/simulation_<date>_<strategy>.csv.

        5. Sensitivity Table Writer: Post-simulation retail rate scaling
           analysis. Scales simulated energy savings across four rate
           scenarios (wholesale, basic retail, full retail, peak summer)
           to show real-world dollar potential beyond ERCOT wholesale
           prices. Writes to logs/sensitivity_<date>.csv.

Inputs:
    - Various arguments depending on function (floats, lists, dicts, paths)
Outputs:
    - Converted float values
    - Configured Logger objects
    - CSV files in logs/ directory
Dependencies:
    - logging, os, csv, numpy, datetime
===============================================================================
"""

import os
import csv
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


# ==============================================================================
# SECTION 1 -- UNIT CONVERSIONS
# ==============================================================================

def gpm_to_ft3_per_hr(gpm: float) -> float:
    """Convert gallons per minute to cubic feet per hour.

    Factor: 1 gal = 1/7.48 ft^3; 60 min/hr -- multiply by 60/7.48.
    """
    return gpm * 60.0 / 7.48


def mgd_to_gpm(mgd: float) -> float:
    """Convert million gallons per day to gallons per minute."""
    return (mgd * 1_000_000.0) / 1440.0


def gal_per_ft(area_sqft: float) -> float:
    """Return gallons per foot of depth for a cylindrical or rectangular vessel.

    Factor: 7.48 gallons per cubic foot.
    """
    return area_sqft * 7.48


def ft_to_psi(level_ft: float, base_psi: float = 40.0) -> float:
    """Convert water column height to gauge pressure in PSI.

    Factor: 0.433 PSI per foot of water. base_psi represents the static
    distribution system set-point (default 40 PSI for residential service).
    """
    return level_ft * 0.433 + base_psi


def hp_to_kw(hp: float, efficiency: float = 1.0) -> float:
    """Convert horsepower to kilowatts with optional motor efficiency correction.

    Factor: 1 HP = 0.7457 kW (shaft power). Pass motor_efficiency < 1.0 to
    get electrical input kW from nameplate shaft HP.
    """
    return hp * 0.7457 / efficiency


# ==============================================================================
# SECTION 2 -- FORECAST / MATRIX HELPERS
# ==============================================================================

def pad_forecast(lst: List[float], n: int) -> List[float]:
    """Pad or truncate a forecast list to exactly n entries.

    If shorter than n, repeats the last value (hold-last extrapolation).
    If longer, truncates. I use hold-last rather than zero-padding because
    zeroing future prices would artificially inflate end-of-horizon pumping.
    """
    if not lst:
        return [0.0] * n
    if len(lst) >= n:
        return list(lst[:n])
    return list(lst) + [lst[-1]] * (n - len(lst))


def clip_control(value: float, lo: float, hi: float) -> float:
    """Clip a solver output to valid control bounds."""
    return float(np.clip(value, lo, hi))


# ==============================================================================
# SECTION 3 -- PYTHON LOGGING SETUP
# ==============================================================================

def setup_logging(log_dir: str = 'logs', level: int = logging.INFO) -> None:
    """Configure root logger with console and rotating file handlers.

    Suppresses noisy third-party loggers (gridstatusio, do_mpc, casadi,
    pyomo) that would otherwise bury simulation progress output.
    """
    os.makedirs(log_dir, exist_ok=True)
    fmt = '%(asctime)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)

    fh = logging.FileHandler(os.path.join(log_dir, 'simulation.log'), mode='a')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)

    for noisy in ('gridstatusio', 'do_mpc', 'casadi', 'pyomo'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ==============================================================================
# SECTION 4 -- SIMLOGGER
# ==============================================================================

_CSV_COLUMNS = [
    # Metadata
    'timestamp', 'strategy', 'sim_start_date', 'day_of_sim', 'hour_of_day',
    # Market
    'price_per_kwh', 'rolling_peak_kw', 'cumulative_demand_charge_usd',
    # GWTP
    'gst_level_ft', 'hpt_level_ft',
    'well_speed', 'booster_speed', 'booster_count',
    'gwtp_power_kw', 'gwtp_energy_cost_cumulative_usd',
    # WWTP
    'do_mg_l', 'blower_duty', 'blower_count',
    'wwtp_power_kw', 'wwtp_energy_cost_cumulative_usd',
    # Lift Station
    'wet_well_level_ft', 'ls_pump_speed', 'ls_pump_count',
    'ls_power_kw', 'ls_energy_cost_cumulative_usd',
    # System aggregate
    'total_power_kw', 'total_energy_cost_cumulative_usd', 'total_bill_cumulative_usd',
    # Solver metadata (blank for baseline)
    'gwtp_solve_time_s', 'wwtp_solve_time_s', 'ls_solve_time_s',
    'gwtp_solver_status', 'wwtp_solver_status', 'ls_solver_status',
]


class SimLogger:
    """
    Per-timestep CSV logger for a single simulation run.

    Instantiate once per run_simulation() call, call log_step() at each
    timestep, then call save() at the end to flush to disk. Accumulates
    cumulative energy cost and rolling peak demand in memory to avoid
    repeated re-summation over the full row list.
    """

    def __init__(
        self,
        strategy: str,
        sim_start_date: str,
        demand_charge_rate: float,
        period_fraction: float,
        log_dir: str = 'logs',
    ):
        self.strategy        = strategy
        self.sim_start_date  = sim_start_date
        self.demand_rate     = demand_charge_rate
        self.period_fraction = period_fraction
        self.log_dir         = log_dir
        self._rows: List[Dict] = []

        # Running accumulators updated each timestep
        self._cum_gwtp = 0.0
        self._cum_wwtp = 0.0
        self._cum_ls   = 0.0
        self._peak_kw  = 0.0

    def log_step(
        self,
        timestamp: datetime,
        price: float,
        kw_gwtp: float,
        kw_wwtp: float,
        kw_ls: float,
        cost_gwtp: float,
        cost_wwtp: float,
        cost_ls: float,
        state_w: Dict,
        state_s: Dict,
        state_ls: Dict,
        well_cmd: float,
        booster_cmd: float,
        blower_cmd: float,
        ls_count: float,
        solver_times: Optional[Dict] = None,
    ) -> None:
        """Record one 15-minute timestep to the in-memory row buffer.

        Args:
            timestamp:    Datetime index for this step.
            price:        ERCOT day-ahead price ($/kWh).
            kw_gwtp:      Instantaneous GWTP power draw (kW).
            kw_wwtp:      Instantaneous WWTP power draw (kW).
            kw_ls:        Instantaneous Lift Station power draw (kW).
            cost_gwtp:    Energy cost this step for GWTP ($).
            cost_wwtp:    Energy cost this step for WWTP ($).
            cost_ls:      Energy cost this step for Lift Station ($).
            state_w:      GWTP state dict from get_system_state().
            state_s:      WWTP state dict from get_system_state().
            state_ls:     Lift Station state dict from get_system_state().
            well_cmd:     Well pump command dispatched this step.
            booster_cmd:  Booster pump command dispatched this step.
            blower_cmd:   Blower duty command dispatched this step.
            ls_count:     Lift station pump command dispatched this step.
            solver_times: Dict of per-facility solve times (s); None for baseline.
        """
        total_kw       = kw_gwtp + kw_wwtp + kw_ls
        self._peak_kw  = max(self._peak_kw, total_kw)
        self._cum_gwtp += cost_gwtp
        self._cum_wwtp += cost_wwtp
        self._cum_ls   += cost_ls

        cum_energy = self._cum_gwtp + self._cum_wwtp + self._cum_ls
        cum_demand = self._peak_kw * self.demand_rate * self.period_fraction
        cum_bill   = cum_energy + cum_demand

        is_nmpc = (self.strategy == 'nmpc')

        # Booster speed column interpretation per strategy:
        #   NMPC     -- booster_cmd is a normalized VFD speed [0, 1]
        #   MILP     -- booster_cmd is the sum of three continuous duty fractions [0, 3]
        #   Baseline -- booster_cmd is a binary on/off command (0.0 or 1.0)
        # All three produce a meaningful value; log directly rather than
        # discarding non-NMPC data by hardcoding 0.0.
        booster_speed = booster_cmd
        booster_count = 0 if is_nmpc else int(round(booster_cmd))

        # Blower duty column interpretation per strategy:
        #   NMPC     -- blower_cmd is a continuous duty fraction [0, 1]
        #   MILP     -- blower_cmd is a continuous duty fraction [0, 1]
        #   Baseline -- blower_cmd is a binary on/off command (0.0 or 1.0)
        # All three are meaningful as a duty fraction; log directly.
        blower_duty  = blower_cmd
        blower_count = 0 if is_nmpc else int(round(blower_cmd))

        # LS pump speed is only meaningful for NMPC, which uses a continuous
        # VFD speed. MILP and baseline dispatch integer pump counts, so
        # ls_pump_speed is zeroed for those strategies to avoid misleading output.
        ls_speed = ls_count if is_nmpc else 0.0
        ls_int   = 0 if is_nmpc else int(round(ls_count))

        st = solver_times or {}

        try:
            start_dt = datetime.strptime(self.sim_start_date, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            start_dt = datetime.strptime(self.sim_start_date, '%Y-%m-%d')
        day_of_sim = (timestamp - start_dt).days + 1

        self._rows.append({
            'timestamp':                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'strategy':                         self.strategy,
            'sim_start_date':                   self.sim_start_date,
            'day_of_sim':                       day_of_sim,
            'hour_of_day':                      timestamp.hour,
            'price_per_kwh':                    round(price, 6),
            'rolling_peak_kw':                  round(self._peak_kw, 3),
            'cumulative_demand_charge_usd':      round(cum_demand, 4),
            'gst_level_ft':                     round(state_w.get('gst_level_ft', 0.0), 3),
            'hpt_level_ft':                     round(state_w.get('hydro_level_ft', 0.0), 3),
            'well_speed':                       round(well_cmd, 4),
            'booster_speed':                    round(booster_speed, 4),
            'booster_count':                    booster_count,
            'gwtp_power_kw':                    round(kw_gwtp, 3),
            'gwtp_energy_cost_cumulative_usd':  round(self._cum_gwtp, 4),
            'do_mg_l':                          round(state_s.get('do_mg_l', 0.0), 3),
            'blower_duty':                      round(blower_duty, 4),
            'blower_count':                     blower_count,
            'wwtp_power_kw':                    round(kw_wwtp, 3),
            'wwtp_energy_cost_cumulative_usd':  round(self._cum_wwtp, 4),
            'wet_well_level_ft':                round(state_ls.get('wet_well_ft', 0.0), 3),
            'ls_pump_speed':                    round(ls_speed, 4),
            'ls_pump_count':                    ls_int,
            'ls_power_kw':                      round(kw_ls, 3),
            'ls_energy_cost_cumulative_usd':    round(self._cum_ls, 4),
            'total_power_kw':                   round(total_kw, 3),
            'total_energy_cost_cumulative_usd': round(cum_energy, 4),
            'total_bill_cumulative_usd':        round(cum_bill, 4),
            'gwtp_solve_time_s':                round(st.get('gwtp', 0.0), 4),
            'wwtp_solve_time_s':                round(st.get('wwtp', 0.0), 4),
            'ls_solve_time_s':                  round(st.get('ls',   0.0), 4),
            'gwtp_solver_status':               st.get('gwtp_status', ''),
            'wwtp_solver_status':               st.get('wwtp_status', ''),
            'ls_solver_status':                 st.get('ls_status',   ''),
        })

    def save(self) -> str:
        """Write all buffered rows to CSV and return the file path."""
        os.makedirs(self.log_dir, exist_ok=True)
        date_tag = self.sim_start_date[:10]
        filepath = os.path.join(self.log_dir, f"simulation_{date_tag}_{self.strategy}.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        logging.info(f"SimLogger: saved {len(self._rows)} rows -> {filepath}")
        return filepath


# ==============================================================================
# SECTION 5 -- SENSITIVITY TABLE WRITER
# ==============================================================================

# Rate scenarios: (label, energy adder $/kWh, demand rate $/kW -- None = use simulation rate)
# Sources:
#   EIA/DOE 2024: Texas commercial average ~$0.09/kWh all-in
#   CenterPoint TDU delivery: ~$0.045-0.05/kWh
#   ERCOT summer on-peak forward strip: $0.12-0.15/kWh range (2025-2026)
_RATE_SCENARIOS = [
    ('Wholesale only (simulated)',  0.00,  None),
    ('+ Basic retail markup',       0.06,  15.0),
    ('+ Full retail all-in',        0.10,  20.0),
    ('+ Peak summer exposure',      0.13,  22.0),
]

_ANNUALIZE    = 52.18   # weeks/year (365.25 / 7)
_SIM_AVG_PRICE = 0.02   # approximate ERCOT January day-ahead average ($/kWh)


def write_sensitivity_table(
    sim_start_date: str,
    res_base: Dict,
    optimized_results: List,
    demand_rate: float,
    period_fraction: float,
    log_dir: str = 'logs',
) -> str:
    """Write a retail rate sensitivity table to CSV.

    The simulation runs against ERCOT wholesale prices. Real municipal utilities
    pay wholesale plus a retail adder (TDU delivery, REP margin, ancillary
    services). This table scales energy savings across four rate scenarios to
    show real-world dollar potential. Savings percentage is invariant to the
    adder; dollar figures and annualized projections scale proportionally.

    Args:
        sim_start_date:    ISO date string for the simulation start (used in filename).
        res_base:          Baseline simulation results dict from run_simulation().
        optimized_results: List of (label, results_dict) tuples for each strategy.
        demand_rate:       Demand charge rate used in simulation ($/kW/month).
        period_fraction:   Simulation duration as fraction of a billing month.
        log_dir:           Output directory (default: logs/).

    Returns:
        Absolute path to the written CSV file.
    """
    os.makedirs(log_dir, exist_ok=True)
    date_tag = sim_start_date[:10]
    filepath = os.path.join(log_dir, f"sensitivity_{date_tag}.csv")

    base_energy = sum(res_base[k] for k in ['gwtp', 'wwtp', 'ls'])

    header = ['scenario', 'energy_adder_per_kwh', 'effective_demand_rate_per_kw',
              'baseline_total_bill_usd']
    for lbl, _ in optimized_results:
        header += [
            f'{lbl}_total_bill_usd',
            f'{lbl}_savings_usd',
            f'{lbl}_savings_pct',
            f'{lbl}_annualized_savings_usd',
        ]

    rows = []
    for scenario_label, adder, dc_override in _RATE_SCENARIOS:
        dc_rate      = dc_override if dc_override is not None else demand_rate
        energy_scale = (_SIM_AVG_PRICE + adder) / _SIM_AVG_PRICE

        scaled_base_e  = base_energy * energy_scale
        scaled_base_dc = res_base['peak_kw'] * dc_rate * period_fraction
        scaled_base    = scaled_base_e + scaled_base_dc

        row = {
            'scenario':                     scenario_label,
            'energy_adder_per_kwh':         adder,
            'effective_demand_rate_per_kw': dc_rate,
            'baseline_total_bill_usd':      round(scaled_base, 2),
        }

        for lbl, res in optimized_results:
            opt_energy   = sum(res[k] for k in ['gwtp', 'wwtp', 'ls'])
            opt_dc       = res['peak_kw'] * dc_rate * period_fraction
            scaled_opt_e = opt_energy * energy_scale
            scaled_opt   = scaled_opt_e + opt_dc
            savings      = scaled_base - scaled_opt
            pct          = (savings / scaled_base * 100) if scaled_base else 0.0
            annual       = savings * _ANNUALIZE

            row[f'{lbl}_total_bill_usd']         = round(scaled_opt, 2)
            row[f'{lbl}_savings_usd']            = round(savings, 2)
            row[f'{lbl}_savings_pct']            = round(pct, 1)
            row[f'{lbl}_annualized_savings_usd'] = round(annual, 2)

        rows.append(row)

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Sensitivity table written -> {filepath}")
    return filepath

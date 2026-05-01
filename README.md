# Smart Scheduling and Demand Response for Water Utilities

**ENG 573 Capstone Project -- MEng Energy Systems**  
**University of Illinois Urbana-Champaign, Spring 2026**  
**Author:** Joseph Bailey  
**Advisor:** Dr. Rizwan Uddin, Department of Nuclear, Plasma, and Radiological Engineering

---

## Project Overview

This repository implements a Python-based optimization framework for municipal water and wastewater utility energy management. The framework models a three-facility system -- a groundwater treatment plant (GWTP), a wastewater treatment plant (WWTP), and a main influent lift station (LS) -- and schedules their energy-intensive equipment against real ERCOT day-ahead electricity price signals to reduce both energy costs and peak demand charges.

The core insight is the **Triple-Battery Concept**: each facility contains a physical storage buffer that can absorb or defer energy consumption without violating operational constraints. By treating these buffers as dispatchable "batteries," the framework applies demand response strategies that are typically reserved for dedicated energy storage systems.

Simulation results over a representative 7-day horizon show a **36.6% reduction in total electricity bill** and a **50.5% reduction in peak demand charges** compared to a hysteresis-based baseline, using ERCOT Houston Load Zone day-ahead prices.

---

## The Triple-Battery Concept

| Battery | Facility | Storage Medium | Control Variable |
|---|---|---|---|
| Elevation Battery | GWTP | Ground Storage Tank (80,000 gal) | Well pump + booster pump scheduling |
| Biological Battery | WWTP | Dissolved oxygen in aeration basin | Howden Roots blower duty cycle |
| Pipeline Battery | Lift Station | Wet well volume | Submersible pump sequencing |

Each battery has a charge/discharge asymmetry relative to electricity prices: the optimizer fills storage (charges) during low-price periods and draws it down (discharges) during high-price periods, shifting load without interrupting service.

---

## Architecture

The framework uses the **Strategy Pattern** to make the optimizer tier interchangeable. `main.py` (the Orchestration Layer) calls `solve()` on each facility optimizer without knowing whether the implementation is MILP, NMPC, or an RL agent. Swapping strategies requires only a command-line flag.

```
main.py  (Orchestration Layer -- receding-horizon simulation loop)
    |
    +-- modules/data_manager.py     (ERCOT price + demand forecasts)
    |
    +-- modules/gwtp.py             (GWTP Digital Twin -- Euler integration)
    +-- modules/wwtp.py             (WWTP Digital Twin -- DO mass balance ODE)
    +-- modules/lift_station.py     (Lift Station Digital Twin -- wet well mass balance)
    |
    +-- modules/optimizers/
            |
            +-- __init__.py         (Strategy factory -- create_optimizers())
            +-- interface.py        (Abstract Base Class -- solve() contract)
            |
            +-- milp/               (Tier 1 -- Pyomo / GLPK)
            |       base_milp.py
            |       gwtp_milp.py
            |       wwtp_milp.py
            |       ls_milp.py
            |
            +-- nmpc/               (Tier 2 -- do-mpc / CasADi / Ipopt)
            |       base_nmpc.py
            |       gwtp_nmpc.py
            |       wwtp_nmpc.py
            |       ls_nmpc.py
            |
            +-- ai/                 (Tier 3 -- Stable Baselines3, stub only)
                    base_agent.py
                    gwtp_agent.py
                    wwtp_agent.py
                    ls_agent.py
```

### Digital Twin Physics

Each facility physics module uses first-order Euler integration to advance state at each 15-minute timestep. WNTR is used in `gwtp.py` solely to construct the hydraulic network topology; no WNTR pressure-driven simulation is invoked. This choice is intentional: WNTR's iterative solver introduces latency incompatible with a 672-step receding-horizon loop.

### Optimizer Tiers

| Tier | Solver | Physics | Key Difference |
|---|---|---|---|
| MILP | Pyomo + GLPK | Linearized | DO gain fixed at nominal deficit; pump power linear in duty |
| NMPC | do-mpc + CasADi + Ipopt | Nonlinear | Exact bilinear DO dynamics; VFD affinity law (P proportional to n^3) |
| RL | Stable Baselines3 | N/A | Stub only -- not yet implemented |

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Optimization modeling | Pyomo | >= 6.6 |
| MILP solver | GLPK | system package |
| NLP solver | Ipopt | system package |
| NMPC framework | do-mpc | >= 4.6 |
| Automatic differentiation | CasADi | >= 3.6 |
| Hydraulic network modeling | WNTR | >= 1.2 |
| Market data | GridStatus.io | >= 0.20 |
| Demand charge data | OpenEI Utility Rate DB | REST API |
| Data handling | pandas, numpy | >= 2.0, >= 1.24 |

---

## Setup

### 1. System-Level Solvers

GLPK and Ipopt are not pip-installable and must be installed at the system level before running the simulation.

**Ubuntu / Debian:**
```bash
sudo apt install glpk-utils coinor-ipopt
```

**macOS (Homebrew):**
```bash
brew install glpk ipopt
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. API Keys

The simulation fetches live ERCOT day-ahead prices from GridStatus.io and demand charge rates from the OpenEI Utility Rate Database. Both calls fall back gracefully to synthetic data if keys are absent.

```bash
# GridStatus.io -- ERCOT day-ahead prices
# Register at: https://www.gridstatus.io
export GRIDSTATUS_API_KEY="your_key_here"

# OpenEI -- CenterPoint Energy demand charge rates
# Register at: https://openei.org/services/api
export OPENEI_API_KEY="your_key_here"
```

### 4. Configuration

Simulation parameters are controlled by YAML files in `config/`:

| File | Controls |
|---|---|
| `global_settings.yaml` | Simulation duration, timestep, start date, financial parameters, active strategy |
| `gwtp_config.yaml` | GST geometry, pump nameplate data, operating limits |
| `wwtp_config.yaml` | Blower specs, biology parameters (KLa, OUR, DO limits) |
| `ls_config.yaml` | Wet well geometry, submersible pump specs |

---

## Running the Simulation

From the repository root (the directory containing `main.py`):

```bash
# Run with the strategy set in global_settings.yaml (default: milp)
python main.py

# Run MILP optimizer explicitly
python main.py --strategy milp

# Run NMPC optimizer
python main.py --strategy nmpc

# Run both side-by-side (prints a 3-column audit table)
python main.py --strategy both
```

### Output

Each run produces:
- **Console:** Timestep snapshot logs every 32 steps + Triple-Battery audit table + solver timing table
- **`logs/simulation_<date>_<strategy>.csv`:** Per-timestep state, control actions, power draw, and cumulative costs
- **`logs/sensitivity_<date>.csv`:** Retail rate scaling analysis across four price scenarios

---

## Repository Structure

```
.
+-- main.py                         Orchestration layer and simulation loop
+-- requirements.txt                Python dependencies
+-- README.md                       This file
+-- .gitignore
|
+-- config/
|   +-- global_settings.yaml        Simulation parameters and financial config
|   +-- gwtp_config.yaml            GWTP asset specifications
|   +-- wwtp_config.yaml            WWTP asset specifications
|   +-- ls_config.yaml              Lift station asset specifications
|
+-- modules/
|   +-- data_manager.py             ERCOT price and demand data service
|   +-- gwtp.py                     GWTP digital twin (Euler integration)
|   +-- wwtp.py                     WWTP digital twin (DO ODE)
|   +-- lift_station.py             Lift station digital twin (wet well mass balance)
|   +-- utils.py                    Unit conversions, SimLogger, sensitivity writer
|   |
|   +-- optimizers/
|       +-- __init__.py             Strategy factory (create_optimizers)
|       +-- interface.py            Abstract base class (solve() contract)
|       |
|       +-- milp/                   MILP tier (Pyomo / GLPK)
|       |   +-- base_milp.py
|       |   +-- gwtp_milp.py
|       |   +-- wwtp_milp.py
|       |   +-- ls_milp.py
|       |
|       +-- nmpc/                   NMPC tier (do-mpc / CasADi / Ipopt)
|       |   +-- base_nmpc.py
|       |   +-- gwtp_nmpc.py
|       |   +-- wwtp_nmpc.py
|       |   +-- ls_nmpc.py
|       |
|       +-- ai/                     RL tier (Stable Baselines3 -- stub only)
|           +-- base_agent.py
|           +-- gwtp_agent.py
|           +-- wwtp_agent.py
|           +-- ls_agent.py
|
+-- logs/                           Simulation output (CSV dataset)
    +-- simulation_<date>_baseline.csv
    +-- simulation_<date>_milp.csv
    +-- simulation_<date>_nmpc.csv
    +-- sensitivity_<date>.csv
```

---

## Simulation Dataset

The `logs/` directory contains the complete CSV output from the four seasonal simulation runs (January, April, July, October 2025-2026) referenced in the final report. Each file captures 672 timesteps (7 days at 15-minute resolution) of per-facility state, control actions, power draw, and cumulative financials for all three strategies (baseline, MILP, NMPC).

---

## Known Limitations

- **WNTR hydraulic simulation not invoked.** GWTP state evolution uses Euler integration. See `gwtp.py` class docstring for rationale.
- **QSDsan removed from scope.** WWTP biology uses a first-order DO ODE in place of full ASM1 stoichiometry. See `wwtp.py` for rationale.
- **RL tier is a stub.** `modules/optimizers/ai/` files define the intended architecture but contain no trained policy.
- **MILP DO linearization.** `wwtp_milp.py` fixes the oxygen transfer deficit at a nominal 2.5 mg/L operating point. The NMPC tier uses the exact bilinear form.

---

## License

This repository is submitted in partial fulfillment of the requirements for the Master of Engineering in Energy Systems degree at the University of Illinois Urbana-Champaign. All rights reserved by the author.

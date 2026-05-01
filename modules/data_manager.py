"""
===============================================================================
Module:       data_manager.py
Project:      Smart Scheduling and Demand Response for Water Utilities
Context:      UIUC ENG 573 Capstone Project for MEng Energy Systems
Author:       Joseph Bailey
Advisor:      Dr. Rizwan Uddin
Created:      Spring 2026

Description:
    Central data service providing boundary conditions for the simulation.
    Integrates with GridStatus.io to fetch real-time and historical ERCOT
    day-ahead market data for the Houston Load Zone (LZ_HOUSTON), and with
    the OpenEI Utility Rate Database to retrieve the applicable CenterPoint
    Energy demand charge rate.

    Key Logic:
        1. API Ingestion: Authenticated fetch from GridStatusIO for ERCOT
           day-ahead hourly prices (ercot_spp_day_ahead_hourly dataset).
        2. Time Alignment: UTC-to-Central (CST) conversion for Texas market
           timestamps before resampling to 15-minute resolution.
        3. Unit Correction: $/MWh to $/kWh conversion for MILP cost functions.
        4. Synthetic Fallback: Sinusoidal duck-curve generator for offline
           testing when API keys are absent or the API is unreachable.

Inputs:
    - Global Configuration dict (simulation_days, timestep_minutes, start_date)
    - Environment Variables: GRIDSTATUS_API_KEY, OPENEI_API_KEY
Outputs:
    - Forecast DataFrame with columns: 'price_kwh', 'demand_gpm'
Dependencies:
    - gridstatusio, pandas, numpy, requests
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import logging
import requests
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta
from typing import Dict


class DataManager:
    """
    Boundary condition provider for price and demand forecasts.

    Fetches ERCOT day-ahead prices and OpenEI demand charge rates on
    initialization. Falls back to synthetic data if API keys are absent
    or requests fail, so the simulation can run fully offline.
    """

    def __init__(self, config: Dict):
        """
        Initialize the DataManager and pre-fetch the demand charge rate.

        Constructs the simulation time index from config, then calls
        fetch_demand_charge_rate() once so main.py has a consistent rate
        available before the simulation loop begins.

        Args:
            config: Dict with keys:
                simulation_days      -- number of days to simulate
                timestep_minutes     -- resolution in minutes (typically 15)
                start_date           -- ISO date string ('YYYY-MM-DD')
                demand_charge_per_kw -- fallback demand rate ($/kW/month)
        """
        self.days      = config.get('simulation_days', 1)
        self.dt        = config.get('timestep_minutes', 60)
        self.freq      = f'{self.dt}min'
        self.ercot_node = 'LZ_HOUSTON'
        self.api_key   = os.getenv('GRIDSTATUS_API_KEY')

        start_date = config.get('start_date', '2026-01-01')
        self.time_index = pd.date_range(
            start=start_date,
            periods=(self.days * 24 * 60) // self.dt,
            freq=self.freq
        )
        logging.info(f"DataManager initialized for {self.days} days from {start_date}.")

        self.demand_charge_rate = self.fetch_demand_charge_rate(
            fallback_rate=config.get('demand_charge_per_kw', 15.0)
        )

    def fetch_demand_charge_rate(self, fallback_rate: float = 15.0) -> float:
        """
        Fetch the current demand charge rate ($/kW/month) from the OpenEI
        Utility Rate Database for CenterPoint Energy Houston Electric
        (EIAID 4716), the TDSP serving the ERCOT Houston load zone.

        I query flatdemandstructure first because that is the populated field
        for CenterPoint in OpenEI; demandratestructure is typically empty for
        this utility. Non-operational tariffs (backup, standby, emergency) are
        excluded to ensure only standard service tariffs are considered.

        Falls back to fallback_rate if the OPENEI_API_KEY environment variable
        is absent, the request times out, or no demand charge is found in the
        returned tariff structures.

        API key registration: https://openei.org/services/api
        Set environment variable: OPENEI_API_KEY

        Args:
            fallback_rate: Rate to use if the API call fails ($/kW/month).
                           Sourced from global_settings.yaml (default: $15.00).

        Returns:
            Demand charge rate in $/kW/month.
        """
        api_key = os.getenv('OPENEI_API_KEY')
        if not api_key:
            logging.warning(
                f"OPENEI_API_KEY not set. Using fallback demand charge rate "
                f"of ${fallback_rate}/kW/month from config."
            )
            return fallback_rate

        try:
            url = 'https://api.openei.org/utility_rates'
            params = {
                'api_key':  api_key,
                'version':  7,
                'format':   'json',
                'eiaid':    4716,
                'sector':   'Commercial',
                'limit':    5,
                'detail':   'full',
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            tariffs = data.get('items', [])
            if not tariffs:
                logging.warning(
                    f"OpenEI returned no tariffs for EIAID 4716. "
                    f"Using fallback rate of ${fallback_rate}/kW/month."
                )
                return fallback_rate

            excluded_keywords = ['backup', 'maintenance', 'standby', 'emergency']

            for tariff in tariffs:
                tariff_name = tariff.get('name', '').lower()
                if any(kw in tariff_name for kw in excluded_keywords):
                    logging.info(f"OpenEI skipping non-applicable tariff: '{tariff.get('name')}'")
                    continue

                # Check flatdemandstructure first -- CenterPoint's populated field.
                flat_units     = tariff.get('flatdemandunit', '')
                flat_structure = tariff.get('flatdemandstructure', [])
                if 'kW' in flat_units and flat_structure:
                    for tier_list in flat_structure:
                        for tier in tier_list:
                            rate = tier.get('rate')
                            if rate is not None and float(rate) > 0:
                                logging.info(
                                    f"OpenEI demand charge found: ${rate}/kW/month "
                                    f"from tariff '{tariff.get('name', 'unknown')}'"
                                )
                                return float(rate)

                # Fall through to demandratestructure if flatdemand is unpopulated.
                demand_units     = tariff.get('demandrateunit', '')
                demand_structure = tariff.get('demandratestructure', [])
                if 'kW' in demand_units and demand_structure:
                    for tier_list in demand_structure:
                        for tier in tier_list:
                            rate = tier.get('rate')
                            if rate is not None and float(rate) > 0:
                                logging.info(
                                    f"OpenEI demand charge found: ${rate}/kW/month "
                                    f"from tariff '{tariff.get('name', 'unknown')}'"
                                )
                                return float(rate)

            logging.warning(
                f"OpenEI tariffs found but no demand charge extracted. "
                f"Using fallback rate of ${fallback_rate}/kW/month."
            )
            return fallback_rate

        except requests.exceptions.Timeout:
            logging.warning(
                f"OpenEI API request timed out. Using fallback rate of "
                f"${fallback_rate}/kW/month."
            )
            return fallback_rate

        except Exception as e:
            logging.error(
                f"OpenEI API error: {e}. Using fallback rate of "
                f"${fallback_rate}/kW/month."
            )
            return fallback_rate

    def fetch_real_ercot_prices(self) -> pd.Series:
        """
        Fetch ERCOT day-ahead hourly prices from GridStatus.io and resample
        to the simulation timestep resolution.

        Converts UTC timestamps to US/Central before resampling so price
        intervals align with local Texas operating hours. Prices are converted
        from $/MWh to $/kWh. Falls back to synthetic data if the API key is
        absent or the request fails.

        Returns:
            pd.Series indexed to self.time_index with name 'price_kwh'.
        """
        if not self.api_key:
            logging.error("GRIDSTATUS_API_KEY not found. Using synthetic prices.")
            return self._generate_electricity_prices()

        try:
            client     = GridStatusClient(api_key=self.api_key)
            start_date = self.time_index[0].strftime('%Y-%m-%d')
            end_date   = (self.time_index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')

            df = client.get_dataset(
                dataset='ercot_spp_day_ahead_hourly',
                start=start_date,
                end=end_date,
                filter_column='location',
                filter_value=self.ercot_node,
            )

            # Convert UTC to Central and drop timezone info for pandas compatibility.
            df['Time'] = (
                pd.to_datetime(df['interval_start_utc'])
                .dt.tz_convert('US/Central')
                .dt.tz_localize(None)
            )
            df = df.set_index('Time').sort_index()

            price_col    = 'spp' if 'spp' in df.columns else 'lmp'
            df_resampled = (df[price_col] / 1000.0).resample(self.freq).ffill()

            prices_kwh      = df_resampled.reindex(self.time_index, method='ffill')
            prices_kwh.name = 'price_kwh'

            logging.info(f"Successfully retrieved {len(prices_kwh)} ERCOT price points.")
            return prices_kwh

        except Exception as e:
            logging.error(f"GridStatusIO fetch error: {e}. Falling back to synthetic prices.")
            return self._generate_electricity_prices()

    def _generate_electricity_prices(self, base_price: float = 0.10) -> pd.Series:
        """
        Generate a synthetic sinusoidal electricity price profile.

        Used as a fallback when the GridStatus.io API is unavailable. The
        sinusoid approximates a mild duck-curve shape (higher prices in the
        morning and evening, lower overnight) without the volatility spikes
        present in real ERCOT data.

        Args:
            base_price: Midpoint price in $/kWh (default: $0.10/kWh).

        Returns:
            pd.Series indexed to self.time_index with name 'price_kwh'.
        """
        hours = self.time_index.hour + self.time_index.minute / 60.0
        price = base_price + 0.02 * np.sin(2 * np.pi * (hours - 6) / 24)
        return pd.Series(data=np.maximum(price, 0.01), index=self.time_index, name='price_kwh')

    def _generate_potable_demand(self, base_demand: float = 250.0) -> pd.Series:
        """
        Generate a synthetic potable water demand profile (GPM).

        Models two daily demand peaks using Gaussian kernels: a morning peak
        (7 AM, sigma=1.5 hr) representing residential morning use, and an
        evening peak (7 PM, sigma=2.0 hr) representing cooking and irrigation.
        This shape is consistent with published AWWA residential demand patterns.

        Args:
            base_demand: Baseline flow in GPM (default: 250 GPM).

        Returns:
            pd.Series indexed to self.time_index with name 'demand_gpm'.
        """
        hours   = self.time_index.hour + self.time_index.minute / 60.0
        morning = 0.5 * base_demand * np.exp(-((hours - 7) ** 2) / (2 * 1.5 ** 2))
        evening = 0.3 * base_demand * np.exp(-((hours - 19) ** 2) / (2 * 2.0 ** 2))
        return pd.Series(data=(base_demand + morning + evening), index=self.time_index, name='demand_gpm')

    def get_forecast(self, system_type: str, use_real_ercot: bool = False) -> pd.DataFrame:
        """
        Return a combined price and demand forecast DataFrame for the simulation.

        Args:
            system_type:    Facility type selector. Currently only 'groundwater'
                            is implemented; other types return an empty DataFrame.
            use_real_ercot: If True, fetch live ERCOT prices via GridStatus.io.
                            If False, use the synthetic sinusoidal fallback.

        Returns:
            DataFrame with columns 'price_kwh' and 'demand_gpm', indexed to
            self.time_index at the configured timestep resolution.
        """
        prices = self.fetch_real_ercot_prices() if use_real_ercot else self._generate_electricity_prices()
        if system_type == 'groundwater':
            load = self._generate_potable_demand()
            return pd.concat([prices, load], axis=1)
        return pd.DataFrame()

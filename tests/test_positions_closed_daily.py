# Copyright 2018 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run: python3 -m unittest discover -s tests/ -p test_*.py -t . -v

import os
import unittest
from unittest.mock import patch
import glob
import pandas as pd
from moonshot import Moonshot
from moonshot.cache import TMP_DIR

class PositionsClosedDailyTestCase(unittest.TestCase):

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    def test_positions_closed_daily(self):
        """
        Tests that the resulting DataFrames are correct after running a
        short-only intraday strategy with POSITIONS_CLOSED_DAILY = True.
        """

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            POSITIONS_CLOSED_DAILY = True
            SLIPPAGE_BPS = 10

            def prices_to_signals(self, prices):
                morning_prices = prices.loc["Open"].xs("09:30:00", level="Time")
                short_signals = morning_prices > 10
                return -short_signals.astype(int)

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def target_weights_to_positions(self, weights, prices):
                # enter on same day
                positions = weights.copy()
                return positions

            def positions_to_gross_returns(self, positions, prices):
                # hold from 10:00-16:00
                closes = prices.loc["Close"]
                entry_prices = closes.xs("09:30:00", level="Time")
                exit_prices = closes.xs("15:30:00", level="Time")
                pct_changes = (exit_prices - entry_prices) / entry_prices
                gross_returns = pct_changes * positions
                return gross_returns

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03"])
            fields = ["Close","Open"]
            times = ["09:30:00", "15:30:00"]
            idx = pd.MultiIndex.from_product(
                [fields, dt_idx, times], names=["Field", "Date", "Time"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9.6,
                        10.45,
                        10.12,
                        15.45,
                        8.67,
                        12.30,
                        # Open
                        9.88,
                        10.34,
                        10.23,
                        16.45,
                        8.90,
                        11.30,
                    ],
                    23456: [
                        # Close
                        10.56,
                        12.01,
                        10.50,
                        9.80,
                        13.40,
                        14.50,
                        # Open
                        9.89,
                        11,
                        8.50,
                        10.50,
                        14.10,
                        15.60
                    ],
                 },
                index=idx
            )

            return prices

        def mock_get_db_config(db):
            return {
                'vendor': 'ib',
                'domain': 'main',
                'bar_size': '1 day'
            }

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
                        "DEF",
                        "STK",
                        "USD",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "ConId"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                    results = ShortAbove10Intraday().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'TotalHoldings',
             'Turnover',
             'AbsWeight',
             'Weight'}
        )

        # replace nan with "nan" to allow equality comparisons
        results = results.round(7)
        results = results.where(results.notnull(), "nan")

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -1.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     -1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.25,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.25]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.25,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.25]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     0.5,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.5]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     0.0005,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0005]}
        )

    def test_positions_not_closed_daily(self):
        """
        Tests that the resulting DataFrames are correct after running a
        short-only intraday strategy with POSITIONS_CLOSED_DAILY = False.
        """

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            # POSITIONS_CLOSED_DAILY defaults to False
            SLIPPAGE_BPS = 10

            def prices_to_signals(self, prices):
                morning_prices = prices.loc["Open"].xs("09:30:00", level="Time")
                short_signals = morning_prices > 10
                return -short_signals.astype(int)

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def target_weights_to_positions(self, weights, prices):
                # enter on same day
                positions = weights.copy()
                return positions

            def positions_to_gross_returns(self, positions, prices):
                # hold from 10:00-16:00
                closes = prices.loc["Close"]
                entry_prices = closes.xs("09:30:00", level="Time")
                exit_prices = closes.xs("15:30:00", level="Time")
                pct_changes = (exit_prices - entry_prices) / entry_prices
                gross_returns = pct_changes * positions
                return gross_returns

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03"])
            fields = ["Close","Open"]
            times = ["09:30:00", "15:30:00"]
            idx = pd.MultiIndex.from_product(
                [fields, dt_idx, times], names=["Field", "Date", "Time"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9.6,
                        10.45,
                        10.12,
                        15.45,
                        8.67,
                        12.30,
                        # Open
                        9.88,
                        10.34,
                        10.23,
                        16.45,
                        8.90,
                        11.30,
                    ],
                    23456: [
                        # Close
                        10.56,
                        12.01,
                        10.50,
                        9.80,
                        13.40,
                        14.50,
                        # Open
                        9.89,
                        11,
                        8.50,
                        10.50,
                        14.10,
                        15.60
                    ],
                 },
                index=idx
            )

            return prices

        def mock_get_db_config(db):
            return {
                'vendor': 'ib',
                'domain': 'main',
                'bar_size': '1 day'
            }

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
                        "DEF",
                        "STK",
                        "USD",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "ConId"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                    results = ShortAbove10Intraday().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'TotalHoldings',
             'Turnover',
             'AbsWeight',
             'Weight'}
        )

        # replace nan with "nan" to allow equality comparisons
        results = results.round(7)
        results = results.where(results.notnull(), "nan")

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -1.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     -1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.25,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.25]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.25,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.25]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: ["nan",
                     0.25,
                     0.25],
             23456: ["nan",
                     0.0,
                     0.25]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     0.00025,
                     0.00025],
             23456: [0.0,
                     0.0,
                     0.00025]}
        )


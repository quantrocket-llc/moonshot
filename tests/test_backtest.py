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

class BacktestTestCase(unittest.TestCase):

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    def test_complain_if_prices_to_signals_not_implemented(self):
        """
        Tests error handling when prices_to_signals hasn't been implemented.
        """

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        # Volume
                        5000,
                        16000,
                        8800
                    ],
                    23456: [
                        # Close
                        12,
                        11,
                        8.50,
                        # Volume
                        15000,
                        14000,
                        28800
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

                    with self.assertRaises(NotImplementedError) as cm:
                        Moonshot().backtest()

        self.assertIn("strategies must implement prices_to_signals", repr(cm.exception))

    def test_basic_long_only_strategy(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        long-only strategy that largely relies on the default methods.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                        # Volume
                        5000,
                        16000,
                        8800,
                        9900
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                        # Volume
                        15000,
                        14000,
                        28800,
                        17000

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

                    results = BuyBelow10().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.0,
                     0.0,
                     0.0,
                     1.0],
             23456: [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.5,
                     0.0,
                     0.0,
                     1.0],
             23456: [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.5,
                     0.0,
                     0.0,
                     1.0],
             23456: [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     0.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     0.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0,
                     1.0,
                     0,
                     0],
             23456: [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     0.5,
                     0.0],
             23456: ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             23456: [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     0.0]}
        )

    def test_basic_long_short_strategy(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        long-short strategy that largely relies on the default methods.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                        # Volume
                        5000,
                        16000,
                        8800,
                        9900
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                        # Volume
                        15000,
                        14000,
                        28800,
                        17000

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

                    results = BuyBelow10ShortAbove10().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0,
                     1.0],
             23456: [1.0,
                     -1.0,
                     1.0,
                     -1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.5,
                     -0.5,
                     -0.5,
                     0.5],
             23456: [0.5,
                     -0.5,
                     0.5,
                     -0.5]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.5,
                     0.5,
                     0.5,
                     0.5],
             23456: [0.5,
                     0.5,
                     0.5,
                     0.5]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     -0.5,
                     -0.5],
             23456: ["nan",
                     0.5,
                     -0.5,
                     0.5]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     0.5,
                     0.5],
             23456: ["nan",
                     0.5,
                     0.5,
                     0.5]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0,
                     1.0,
                     1.0,
                     1.0],
             23456: [0,
                     1.0,
                     1.0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     1.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     1.0,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     0.0242857], # (9.99 - 10.50)/10.50 * -0.5
             23456: [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     -0.1176471] # (10.50 - 8.50)/8.50 * -0.5
             }
        )

    def test_long_short_strategy_override_methods(self):
        """
        Tests that the resulting DataFrames are correct after running a
        long-short strategy that overrides the major backtesting methods.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10 and holds overnight.
            """

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def target_weights_to_positions(self, weights, prices):
                # enter on close same day
                positions = weights.copy()
                return positions

            def positions_to_gross_returns(self, positions, prices):
                # hold on close till next day open
                closes = prices.loc["Close"]
                opens = prices.loc["Open"]
                pct_changes = (opens - closes.shift()) / closes.shift()
                gross_returns = pct_changes * positions.shift()
                return gross_returns

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Open","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9.6,
                        10.45,
                        10.23,
                        8.67,
                        # Open
                        9,
                        11,
                        10.50,
                        9.99,
                        # Volume
                        5000,
                        16000,
                        8800,
                        9900
                    ],
                    23456: [
                        # Close
                        10.56,
                        12.01,
                        10.50,
                        9.80,
                        # Open
                        9.89,
                        11,
                        8.50,
                        10.50,
                        # Volume
                        15000,
                        14000,
                        28800,
                        17000

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

                    results = BuyBelow10ShortAbove10Overnight().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0,
                     1.0],
             23456: [1.0,
                     -1.0,
                     1.0,
                     -1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.25,
                     -0.25,
                     -0.25,
                     0.25],
             23456: [0.25,
                     -0.25,
                     0.25,
                     -0.25]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.25,
                     0.25,
                     0.25,
                     0.25],
             23456: [0.25,
                     0.25,
                     0.25,
                     0.25]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.25,
                     -0.25,
                     -0.25,
                     0.25],
             23456: [0.25,
                     -0.25,
                     0.25,
                     -0.25]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.25,
                     0.25,
                     0.25,
                     0.25],
             23456: [0.25,
                     0.25,
                     0.25,
                     0.25]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.0,
                     1.0,
                     1.0,
                     1.0],
             23456: [1.0,
                     1.0,
                     1.0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     0.0,
                     0.5],
             23456: ["nan",
                     0.5,
                     0.5,
                     0.5]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0364583, # (11 - 9.6)/9.6 * 0.25
                     -0.0011962, # (10.50 - 10.45)/10.45 * -0.25
                     0.0058651], # (9.99 - 10.23)/10.23 * 0.25
             23456: [0.0,
                     0.0104167,# (11 - 10.56)/10.56 * 0.25
                     0.0730641, # (8.50 - 12.01)/12.01 * -0.25
                     0.0] # (10.50 - 10.50)/10.50 * 0.25
             }
        )

    def test_short_only_once_a_day_intraday_strategy(self):
        """
        Tests that the resulting DataFrames are correct after running a
        short-only intraday strategy.
        """

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            POSITIONS_CLOSED_DAILY = True

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

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     0.25,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.25]}
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

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     0.25,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.25]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     1.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     1.0]}
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

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0]}
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
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.13167, # (15.45 - 10.12)/10.12 * -0.25
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.0205224] # (14.50 - 13.40)/13.40 * 0.25
             }
        )

    def test_continuous_intraday_strategy(self):

        """
        Tests that the resulting DataFrames are correct after running a
        long-short continuous intraday strategy.
        """

        class BuyBelow10ShortAbove10ContIntraday(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02"])
            fields = ["Close"]
            times = ["10:00:00", "11:00:00", "12:00:00"]
            idx = pd.MultiIndex.from_product([fields, dt_idx, times], names=["Field", "Date", "Time"])

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
                    ],
                    23456: [
                        # Close
                        10.56,
                        12.01,
                        10.50,
                        9.80,
                        13.40,
                        7.50,
                    ],
                 },
                index=idx
            )

            return prices

        def mock_get_db_config(db):
            return {
                'vendor': 'ib',
                'domain': 'main',
                'bar_size': '1 hour'
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

                    results = BuyBelow10ShortAbove10ContIntraday().backtest()

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
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0,
                     -1.0,
                     1.0,
                     -1.0],
             23456: [-1.0,
                     -1.0,
                     -1.0,
                     1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [0.5,
                     -0.5,
                     -0.5,
                     -0.5,
                     0.5,
                     -0.5],
             23456: [-0.5,
                     -0.5,
                     -0.5,
                     0.5,
                     -0.5,
                     0.5]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5],
             23456: [0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5]}
        )


        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: ['nan',
                     0.5,
                     -0.5,
                     -0.5,
                     -0.5,
                     0.5],
             23456: ['nan',
                     -0.5,
                     -0.5,
                     -0.5,
                     0.5,
                     -0.5]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: ['nan',
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5],
             23456: ['nan',
                     0.5,
                     0.5,
                     0.5,
                     0.5,
                     0.5]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [0,
                     1.0,
                     1.0,
                     1.0,
                     1.0,
                     1.0],
             23456: [0,
                     1.0,
                     1.0,
                     1.0,
                     1.0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: ['nan',
                     0.5,
                     1.0,
                     0.0,
                     0.0,
                     1.0],
             23456: ['nan',
                     0.5,
                     0.0,
                     0.0,
                     1.0,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"].reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-02T00:00:00'],
             'Time': ['10:00:00',
                      '11:00:00',
                      '12:00:00',
                      '10:00:00',
                      '11:00:00',
                      '12:00:00'],
             12345: [0.0,
                     0.0,
                     -0.0157895, # (10.12-10.45)/10.45 * 0.5
                     -0.2633399, # (15.45-10.12)/10.12 * -0.5
                     0.2194175,  # (8.67-15.45)/15.45 * -0.5
                     -0.2093426  # (12.30-8.67)/8.67 * -0.5
                     ],
             23456: [0.0,
                     0.0,
                     0.0628643, # (10.50-12.01)/12.01 * -0.5
                     0.0333333, # (9.80-10.50)/10.50 * -0.5
                     -0.1836735, # (13.40-9.80)/9.80 * -0.5
                     -0.2201493 # (7.50-13.40)/13.40 * 0.5
                     ]}
        )

    def test_pass_allocation(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        long-short strategy and passing an allocation.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                        # Volume
                        5000,
                        16000,
                        8800,
                        9900
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                        # Volume
                        15000,
                        14000,
                        28800,
                        17000

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

                    results = BuyBelow10ShortAbove10().backtest(allocation=3.0)

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0,
                     1.0],
             23456: [1.0,
                     -1.0,
                     1.0,
                     -1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.5,
                     -1.5,
                     -1.5,
                     1.5],
             23456: [1.5,
                     -1.5,
                     1.5,
                     -1.5]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [1.5,
                     1.5,
                     1.5,
                     1.5],
             23456: [1.5,
                     1.5,
                     1.5,
                     1.5]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     1.5,
                     -1.5,
                     -1.5],
             23456: ["nan",
                     1.5,
                     -1.5,
                     1.5]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     1.5,
                     1.5,
                     1.5],
             23456: ["nan",
                     1.5,
                     1.5,
                     1.5]}
        )
        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0,
                     1.0,
                     1.0,
                     1.0],
             23456: [0,
                     1.0,
                     1.0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     1.5,
                     3.0,
                     0.0],
             23456: ["nan",
                     1.5,
                     3.0,
                     3.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: [0.0,
                     0.0,
                     -0.0681818, # (10.50 - 11)/11 * 1.5
                     0.0728571], # (9.99 - 10.50)/10.50 * -1.5
             23456: [0.0,
                     0.0,
                     -0.3409091, # (8.50 - 11)/11 * 1.5
                     -0.3529412] # (10.50 - 8.50)/8.50 * -1.5
             }
        )

    def test_label_conids(self):
        """
        Tests that the label_conids param causes symbols to be included in
        the resulting columns. For forex, symbol.currency should be used as
        the label.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                        # Volume
                        5000,
                        16000,
                        8800,
                        9900
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                        # Volume
                        15000,
                        14000,
                        28800,
                        17000

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
                        "AAPL",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
                        "EUR",
                        "CASH",
                        "JPY",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "ConId"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        # control: run without label_conids
        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                    results = BuyBelow10().backtest()

        self.assertSetEqual(
            set(results.columns),
            {12345,
             23456}
        )

        # control: run with label_conids
        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                    results = BuyBelow10().backtest(label_conids=True)

        self.assertSetEqual(
            set(results.columns),
            {"AAPL(12345)",
             "EUR.JPY(23456)"}
        )

    def test_truncate_at_start_date(self):
        """
        Tests that the resulting DataFrames are truncated at the requested
        start date even if the data predates the start date due to lookback
        window.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            LOOKBACK_WINDOW = 10 # Due to mock, this isn't actually having any effect

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                        # Volume
                        5000,
                        16000,
                        8800,
                        9900
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                        # Volume
                        15000,
                        14000,
                        28800,
                        17000

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

                    results = BuyBelow10().backtest(start_date="2018-05-03")

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

        self.assertEqual(results.index.get_level_values("Date").min(), pd.Timestamp("2018-05-03"))

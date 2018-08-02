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
from moonshot.slippage import FixedSlippage
from moonshot.cache import TMP_DIR

class MoonshotSlippgeTestCase(unittest.TestCase):
    """
    Test cases related to applying slippage in a backtest.
    """

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    def test_no_slippage(self):
        """
        Tests that the resulting DataFrames are correct when no slippage is
        applied.
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

            master_fields = ["Timezone"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York"
                    ],
                    23456: [
                        "America/New_York"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow10ShortAbove10().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'Trade',
             'AbsWeight',
             'Weight'}
        )

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

        net_exposures = results.loc["NetExposure"].reset_index()
        net_exposures.loc[:, "Date"] = net_exposures.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_exposures.to_dict(orient="list"),
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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     -1.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     -1.0,
                     1.0]}
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

    def test_apply_slippage(self):
        """
        Tests that the resulting DataFrames are correct when a single
        slippage class is applied.
        """

        class TestSlippage(FixedSlippage):

            ONE_WAY_SLIPPAGE = 0.001 # 10 BPS

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """

            SLIPPAGE_CLASSES = TestSlippage

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

            master_fields = ["Timezone"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York"
                    ],
                    23456: [
                        "America/New_York"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow10ShortAbove10().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'Trade',
             'AbsWeight',
             'Weight'}
        )

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

        net_exposures = results.loc["NetExposure"].reset_index()
        net_exposures.loc[:, "Date"] = net_exposures.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_exposures.to_dict(orient="list"),
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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     -1.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     -1.0,
                     1.0]}
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
                     0.0005,
                     0.001,
                     0.0],
             23456: [0.0,
                     0.0005,
                     0.001,
                     0.001]}
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
                     -0.0005,
                     -0.0237273, # (10.50 - 11)/11 * 0.5 - 0.001
                     0.0242857], # (9.99 - 10.50)/10.50 * -0.5
             23456: [0.0,
                     -0.0005,
                     -0.1146364, # (8.50 - 11)/11 * 0.5 - 0.001
                     -0.1186471] # (10.50 - 8.50)/8.50 * -0.5 - 0.001
             }
        )

    def test_apply_SLIPPAGE_BPS(self):
        """
        Tests that the resulting DataFrames are correct when SLIPPAGE_BPS is
        applied.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """

            SLIPPAGE_BPS = 20

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

            master_fields = ["Timezone"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York"
                    ],
                    23456: [
                        "America/New_York"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow10ShortAbove10().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'Trade',
             'AbsWeight',
             'Weight'}
        )

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

        net_exposures = results.loc["NetExposure"].reset_index()
        net_exposures.loc[:, "Date"] = net_exposures.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_exposures.to_dict(orient="list"),
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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     -1.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     -1.0,
                     1.0]}
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
                     0.001,
                     0.002,
                     0.0],
             23456: [0.0,
                     0.001,
                     0.002,
                     0.002]}
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
                     -0.001,
                     -0.0247273, # (10.50 - 11)/11 * 0.5 - 0.002
                     0.0242857], # (9.99 - 10.50)/10.50 * -0.5
             23456: [0.0,
                     -0.001,
                     -0.1156364, # (8.50 - 11)/11 * 0.5 - 0.002
                     -0.1196471] # (10.50 - 8.50)/8.50 * -0.5 - 0.002
             }
        )

    def test_apply_mulitple_slippages(self):
        """
        Tests that the resulting DataFrames are correct when multiple
        slippage classes and SLIPPAGE_BPS are applied.
        """

        class TestSlippage1(FixedSlippage):

            ONE_WAY_SLIPPAGE = 0.003 # 30 BPS

        class TestSlippage2(FixedSlippage):

            ONE_WAY_SLIPPAGE = 0.002 # 20 BPS

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """

            SLIPPAGE_CLASSES = (TestSlippage1, TestSlippage2)
            SLIPPAGE_BPS = 50

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

            master_fields = ["Timezone"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York"
                    ],
                    23456: [
                        "America/New_York"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow10ShortAbove10().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'Trade',
             'AbsWeight',
             'Weight'}
        )

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

        net_exposures = results.loc["NetExposure"].reset_index()
        net_exposures.loc[:, "Date"] = net_exposures.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_exposures.to_dict(orient="list"),
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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.5,
                     -1.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     -1.0,
                     1.0]}
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
                     0.005,
                     0.01,
                     0.0],
             23456: [0.0,
                     0.005,
                     0.01,
                     0.01]}
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
                     -0.005,
                     -0.0327273, # (10.50 - 11)/11 * 0.5 - 0.01
                     0.0242857], # (9.99 - 10.50)/10.50 * -0.5
             23456: [0.0,
                     -0.005,
                     -0.1236364, # (8.50 - 11)/11 * 0.5 - 0.001
                     -0.1276471] # (10.50 - 8.50)/8.50 * -0.5 - 0.01
             }
        )

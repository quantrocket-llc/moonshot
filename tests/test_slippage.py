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

    def test_apply_slippage_continuous_intraday(self):
        """
        Tests that the resulting DataFrames are correct when a single
        slippage class is applied on a continuous intraday strategy.
        """

        class TestSlippage(FixedSlippage):

            ONE_WAY_SLIPPAGE = 0.001 # 10 BPS

        class BuyBelow10ShortAbove10ContIntraday(Moonshot):
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
                     0.0005,
                     0.001,
                     0.0,
                     0.0,
                     0.001],
             23456: [0.0,
                     0.0005,
                     0.0,
                     0.0,
                     0.001,
                     0.001]}
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
                     -0.0005,
                     -0.0167895, # (10.12-10.45)/10.45 * 0.5 - 0.001
                     -0.2633399, # (15.45-10.12)/10.12 * -0.5
                     0.2194175,  # (8.67-15.45)/15.45 * -0.5
                     -0.2103426  # (12.30-8.67)/8.67 * -0.5 - 0.001
                     ],
             23456: [0.0,
                     -0.0005,
                     0.0628643, # (10.50-12.01)/12.01 * -0.5
                     0.0333333, # (9.80-10.50)/10.50 * -0.5
                     -0.1846735, # (13.40-9.80)/9.80 * -0.5 - 0.001
                     -0.2211493 # (7.50-13.40)/13.40 * 0.5 - 0.001
                     ]}
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

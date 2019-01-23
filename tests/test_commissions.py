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
from moonshot.commission import PercentageCommission, FuturesCommission
from moonshot.exceptions import MoonshotParameterError

class MoonshotCommissionsTestCase(unittest.TestCase):
    """
    Test cases related to applying commissions in a backtest.
    """

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    def test_no_commission(self):
        """
        Tests that the resulting DataFrames are correct when no commissions
        are applied.
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
             'TotalHoldings',
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

    def test_apply_commissions_eod_no_nlv(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied and no NLV is specified.
        """
        class TestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS
            EXCHANGE_FEE_RATE = 0
            MIN_COMMISSION = 800000000 # set high to verify ignored

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            COMMISSION_CLASS = TestCommission

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

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        1,
                        1
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        None,
                        None
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
             'TotalHoldings',
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

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.00005,
                     0.0001,
                     0.0],
             23456: ["nan",
                     0.00005,
                     0.0001,
                     0.0001]}
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
             12345: ["nan",
                     -0.00005,
                     -0.0228273, # (10.50 - 11)/11 * 0.5 - 0.0001
                     0.0242857], # (9.99 - 10.50)/10.50 * -0.5
             23456: ["nan",
                     -0.00005,
                     -0.1137364, # (8.50 - 11)/11 * 0.5
                     -0.1177471] # (10.50 - 8.50)/8.50 * -0.5
             }
        )

    def test_apply_commissions_eod_with_nlv(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied and NLV is specified.
        """
        class TestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS
            EXCHANGE_FEE_RATE = 0
            MIN_COMMISSION = 500

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            COMMISSION_CLASS = TestCommission

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                self.save_to_results("Nlv", prices.loc["Nlv"])
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

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        1,
                        1,
                        "USD"
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        None,
                        None,
                        "USD"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow10ShortAbove10().backtest(nlv={"USD":50000})

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'TotalHoldings',
             'Trade',
             'AbsWeight',
             'Weight',
             'Nlv'}
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

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00'],
             12345: [50000],
             23456: [50000]}
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
             12345: ["nan",
                     0.01,
                     0.01,
                     0.0],
             23456: ["nan",
                     0.01,
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
             12345: ["nan",
                     -0.01,
                     -0.03272727272727271, # (10.50 - 11)/11 * 0.5 - 0.01
                     0.0242857142857143], # (9.99 - 10.50)/10.50 * -0.5 - 0.01
             23456: ["nan",
                     -0.01,
                     -0.12363636363636364, # (8.50 - 11)/11 * 0.5 - 0.01
                     -0.12764705882352945] # (10.50 - 8.50)/8.50 * -0.5 - 0.01
             }
        )

    def test_commissions_by_exchange_sectype_currency_complain_if_missing(self):
        """
        Tests error handling when commissions are specified per sec
        type/exchange/currency but not all required sec
        type/exchanges/currencies are provided.
        """
        class TsejTestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS

        class OseTestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0002 # 2 BPS

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            COMMISSION_CLASS = {
                ("STK", "TSEJ", "JPY"): TsejTestCommission,
                ("FUT", "OSE", "JPY"): OseTestCommission,
                }

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier", "Currency", "PrimaryExchange"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "Japan",
                        "STK",
                        1,
                        1,
                        "JPY",
                        "TSEJ"
                    ],
                    23456: [
                        "Japan",
                        "FUT",
                        None,
                        None,
                        "HKD",
                        "OSE"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            with self.assertRaises(MoonshotParameterError) as cm:
                BuyBelow10ShortAbove10().backtest()

        self.assertIn((
            "expected a commission class for each combination of (sectype,exchange,currency) "
            "but none is defined for (FUT,OSE,HKD)"), repr(cm.exception))

    def test_apply_commissions_by_exchange_sectype_currency_eod(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied per sec type/exchange/currency.
        """
        class TsejTestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS

        class OseTestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0002 # 2 BPS

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            COMMISSION_CLASS = {
                ("STK", "TSEJ", "JPY"): TsejTestCommission,
                ("FUT", "OSE", "JPY"): OseTestCommission,
                }

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier", "Currency", "PrimaryExchange"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "Japan",
                        "STK",
                        1,
                        1,
                        "JPY",
                        "TSEJ"
                    ],
                    23456: [
                        "Japan",
                        "FUT",
                        None,
                        None,
                        "JPY",
                        "OSE"
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
             'TotalHoldings',
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

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             12345: ["nan",
                     0.00005,
                     0.0001,
                     0.0],
             23456: ["nan",
                     0.0001,
                     0.0002,
                     0.0002]}
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
             12345: ["nan",
                     -0.00005,
                     -0.022827272727272706, # (10.50 - 11)/11 * 0.5 - 0.0001
                     0.0242857142857143], # (9.99 - 10.50)/10.50 * -0.5
             23456: ["nan",
                     -0.0001,
                     -0.11383636363636365, # (8.50 - 11)/11 * 0.5
                     -0.11784705882352944] # (10.50 - 8.50)/8.50 * -0.5
             }
        )
    def test_apply_commissions_intraday_no_nlv(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied and no NLV is specified on an intraday strategy.
        """
        class TestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS
            EXCHANGE_FEE_RATE = 0
            MIN_COMMISSION = 800000000 # set high to verify ignored

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            COMMISSION_CLASS = TestCommission
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

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product(
                (master_fields, [dt_idx[0]], [times[0]]), names=["Field", "Date", "Time"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        1,
                        1
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        None,
                        None
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.5,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.5]}
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
                     0.00005,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.00005]}
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
                     -0.13172, # (15.45 - 10.12)/10.12 * -0.25 - 0.00005
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.0205724] # (14.50 - 13.40)/13.40 * 0.25 - 0.00005
             }
        )

    def test_apply_commissions_intraday_with_nlv(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied and NLV is specified on an intraday strategy.
        """
        class TestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS
            EXCHANGE_FEE_RATE = 0
            MIN_COMMISSION = 500

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            COMMISSION_CLASS = TestCommission
            POSITIONS_CLOSED_DAILY = True

            def prices_to_signals(self, prices):
                morning_prices = prices.loc["Open"].xs("09:30:00", level="Time")
                short_signals = morning_prices > 10
                self.save_to_results("Nlv", prices.loc["Nlv"].xs("09:30:00", level="Time"))
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

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier", "Currency"]
            idx = pd.MultiIndex.from_product(
                (master_fields, [dt_idx[0]], [times[0]]), names=["Field", "Date", "Time"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        1,
                        1,
                        "USD"
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        None,
                        None,
                        "USD"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = ShortAbove10Intraday().backtest(nlv={"USD":50000})

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'TotalHoldings',
             'Trade',
             'AbsWeight',
             'Weight',
             'Nlv'}
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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.5,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.5]}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00'],
             12345: [50000],
             23456: [50000]}
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
                     0.01,
                     0.0],
             23456: [0.0,
                     0.0,
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
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.14166996047430833, # (15.45 - 10.12)/10.12 * -0.25 - 0.01
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.030522388059701484] # (14.50 - 13.40)/13.40 * 0.25 - 0.01
             }
        )

    def test_apply_commissions_by_exchange_sectype_currency_intraday(self):
        """
        Tests that the resulting DataFrames are correct when commissions are
        applied per sec type/exchange/currency for an intraday strategy.
        """
        class TsejTestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0001 # 1 BPS

        class OseTestCommission(PercentageCommission):
            IB_COMMISSION_RATE = 0.0002 # 2 BPS

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            COMMISSION_CLASS = {
                ("STK", "TSEJ", "JPY"): TsejTestCommission,
                ("FUT", "OSE", "JPY"): OseTestCommission,
            }
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

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier", "Currency", "PrimaryExchange"]
            idx = pd.MultiIndex.from_product(
                (master_fields, [dt_idx[0]], [times[0]]), names=["Field", "Date", "Time"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "Japan",
                        "STK",
                        1,
                        1,
                        "JPY",
                        "TSEJ"
                    ],
                    23456: [
                        "Japan",
                        "FUT",
                        None,
                        None,
                        "JPY",
                        "OSE"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

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

        trades = results.loc["Trade"].reset_index()
        trades.loc[:, "Date"] = trades.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            trades.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.0,
                     -0.5,
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.5]}
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
                     0.00005,
                     0.0],
             23456: [0.0,
                     0.0,
                     0.0001]}
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
                     -0.1317199604743083, # (15.45 - 10.12)/10.12 * -0.25 - 0.00005
                     0.0],
             23456: [0.0,
                     0.0,
                     -0.020622388059701485] # (14.50 - 13.40)/13.40 * 0.25 - 0.0001
             }
        )

    def test_apply_commissions_eod_with_multiplier(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied for a security with a multiplier.
        """
        class TestFuturesCommission(FuturesCommission):
            IB_COMMISSION_PER_CONTRACT = 2

        class BuyBelow1000ShortAbove1000(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            COMMISSION_CLASS = TestFuturesCommission

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 1000
                short_signals = prices.loc["Close"] > 1000
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        900,
                        1100,
                        1050,
                        999,
                    ],
                    23456: [
                        # Close
                        900,
                        1100,
                        1050,
                        999,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/Chicago",
                        "FUT",
                        1,
                        10
                    ],
                    23456: [
                        "America/Chicago",
                        "FUT",
                        None,
                        20
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow1000ShortAbove1000().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'TotalHoldings',
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
             12345: [1,
                     -1,
                     -1,
                     1],
             23456: [1,
                     -1,
                     -1,
                     1]}
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
                     -0.5,
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
                     -0.5]}
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
                     0.0]}
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
             12345: ["nan",
                     0.0000909090909090909,
                     0.00019047619047619048,
                     0.0],
             23456: ["nan",
                     0.00004545454545454545,
                     0.00009523809523809524,
                     0.0]}
        )

        # The FUT with double the multiplier requires half the contracts and
        # therefore incurs half the commission
        self.assertAlmostEqual(
            commissions[12345].iloc[1], commissions[23456].iloc[1] * 2
        )

    def test_apply_commissions_eod_with_price_magnifier(self):
        """
        Tests that the resulting DataFrames are correct when commissions
        are applied for a security with a price magnifier.
        """
        class TestFuturesCommission(FuturesCommission):
            IB_COMMISSION_PER_CONTRACT = 2

        class BuyBelow1000ShortAbove1000(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            COMMISSION_CLASS = TestFuturesCommission

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 1000
                short_signals = prices.loc["Close"] > 1000
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        900,
                        1100,
                        1050,
                        999,
                    ],
                    23456: [
                        # Close
                        900,
                        1100,
                        1050,
                        999,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/Chicago",
                        "FUT",
                        1,
                        10
                    ],
                    23456: [
                        "America/Chicago",
                        "FUT",
                        100,
                        10
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            results = BuyBelow1000ShortAbove1000().backtest()

        self.assertSetEqual(
            set(results.index.get_level_values("Field")),
            {'Commission',
             'AbsExposure',
             'Signal',
             'Return',
             'Slippage',
             'NetExposure',
             'TotalHoldings',
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
             12345: [1,
                     -1,
                     -1,
                     1],
             23456: [1,
                     -1,
                     -1,
                     1]}
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
                     -0.5,
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
                     -0.5]}
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
                     0.0]}
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
             12345: ["nan",
                     0.0000909090909090909,
                     0.00019047619047619048,
                     0.0],
             23456: ["nan",
                     0.00909090909090909,
                     0.019047619047619048,
                     0.0]}
        )

        # The FUT with 100x the price magnifier (23456) requires 100x the
        # contracts and therefore incurs 100x the commission
        self.assertAlmostEqual(
            commissions[12345].iloc[1] * 100, commissions[23456].iloc[1]
        )

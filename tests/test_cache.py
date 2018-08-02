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
from quantrocket.exceptions import ImproperlyConfigured

class HistoricalPricesCacheTestCase(unittest.TestCase):

    def test_10_complain_if_houston_not_set(self):
        """
        Tests that a "HOUSTON_URL not set" error is raised if a backtest is
        run without mock. This is a control for later tests.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        with self.assertRaises(ImproperlyConfigured) as cm:

            BuyBelow10().backtest()

        self.assertIn("HOUSTON_URL is not set", repr(cm.exception))

    def test_20_load_history_from_mock(self):
        """
        Runs a strategy using mock to fill the history cache.
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

            results = BuyBelow10().backtest()

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
                     0.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_exposures = results.loc["AbsExposure"].reset_index()
        abs_exposures.loc[:, "Date"] = abs_exposures.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_exposures.to_dict(orient="list"),
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
                     -0.5,
                     0.0],
             23456: ["nan",
                     0.5,
                     -0.5,
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

    def test_30_load_history_from_cache(self):
        """
        Re-Runs the strategy without using mock to show that the history
        cache is used.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        results = BuyBelow10().backtest()

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
                     0.0,
                     0.0],
             23456: ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_exposures = results.loc["AbsExposure"].reset_index()
        abs_exposures.loc[:, "Date"] = abs_exposures.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_exposures.to_dict(orient="list"),
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
                     -0.5,
                     0.0],
             23456: ["nan",
                     0.5,
                     -0.5,
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

    def test_40_dont_use_cache(self):
        """
        Re-runs the strategy without using mock and specifying not to use the
        cache, which should trigger ImproperlyConfigured.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        with self.assertRaises(ImproperlyConfigured) as cm:

            BuyBelow10().backtest(history_cache="0H")

        self.assertIn("HOUSTON_URL is not set", repr(cm.exception))

        # Finally, remove cached files
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)
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
from moonshot.exceptions import MoonshotParameterError
from moonshot.cache import TMP_DIR

class HistoricalPricesTestCase(unittest.TestCase):

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_pass_history_and_master_db_params_correctly(self, mock_get_historical_prices):
        """
        Tests that params related to querying the history and master DBs are
        passed correctly to the underlying functions.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            DB = 'test-db'
            DB_FIELDS = ["Volume", "Wap", "Close"]
            MASTER_FIELDS = ["Timezone", "PrimaryExchange"]
            DB_TIME_FILTERS = ["00:00:00"]
            UNIVERSES = "us-stk"
            CONIDS = [12345,23456]
            EXCLUDE_CONIDS = 34567
            EXCLUDE_UNIVERSES = ["usa-stk-pharm", "usa-stk-biotech"]

            def prices_to_signals(self, prices):
                signals = prices.loc["Wap"] < 10
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Wap","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
                        9,
                        11,
                        10.50,
                        9.99,
                        # Wap
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
                        # Wap
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

            master_fields = ["Timezone", "PrimaryExchange"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "NASDAQ"
                    ],
                    23456: [
                        "America/New_York",
                        "NASDAQ"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_historical_prices_call = mock_get_historical_prices.mock_calls[0]
        _, args, kwargs = get_historical_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2017-03-25") # default 252+ trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["universes"], "us-stk")
        self.assertEqual(kwargs["conids"], [12345, 23456])
        self.assertEqual(kwargs["exclude_universes"], ['usa-stk-pharm', 'usa-stk-biotech'])
        self.assertEqual(kwargs["exclude_conids"], 34567)
        self.assertEqual(kwargs["fields"], ['Volume', 'Wap', 'Close'])
        self.assertEqual(kwargs["times"], ["00:00:00"])
        self.assertEqual(kwargs["master_fields"], ['Timezone', 'PrimaryExchange'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_set_lookback_window(self, mock_get_historical_prices):
        """
        Tests that setting LOOKBACK_WINDOW results in an appropriate start
        date for historical prices.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            DB = 'test-db'
            LOOKBACK_WINDOW = 350

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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
                        "America/New_York",
                    ],
                    23456: [
                        "America/New_York",
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_historical_prices_call = mock_get_historical_prices.mock_calls[0]
        _, args, kwargs = get_historical_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2016-10-24") # 350+ trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertIsNone(kwargs["universes"])
        self.assertIsNone(kwargs["conids"])
        self.assertIsNone(kwargs["exclude_universes"])
        self.assertIsNone(kwargs["exclude_conids"])
        self.assertEqual(kwargs["fields"], ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertIsNone(kwargs["times"])
        self.assertEqual(kwargs["master_fields"], ['Currency', 'MinTick', 'Multiplier', 'PriceMagnifier', 'PrimaryExchange', 'SecType', 'Symbol', 'Timezone'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_derive_lookback_window_from_window_params(self, mock_get_historical_prices):
        """
        Tests that lookback window is derived from the max of *_WINDOW params.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            DB = 'test-db'
            SOME_WINDOW = 100
            SOME_OTHER_WINDOW = 5

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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
                        "America/New_York",
                    ],
                    23456: [
                        "America/New_York",
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_historical_prices_call = mock_get_historical_prices.mock_calls[0]
        _, args, kwargs = get_historical_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2017-11-16") # 100+ trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["fields"], ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertEqual(kwargs["master_fields"], ['Currency', 'MinTick', 'Multiplier', 'PriceMagnifier', 'PrimaryExchange', 'SecType', 'Symbol', 'Timezone'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_complain_if_not_timezone_and_timezone_not_in_master_fields(self, mock_get_historical_prices):
        """
        Tests error handling when TIMEZONE is not set and Timezone is not in
        MASTER_FIELDS.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            MASTER_FIELDS = ["Symbol"]

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Symbol"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "ABC",
                    ],
                    23456: [
                        "DEF",
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        with self.assertRaises(MoonshotParameterError) as cm:
            BuyBelow10().backtest()

        self.assertIn((
            "cannot determine strategy timezone, please set TIMEZONE parameter "
            "or include 'Timezone' in MASTER_FIELDS"
            ), repr(cm.exception))

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_complain_if_nlv_missing_required_currencies(self, mock_get_historical_prices):
        """
        Tests error handling when NLV is provided but is missing required currencies.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Timezone", "SecType", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "MXN"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        with self.assertRaises(MoonshotParameterError) as cm:
            BuyBelow10().backtest(nlv={"USD":100000, "JPY":10000000})

        self.assertIn(
            "NLV dict is missing values for required currencies: MXN", repr(cm.exception))

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_complain_if_nlv_and_missing_sectype_currency(self, mock_get_historical_prices):
        """
        Tests error handling when NLV is provided but MASTER_FIELDS does not
        include Currency or SecType.
        """

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Timezone", "SecType", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "MXN"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            MASTER_FIELDS = ["Timezone", "Currency"]

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        with self.assertRaises(MoonshotParameterError) as cm:
            BuyBelow10().backtest(nlv={"USD":100000, "JPY":10000000})

        self.assertIn(
            "MASTER_FIELDS must include SecType and Currency if providing NLV", repr(cm.exception))

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            MASTER_FIELDS = ["Timezone", "SecType"]

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        with self.assertRaises(MoonshotParameterError) as cm:
            BuyBelow10().backtest(nlv={"USD":100000, "JPY":10000000})

        self.assertIn(
            "MASTER_FIELDS must include SecType and Currency if providing NLV", repr(cm.exception))

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_append_nlv_from_class_param(self, mock_get_historical_prices):
        """
        Tests appending of NLV when provided as a class param.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            NLV = {
                "USD": 50000,
                "MXN": 1000000
            }

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                self.save_to_results("Nlv", prices.loc["Nlv"])
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Timezone", "SecType", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "MXN"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

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
             'Weight',
             'Nlv'}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00'],
             12345: [50000],
             23456: [1000000]}
        )

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_append_nlv_from_arg(self, mock_get_historical_prices):
        """
        Tests appending of NLV when provided as an arg to backtest().
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                self.save_to_results("Nlv", prices.loc["Nlv"])
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Timezone", "SecType", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "MXN"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        results = BuyBelow10().backtest(nlv={
                "USD": 50000,
                "MXN": 1000000
            })

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
             'Weight',
             'Nlv'}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00'],
             12345: [50000],
             23456: [1000000]}
        )

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_append_forex_nlv_based_on_symbol(self, mock_get_historical_prices):
        """
        Tests that FX NLV is appended based on the Symbol, not the Currency.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                self.save_to_results("Nlv", prices.loc["Nlv"])
                return signals.astype(int)

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Timezone", "SecType", "Symbol", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "CASH",
                        "EUR",
                        "USD",
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        "EUR",
                        "USD"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        results = BuyBelow10().backtest(nlv={
                "USD": 50000,
                "EUR": 40000,
            })

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
             'Weight',
             'Nlv'}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00'],
             12345: [40000],
             23456: [50000]}
        )

    @patch("moonshot.strategies.base.get_historical_prices")
    def test_complain_if_nlv_for_fx_and_missing_symbol(self, mock_get_historical_prices):
        """
        Tests error handling when NLV is provided for CASH but MASTER_FIELDS does not
        include Symbol.
        """

        def _mock_get_historical_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
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

            master_fields = ["Timezone", "SecType", "Currency"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "CASH",
                        "USD",
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "MXN"
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            MASTER_FIELDS = ["Timezone", "SecType", "Currency"]

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        with self.assertRaises(MoonshotParameterError) as cm:
            BuyBelow10().backtest(nlv={"USD":100000, "MXN":10000000})

        self.assertIn(
            "MASTER_FIELDS must include Symbol if providing NLV and using CASH instruments", repr(cm.exception))

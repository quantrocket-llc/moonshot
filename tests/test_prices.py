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

class GetPricesTestCase(unittest.TestCase):

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    @patch("moonshot.strategies.base.get_prices")
    @patch("moonshot.strategies.base.download_master_file")
    def test_pass_history_and_master_db_params_correctly(self,
                                                         mock_download_master_file,
                                                         mock_get_prices):
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
            DB_TIMES = ["00:00:00"]
            DB_DATA_FREQUENCY = "daily"
            UNIVERSES = "us-stk"
            SIDS = ["FI12345","FI23456"]
            EXCLUDE_SIDS = "FI34567"
            EXCLUDE_UNIVERSES = ["usa-stk-pharm", "usa-stk-biotech"]
            CONT_FUT = False

            def prices_to_signals(self, prices):
                signals = prices.loc["Wap"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Wap","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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
            return prices

        def _mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier", "Exchange"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None,
                        "NASDAQ",
                    ],
                    "FI23456": [
                        "America/New_York",
                        "DEF",
                        "STK",
                        "USD",
                        None,
                        None,
                        "NASDAQ",
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_download_master_file.side_effect = _mock_download_master_file
        mock_get_prices.return_value = _mock_get_prices()

        results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_prices_call = mock_get_prices.mock_calls[0]
        _, args, kwargs = get_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2017-03-25") # default 252+ trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["universes"], "us-stk")
        self.assertEqual(kwargs["sids"], ["FI12345", "FI23456"])
        self.assertEqual(kwargs["exclude_universes"], ['usa-stk-pharm', 'usa-stk-biotech'])
        self.assertEqual(kwargs["exclude_sids"], "FI34567")
        self.assertEqual(kwargs["fields"], ['Volume', 'Wap', 'Close'])
        self.assertEqual(kwargs["times"], ["00:00:00"])
        self.assertEqual(kwargs["data_frequency"], "daily")
        self.assertFalse(kwargs["cont_fut"])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

        download_master_file_call = mock_download_master_file.mock_calls[0]
        _, args, kwargs = download_master_file_call
        self.assertListEqual(kwargs["sids"], ["FI12345", "FI23456"])
        self.assertListEqual(kwargs["fields"], [
            "Currency", "Multiplier", "PriceMagnifier",
            "Exchange", "SecType", "Symbol", "Timezone"])

    @patch("moonshot.strategies.base.get_prices")
    def test_set_lookback_window(self, mock_get_prices):
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

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
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
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_prices_call = mock_get_prices.mock_calls[0]
        _, args, kwargs = get_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2016-10-24") # 350+ trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertIsNone(kwargs["universes"])
        self.assertListEqual(kwargs["sids"], [])
        self.assertIsNone(kwargs["exclude_universes"])
        self.assertIsNone(kwargs["exclude_sids"])
        self.assertEqual(kwargs["fields"], ['Open', 'Close', 'Volume'])
        self.assertIsNone(kwargs["times"])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_prices")
    def test_derive_lookback_window_from_window_params(self, mock_get_prices):
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
            SOME_NONINT_WINDOW = "foo" # make sure ignored

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
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
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_prices_call = mock_get_prices.mock_calls[0]
        _, args, kwargs = get_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2017-11-16") # 100+ trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["fields"], ['Open', 'Close', 'Volume'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_prices")
    def test_derive_lookback_window_from_window_and_interval_params(self, mock_get_prices):
        """
        Tests that lookback window is derived from the max of *_WINDOW
        params plus the max of *_INTERVAL params.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            DB = 'test-db'
            SOME_WINDOW = 100
            SOME_OTHER_WINDOW = 5
            SOME_NONINT_WINDOW = "foo" # make sure ignored
            REBALANCE_INTERVAL = "Q"
            OTHER_INTERVAL = "MS"
            INVALID_INTERNVAL = "invalid" # make sure ignored

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
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
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_prices_call = mock_get_prices.mock_calls[0]
        _, args, kwargs = get_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertIn(kwargs["start_date"], ("2017-08-04", "2017-08-05", "2017-08-06")) # 100 + 60ish trading days before requested start_date
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["fields"], ['Open', 'Close', 'Volume'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_prices")
    def test_zero_lookback_window(self, mock_get_prices):
        """
        Tests that a lookback window of 0 is respected, i.e. no buffer is
        added.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            DB = 'test-db'
            LOOKBACK_WINDOW = 0

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
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
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_prices_call = mock_get_prices.mock_calls[0]
        _, args, kwargs = get_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2018-05-01")
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["fields"], ['Open', 'Close', 'Volume'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_prices")
    def test_under_one_week_lookback_window(self, mock_get_prices):
        """
        Tests that a smaller buffer is used for a lookback window of less
        than a week.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            DB = 'test-db'
            LOOKBACK_WINDOW = 2

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
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
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            results = BuyBelow10().backtest(start_date="2018-05-01", end_date="2018-05-04")

        get_prices_call = mock_get_prices.mock_calls[0]
        _, args, kwargs = get_prices_call
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2018-04-25")
        self.assertEqual(kwargs["end_date"], "2018-05-04")
        self.assertEqual(kwargs["fields"], ['Open', 'Close', 'Volume'])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

    @patch("moonshot.strategies.base.get_prices")
    def test_complain_if_cannot_infer_timezone(self, mock_get_prices):
        """
        Tests error handling when multiple timezones are present and we
        cannot infer the time zone.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/Mexico_City",
                        "DEF",
                        "STK",
                        "MXN",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            with self.assertRaises(MoonshotParameterError) as cm:
                BuyBelow10().backtest(nlv={"USD":100000, "JPY":10000000})

        self.assertIn(
            "cannot infer timezone because multiple timezones are present "
            "in data, please specify TIMEZONE explicitly (timezones: America/New_York, America/Mexico_City)", repr(cm.exception))

    @patch("moonshot.strategies.base.get_prices")
    def test_complain_if_nlv_missing_required_currencies(self, mock_get_prices):
        """
        Tests error handling when NLV is provided but is missing required currencies.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """

            TIMEZONE = "America/Mexico_City"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/Mexico_City",
                        "DEF",
                        "STK",
                        "MXN",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
            with self.assertRaises(MoonshotParameterError) as cm:
                BuyBelow10().backtest(nlv={"USD":100000, "JPY":10000000})

        self.assertIn(
            "NLV dict is missing values for required currencies: MXN", repr(cm.exception))

    @patch("moonshot.strategies.base.get_prices")
    def test_append_nlv_from_class_param(self, mock_get_prices):
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
            TIMEZONE = "America/Mexico_City"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                self.save_to_results("Nlv", signals.apply(lambda x: self._securities_master.Nlv, axis=1))
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/Mexico_City",
                        "DEF",
                        "STK",
                        "MXN",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
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
             'Weight',
             'Nlv'}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                "2018-05-01T00:00:00","2018-05-02T00:00:00","2018-05-03T00:00:00", "2018-05-04T00:00:00"],
             "FI12345": [50000.0,50000.0,50000.0,50000.0],
             "FI23456": [1000000.0,1000000.0,1000000.0,1000000.0]}
        )

    @patch("moonshot.strategies.base.get_prices")
    def test_append_nlv_from_arg(self, mock_get_prices):
        """
        Tests appending of NLV when provided as an arg to backtest().
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            TIMEZONE = "America/Mexico_City"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                self.save_to_results("Nlv", signals.apply(lambda x: self._securities_master.Nlv, axis=1))
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "ABC",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/Mexico_City",
                        "DEF",
                        "STK",
                        "MXN",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
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
             'TotalHoldings',
             'Turnover',
             'AbsWeight',
             'Weight',
             'Nlv'}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                "2018-05-01T00:00:00","2018-05-02T00:00:00","2018-05-03T00:00:00", "2018-05-04T00:00:00"],
             "FI12345": [50000.0,50000.0,50000.0,50000.0],
             "FI23456": [1000000.0,1000000.0,1000000.0,1000000.0]}
        )

    @patch("moonshot.strategies.base.get_prices")
    def test_append_fx_nlv_based_on_symbol(self, mock_get_prices):
        """
        Tests that FX NLV is appended based on the Symbol, not the Currency.
        """
        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                self.save_to_results("Nlv", signals.apply(lambda x: self._securities_master.Nlv, axis=1))
                return signals.astype(int)

        def _mock_get_prices():

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close","Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
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
                    "FI23456": [
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

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "Symbol", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "EUR",
                        "CASH",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/New_York",
                        "EUR",
                        "STK",
                        "USD",
                        None,
                        None,
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "Sid"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_get_prices.return_value = _mock_get_prices()

        with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
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
             'TotalHoldings',
             'Turnover',
             'AbsWeight',
             'Weight',
             'Nlv'}
        )

        nlvs = results.loc["Nlv"].reset_index()
        nlvs.loc[:, "Date"] = nlvs.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            nlvs.to_dict(orient="list"),
            {'Date': [
                "2018-05-01T00:00:00","2018-05-02T00:00:00","2018-05-03T00:00:00", "2018-05-04T00:00:00"],
             "FI12345": [40000.0,40000.0,40000.0,40000.0],
             "FI23456": [50000.0,50000.0,50000.0,50000.0]}
        )

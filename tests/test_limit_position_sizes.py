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
import glob
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from moonshot import Moonshot
from moonshot.exceptions import MoonshotParameterError
from moonshot.cache import TMP_DIR

class LimitPositionSizesBacktestTestCase(unittest.TestCase):

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    def test_complain_if_limit_position_sizes_no_nlv(self):
        """
        Tests error handling when limit position sizes is implemented but NLV
        is not provided in the backtest.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):

                closes = prices.loc["Close"]
                max_positions_for_longs = pd.DataFrame(100, index=closes.index, columns=closes.columns)
                max_positions_for_shorts = pd.DataFrame(100, index=closes.index, columns=closes.columns)
                return max_positions_for_longs, max_positions_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
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

                    with self.assertRaises(MoonshotParameterError) as cm:

                        BuyBelow10ShortAbove10Overnight().backtest()

        self.assertIn("must provide NLVs if using limit_position_sizes", repr(cm.exception))


    def test_no_limit_position_sizes(self):
        """
        Tests running a backtest in which position sizes aren't limited.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
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
                '2018-05-03T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0],
             23456: [1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [0.5,
                     -0.5,
                     -0.5],
             23456: [0.5,
                     -0.5,
                     0.5]}
        )

    def test_limit_position_sizes_by_volume(self):
        """
        Tests running a backtest in which position sizes are limited by volume.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                volumes = prices.loc["Volume"]
                max_shares = (volumes * 0.01).round()
                max_quantities_for_longs = max_quantities_for_shorts = max_shares
                return max_quantities_for_longs, max_quantities_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close", "Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        # Volume
                        100000,
                        150000,
                        125000
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        # Volume
                        50000,
                        60000,
                        70000000
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

                    results = BuyBelow10ShortAbove10Overnight().backtest(nlv={"USD":50000})

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

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0],
             23456: [1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [
                 # 100K volume * 1% * 9 / 50K
                 0.18,
                 # 150K volume * 1% * 11 / 50K
                 -0.33,
                 # 125K volume * 1% * 10.50 / 50K
                 -0.2625],
             23456: [
                 # 50K volume * 1% * 9.89 / 50K
                 0.0989,
                 # 60K volume * 1% * 11 / 50K
                 -0.132,
                 # 0.5 expected but watch out for floating point
                 0.49997]}
        )

    def test_limit_position_sizes_once_a_day_intraday_strategy(self):
        """
        Tests running a backtest of a once a day intraday strategy in which
        position sizes are limited by volume.
        """

        class BuyBelow10ShortAbove10(Moonshot):

            def prices_to_signals(self, prices):
                morning_prices = prices.loc["Close"].xs("09:30:00", level="Time")
                short_signals = morning_prices > 10
                long_signals = morning_prices < 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.5)
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

            def limit_position_sizes(self, prices):

                closes = prices.loc["Close"].xs("09:30:00", level="Time")
                max_shares_for_longs = pd.DataFrame(
                    300, index=closes.index, columns=closes.columns
                )
                max_shares_for_shorts = max_shares_for_longs * 2
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03"])
            fields = ["Close"]
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
                    ],
                    23456: [
                        # Close
                        10.56,
                        12.01,
                        8.50,
                        9.80,
                        13.40,
                        14.50,
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

                    results = BuyBelow10ShortAbove10().backtest(nlv={"USD": 100000})

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

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [1,
                     -1,
                     1],
             23456: [-1,
                     1,
                     -1]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [
                 # 300 * 9.6 / 100K
                 0.0288,
                 # 600 * 10.12 / 100K
                 -0.06071999999999999,
                 # 300 * 8.67 / 100K
                 0.02601],
             23456: [
                 # 600 * 10.56 / 100K
                 -0.06336,
                 # 300 * 8.5 / 100K
                 0.0255,
                 # 600 * 13.40 / 100K
                 -0.0804]}
        )

    def test_limit_position_sizes_continuous_intraday_strategy(self):
        """
        Tests running a backtest of a continuous intraday strategy in which
        position sizes are limited by volume.
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

            def limit_position_sizes(self, prices):

                closes = prices.loc["Close"]
                max_shares_for_longs = pd.DataFrame(
                    300, index=closes.index, columns=closes.columns
                )
                max_shares_for_shorts = max_shares_for_longs * 2
                return max_shares_for_longs, max_shares_for_shorts

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

                    results = BuyBelow10ShortAbove10ContIntraday().backtest(nlv={"USD": 100000})

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
             12345: [0.0288, # 300 * 9.6 / 100K
                     -0.0627, # 600 * 10.45 / 100K
                     -0.06072, # 600 * 10.12 / 100K
                     -0.0927, # 600 * 15.45 / 100K
                     0.02601, # 300 * 8.67 / 100K
                     -0.0738], # 600 * 12.30 / 100K
             23456: [-0.06336, # 600 * 10.56 / 100K
                     -0.07206, # 600 * 12.01 / 100K
                     -0.063, # 600 * 10.50 / 100K
                     0.0294, # 300 * 9.80 / 100K
                     -0.0804, # 600 * 13.40 / 100K
                     0.0225 # 300 * 7.50 / 100K
                     ]}
        )

    def test_limit_short_position_sizes_only(self):
        """
        Tests running a backtest in which shorts are limited but not longs.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                volumes = prices.loc["Volume"]
                max_shares = (volumes * 0.01).round()
                max_quantities_for_shorts = max_shares
                return None, max_quantities_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close", "Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        # Volume
                        100000,
                        150000,
                        125000
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        # Volume
                        50000,
                        60000,
                        70000000
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

                    results = BuyBelow10ShortAbove10Overnight().backtest(nlv={"USD":50000})

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

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0],
             23456: [1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [
                 # 0.5 expected but watch out for floating point
                 0.50004,
                 # 150K volume * 1% * 11 / 50K
                 -0.33,
                 # 125K volume * 1% * 10.50 / 50K
                 -0.2625],
             23456: [
                 # 0.5 expected but watch out for floating point
                 0.5000384,
                 # 60K volume * 1% * 11 / 50K
                 -0.132,
                 # 0.5 expected but watch out for floating point
                 0.49997]}
        )

    def test_ignore_nans(self):
        """
        Tests that NaNs and Nones returned by limit_position_sizes are interpreted as
        "no limit" for the particular day.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_longs = max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            300,
                            None,
                            np.nan
                        ],
                        23456:[
                            np.nan,
                            400,
                            None
                        ]
                    }, index=prices.loc["Close"].index
                )
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close", "Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        # Volume
                        100000,
                        150000,
                        125000
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        # Volume
                        50000,
                        60000,
                        70000000
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

                    results = BuyBelow10ShortAbove10Overnight().backtest(nlv={"USD":50000})

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

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0],
             23456: [1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [
                 # 300 * 9 / 50K
                 0.054,
                 # -0.5 expected but watch out for floating point
                 -0.50006,
                 # -0.5 expected but watch out for floating point
                 -0.50001],
             23456: [
                 # 0.5 expected but watch out for floating point
                 0.5000384,
                 # 400 * 11 / 50K
                 -0.088,
                 # 0.5 expected but watch out for floating point
                 0.49997]}
        )

    def test_limit_position_sizes_forex(self):
        """
        Tests that Forex position sizes are limited based on the NLV of the
        Symbol, not the Currency, and are based on a contract value of 1. See
        also
        test_historical_prices.HistoricalPricesTestCase.test_append_forex_nlv_based_on_symbol
        and test_trade.TradeTestCase.test_forex.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_longs = max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            300,
                            400,
                            500
                        ],
                        23456:[
                            300,
                            400,
                            500
                        ]
                    }, index=prices.loc["Close"].index
                )
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close", "Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        # Volume
                        100000,
                        150000,
                        125000
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        # Volume
                        50000,
                        60000,
                        70000000
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
                        "EUR",
                        "CASH",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
                        "ABC",
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

                    results = BuyBelow10ShortAbove10Overnight().backtest(
                        nlv={
                            "USD":50000,
                            "EUR": 35000,
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
             'Weight'}
        )

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0],
             23456: [1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [
                 # 300 / 35K EUR
                 0.008571428571428572,
                 # 400 / 35K EUR
                 -0.011428571428571429,
                 # 500 / 35K EUR
                 -0.014285714285714285],
             23456: [
                 # 300 * 9.89 / 50K USD
                 0.05934,
                 # 400 * 11 / 50K USD
                 -0.088,
                 # 500 * 8.5 / 50K USD
                 0.085]}
        )

    def test_price_magnifier_and_multiplier(self):
        """
        Tests that limiting position sizes incorporates Multipliers and
        PriceMagnifiers.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_longs = max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            30,
                            40,
                            50
                        ],
                        23456:[
                            30,
                            40,
                            50
                        ]
                    }, index=prices.loc["Close"].index
                )
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01", "2018-05-02", "2018-05-03"])
            fields = ["Close", "Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        9,
                        11,
                        10.50,
                        # Volume
                        100000,
                        150000,
                        125000
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        # Volume
                        50000,
                        60000,
                        70000000
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
                        "America/Chicago",
                        "ABC",
                        "FUT",
                        "USD",
                        None,
                        20
                    ],
                    23456: [
                        "America/Chicago",
                        "DEF",
                        "FUT",
                        "USD",
                        10,
                        50,
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

                    results = BuyBelow10ShortAbove10Overnight().backtest(nlv={"USD":500000})

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

        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [1.0,
                     -1.0,
                     -1.0],
             23456: [1.0,
                     -1.0,
                     1.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00'],
             12345: [
                 # 30 * 20 * 9 / 500K
                 0.0108,
                 # 40 * 20 * 11 / 500K
                 -0.0176,
                 # 50 * 20 * 10.50 / 500K
                 -0.021],
             23456: [
                 # 30 * 50 / 10 * 9.89 / 500K
                 0.002967,
                 # 40 * 50 / 10 * 11 / 500K
                 -0.004400000000000001,
                 # 50 * 50 / 10 * 8.5 / 500K
                 0.00425]}
        )

class LimitPositionSizesTradeTestCase(unittest.TestCase):

    def test_no_limit_position_sizes(self):
        """
        Tests running a strategy without limiting position sizes.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Open"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Open
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Open
                        9.89,
                        11,
                        8.50,
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            return []

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10ShortAbove10().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    # allocation 1.0 * weight 0.5 * 60K NLV / 10.50
                    'TotalQuantity': 2857,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # allocation 1.0 * weight 0.5 * 60K NLV / 8.50
                    'TotalQuantity': 3529,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_limit_position_sizes(self):
        """
        Tests running a strategy and limiting position sizes.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_longs = max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            1200,
                            1200,
                            1350
                            ],
                        23456:[
                            2300,
                            2300,
                            2199
                        ]
                        }, index=prices.loc["Open"].index
                )
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Open"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Open
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Open
                        9.89,
                        11,
                        8.50,
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            return []

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10ShortAbove10().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 1350,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 2199,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_limit_short_position_sizes_only(self):
        """
        Tests running a strategy and limiting short position sizes only.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            1200,
                            1200,
                            1350
                            ],
                        23456:[
                            2300,
                            2300,
                            2199
                        ]
                        }, index=prices.loc["Open"].index
                )
                return None, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Open"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Open
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Open
                        9.89,
                        11,
                        8.50,
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            return []

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10ShortAbove10().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 1350,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 3529,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_ignore_nans(self):
        """
        Tests that NaNs and Nones returned by limit_position_sizes are interpreted as
        "no limit" for the particular day.
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_longs = max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            1200,
                            1200,
                            1450
                            ],
                        23456:[
                            2300,
                            2300,
                            None
                        ],
                        34567: [
                            None,
                            500,
                            np.nan
                        ]
                        }, index=prices.loc["Open"].index
                )
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Open"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Open
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Open
                        9.89,
                        11,
                        8.50,
                    ],
                    34567: [
                        # Open
                        9.99,
                        10,
                        10.50,
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None,
                    ],
                    34567: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            return []

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):
                                    orders = BuyBelow10ShortAbove10().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 1450, # limited
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # allocation 1.0 * weight 0.3333 * 60K NLV / 8.50
                    'TotalQuantity': 2353,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 34567,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    # allocation 1.0 * weight 0.3333 * 60K NLV / 10.50
                    'TotalQuantity': 1905,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
            ]
        )

    def test_limit_position_sizes_with_existing_position(self):
        """
        Tests running a strategy and limiting position sizes when there are
        existing positions. The limits should be applied before applying the
        existing position diff. (The reason for this design is to mirror
        backtesting.)
        """

        class BuyBelow10ShortAbove10(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):
                max_shares_for_longs = max_shares_for_shorts = pd.DataFrame(
                    {
                        12345: [
                            1200,
                            1200,
                            1350
                            ],
                        23456:[
                            2300,
                            2300,
                            2199
                        ]
                        }, index=prices.loc["Open"].index
                )
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Open"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Open
                        9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Open
                        9.89,
                        11,
                        8.50,
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            positions = [
                {
                    "Account": "U123",
                    "OrderRef": "long-short-10",
                    "ConId": 23456,
                    "Quantity": 400
                    },
            ]
            return positions

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10ShortAbove10().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 1350,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 1799, # 2199 - 400
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_limit_position_sizes_once_a_day_intraday_strategy(self):
        """
        Tests running a once a day intraday strategy and limiting position
        sizes.
        """

        class BuyBelow10ShortAbove10(Moonshot):

            CODE = "pivot-10"

            def prices_to_signals(self, prices):
                morning_prices = prices.loc["Close"].xs("09:30:00", level="Time")
                short_signals = morning_prices > 10
                long_signals = morning_prices < 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.5)
                return weights

            def limit_position_sizes(self, prices):

                closes = prices.loc["Close"].xs("09:30:00", level="Time")
                max_shares_for_longs = pd.DataFrame(
                    300, index=closes.index, columns=closes.columns
                )
                max_shares_for_shorts = max_shares_for_longs * 2
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Close"]
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
                    ],
                    23456: [
                        # Close
                        10.56,
                        12.01,
                        10.50,
                        9.80,
                        13.40,
                        14.50,
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            return []

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10ShortAbove10().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )
        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'pivot-10',
                    # 1.0 allocation * 0.25 weight * 60K / 12.30 = 1220, but reduced to 300
                    'TotalQuantity': 300,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'pivot-10',
                    'TotalQuantity': 600,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_limit_position_sizes_continuous_intraday_strategy(self):
        """
        Tests running a continuous intraday strategy and limiting position
        sizes.
        """
        class BuyBelow10ShortAbove10ContIntraday(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "c-intraday-pivot-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Close"] <= 10
                short_signals = prices.loc["Close"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def limit_position_sizes(self, prices):

                closes = prices.loc["Close"]
                max_shares_for_longs = pd.DataFrame(
                    300, index=closes.index, columns=closes.columns
                )
                max_shares_for_shorts = max_shares_for_longs * 2
                return max_shares_for_longs, max_shares_for_shorts

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02"])
            fields = ["Close"]
            times = ["10:00:00", "11:00:00", "12:00:00"]
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/New_York",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[60000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD"],
                                      QuoteCurrency=["USD"],
                                         Rate=[1.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        def mock_list_positions(**kwargs):
            return []

        def mock_download_order_statuses(f, **kwargs):
            pass

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10ShortAbove10ContIntraday().trade(
                                        {"U123": 1.0}, review_date="2018-05-02 12:05:00")

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'Exchange',
             'OrderType',
             'Tif'}
        )
        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'c-intraday-pivot-10',
                    # 1.0 allocation * 0.5 weight * 60K / 12.30 = 2439, but reduced to 600
                    'TotalQuantity': 600,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'c-intraday-pivot-10',
                    'TotalQuantity': 300,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

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

import unittest
from unittest.mock import patch
import pandas as pd
import json
from moonshot import Moonshot
from moonshot.exceptions import MoonshotParameterError

class TradeTestCase(unittest.TestCase):

    def test_basic_long_only_strategy(self):
        """
        Tests that the resulting orders DataFrame is correct after running a basic
        long-only strategy that largely relies on the default methods.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
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
                                         NetLiquidation=[55000],
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

                                    orders = BuyBelow10().trade({"U123": 1.0})

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

        # expected quantity for 23456:
        # 1.0 weight * 1.0 allocation * 55K / 8.50 = 6471

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef':
                    'buy-below-10',
                    'TotalQuantity': 6471,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_no_orders(self):
        """
        Tests running a strategy that returns no orders.
        """

        class BuyBelow1(Moonshot):
            """
            A basic test strategy that buys below 1.
            """
            CODE = "buy-below-1"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 1
                return signals.astype(int)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
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
                                         NetLiquidation=[55000],
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

                                    orders = BuyBelow1().trade({"U123": 1.0})

        self.assertIsNone(orders)

    @patch("moonshot.strategies.base.get_historical_prices")
    @patch("moonshot.strategies.base.download_account_balances")
    @patch("moonshot.strategies.base.download_exchange_rates")
    @patch("moonshot.strategies.base.list_positions")
    @patch("moonshot.strategies.base.download_order_statuses")
    @patch("moonshot.strategies.base.download_master_file")
    @patch("moonshot.strategies.base.get_db_config")
    def test_pass_quantrock_client_params_correctly(
        self, mock_get_db_config, mock_download_master_file, mock_download_order_statuses, mock_list_positions,
        mock_download_exchange_rates, mock_download_account_balances,
        mock_get_historical_prices):
        """
        Tests that params are correctly passed to underlying client functions.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"
            DB = 'test-db'
            DB_FIELDS = ["Volume", "Wap", "Close"]
            MASTER_FIELDS = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier", "PrimaryExchange"]
            DB_TIMES = ["00:00:00"]
            UNIVERSES = "us-stk"
            CONIDS = [12345,23456]
            EXCLUDE_CONIDS = 34567
            EXCLUDE_UNIVERSES = ["usa-stk-pharm", "usa-stk-biotech"]
            CONT_FUT = False

            def prices_to_signals(self, prices):
                signals = prices.loc["Wap"] < 10
                return signals.astype(int)

        def _mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02", "2018-05-03"])
            fields = ["Close", "Wap", "Volume"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        #Close
                        9,
                        11,
                        10.50,
                        # Wap
                        9,
                        11,
                        10.50,
                        # Volume
                        5000,
                        16000,
                        8800,
                    ],
                    23456: [
                        # Close
                        9.89,
                        11,
                        8.50,
                        # Wap
                        9.89,
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

        mock_get_historical_prices.return_value = _mock_get_historical_prices()

        def _mock_get_db_config():
            return {
                'vendor': 'ib',
                'domain': 'main',
                'bar_size': '1 day'
            }

        mock_get_db_config.return_value = _mock_get_db_config()

        def _mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier", "PrimaryExchange"]
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None,
                        "NYSE"
                    ],
                    23456: [
                        "America/New_York",
                        "STK",
                        "CAD",
                        None,
                        None,
                        "NASDAQ"
                    ]
                },
                index=master_fields
            )
            securities.columns.name = "ConId"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        mock_download_master_file.side_effect = _mock_download_master_file

        def _mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[55000],
                                         Currency=["EUR"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        mock_download_account_balances.side_effect = _mock_download_account_balances

        def _mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["EUR"],
                                      QuoteCurrency=["CAD"],
                                         Rate=[2.0]))
            rates.to_csv(f, index=False)
            f.seek(0)

        mock_download_exchange_rates.side_effect = _mock_download_exchange_rates

        def _mock_list_positions(**kwargs):
            return []

        mock_list_positions.side_effect = _mock_list_positions

        def _mock_download_order_statuses(f, **kwargs):
            pass

        mock_download_order_statuses.side_effect = _mock_download_order_statuses

        # use review_date so we can validate start_date
        orders = BuyBelow10().trade({"U123": 1.0}, review_date="2018-05-03")

        get_historical_prices_call = mock_get_historical_prices.mock_calls[0]
        _, args, kwargs = get_historical_prices_call
        self.assertFalse(bool(args))
        self.assertListEqual(kwargs["codes"], ["test-db"])
        self.assertEqual(kwargs["start_date"], "2017-03-27") # default 252+ trading days before requested start_date
        self.assertIsNone(kwargs["end_date"])
        self.assertEqual(kwargs["universes"], "us-stk")
        self.assertEqual(kwargs["conids"], [12345, 23456])
        self.assertEqual(kwargs["exclude_universes"], ['usa-stk-pharm', 'usa-stk-biotech'])
        self.assertEqual(kwargs["exclude_conids"], 34567)
        self.assertEqual(kwargs["fields"], ['Volume', 'Wap', 'Close'])
        self.assertEqual(kwargs["times"], ["00:00:00"])
        self.assertFalse(kwargs["cont_fut"])
        self.assertEqual(kwargs["master_fields"],
                         ["Timezone", "SecType", "Currency", "PriceMagnifier",
                          "Multiplier", "PrimaryExchange"])
        self.assertIsNone(kwargs["timezone"])
        self.assertTrue(kwargs["infer_timezone"])

        download_account_balances_call = mock_download_account_balances.mock_calls[0]
        _, args, kwargs = download_account_balances_call
        self.assertTrue(kwargs["latest"])
        self.assertListEqual(kwargs["accounts"], ["U123"])
        self.assertListEqual(kwargs["fields"], ["NetLiquidation"])

        download_exchange_rates_call = mock_download_exchange_rates.mock_calls[0]
        _, args, kwargs = download_exchange_rates_call
        self.assertTrue(kwargs["latest"])
        self.assertListEqual(kwargs["base_currencies"], ["EUR"])
        self.assertListEqual(kwargs["quote_currencies"], ["USD", "CAD"])

        list_positions_call = mock_list_positions.mock_calls[0]
        _, args, kwargs = list_positions_call
        self.assertFalse(bool(args))
        self.assertListEqual(kwargs["order_refs"], ["buy-below-10"])
        self.assertListEqual(kwargs["accounts"], ["U123"])
        self.assertListEqual(kwargs["conids"], [12345, 23456])

        download_order_statuses_call = mock_download_order_statuses.mock_calls[0]
        _, args, kwargs = download_order_statuses_call
        self.assertEqual(kwargs["output"], "json")
        self.assertTrue(kwargs["open_orders"])
        self.assertListEqual(kwargs["order_refs"], ["buy-below-10"])
        self.assertListEqual(kwargs["accounts"], ["U123"])
        self.assertListEqual(kwargs["conids"], [12345, 23456])

    def test_long_short_strategy_override_methods(self):
        """
        Tests that the orders DataFrame is correct after running a
        long-short strategy that overrides the major trade methods.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10 and holds overnight.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def order_stubs_to_orders(self, orders, prices):
                orders["Exchange"] = "NYSE"
                orders["OrderType"] = 'LMT'
                orders["LmtPrice"] = 10.00
                orders["Tif"] = "GTC"
                return orders

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

                                    orders = BuyBelow10ShortAbove10Overnight().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'ConId',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'LmtPrice',
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
                    # allocation 1.0 * weight 0.25 * 60K NLV / 10.50
                    'TotalQuantity': 1429,
                    'Exchange': 'NYSE',
                    'OrderType': 'LMT',
                    'LmtPrice': 10.0,
                    'Tif': 'GTC'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # allocation 1.0 * weight 0.25 * 60K NLV / 8.50
                    'TotalQuantity': 1765,
                    'Exchange': 'NYSE',
                    'OrderType': 'LMT',
                    'LmtPrice': 10.0,
                    'Tif': 'GTC'
                }
            ]
        )

    def test_short_only_once_a_day_intraday_strategy(self):
        """
        Tests that the resulting DataFrames are correct after running a
        short-only once-a-day intraday strategy.
        """

        class ShortAbove10Intraday(Moonshot):
            """
            A basic test strategy that shorts above 10 and holds intraday.
            """
            CODE = "short-above-10"

            def prices_to_signals(self, prices):
                morning_prices = prices.loc["Open"].xs("09:30:00", level="Time")
                short_signals = morning_prices > 10
                return -short_signals.astype(int)

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
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

                                    orders = ShortAbove10Intraday().trade({"U123": 1.0})

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
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'short-above-10',
                    # 1.0 allocation * 0.25 weight * 60K / 14.50
                    'TotalQuantity': 1034,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_continuous_intraday_strategy(self):
        """
        Tests that the resulting DataFrames are correct after running a
        continuous intraday strategy.
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
                    # 1.0 allocation * 0.5 weight * 60K / 12.30 = 2439
                    'TotalQuantity': 2439,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'c-intraday-pivot-10',
                    # 1.0 allocation * 0.5 weight * 60K / 7.50 = 4000
                    'TotalQuantity': 4000,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_single_account(self):
        """
        Tests that the orders DataFrame is correct after running a
        long-short strategy allocated to a single account.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def order_stubs_to_orders(self, orders, prices):
                orders["Exchange"] = "SMART"
                orders["OrderType"] = 'MKT'
                orders["Tif"] = "GTC"
                return orders

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
                                         NetLiquidation=[85000],
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

                                    orders = BuyBelow10ShortAbove10Overnight().trade({"U123": 0.5})

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
                    # allocation 0.5 * weight 0.25 * 85K NLV / 10.50
                    'TotalQuantity': 1012,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # allocation 0.5 * weight 0.25 * 85K NLV / 8.50
                    'TotalQuantity': 1250,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                }
            ]
        )

    def test_multiple_accounts(self):
        """
        Tests that the orders DataFrame is correct after running a
        long-short strategy allocated to multiple accounts.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def order_stubs_to_orders(self, orders, prices):
                orders["Exchange"] = "SMART"
                orders["OrderType"] = 'MKT'
                orders["Tif"] = "GTC"
                return orders

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
            balances = pd.DataFrame(dict(Account=["U123", "DU234"],
                                         NetLiquidation=[85000, 450000],
                                         Currency=["USD", "USD"]))
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

                                    orders = BuyBelow10ShortAbove10Overnight().trade({"U123": 0.5, "DU234": 0.3})

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
                    # 0.5 allocation * 0.25 weight * 85K / 10.50
                    'TotalQuantity': 1012,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                },
                {
                    'ConId': 12345,
                    'Account': 'DU234',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    # 0.3 allocation * 0.25 weight * 450K / 10.50
                    'TotalQuantity': 3214,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # 0.5 allocation * 0.25 weight * 85K / 8.50
                    'TotalQuantity': 1250,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                },
                {
                    'ConId': 23456,
                    'Account': 'DU234',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # 0.3 allocation * 0.25 weight * 450K / 8.50
                    'TotalQuantity': 3971,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                }
            ]
        )

    def test_existing_positions(self):
        """
        Tests that the orders DataFrame is correct after running a long only
        strategy allocated to multiple accounts where some of the accounts
        have existing positions.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

            def signals_to_target_weights(self, signals, prices):
                return self.allocate_fixed_weights(signals, 0.5)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
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
            balances = pd.DataFrame(dict(Account=["U123", "DU234", "U999", "DU111"],
                                         NetLiquidation=[85000, 450000, 56000, 150000],
                                         Currency=["USD", "USD", "USD", "USD"]))
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
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": 400
                },
                # this is the position we want, so no order will be needed
                {
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": 7941
                },
                {
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 300
                },
                {
                    "Account": "DU111",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": -300
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

                                    orders = BuyBelow10().trade(
                                        {"U123": 0.5,
                                         "DU234": 0.3,
                                         "U999": 0.6,
                                         "DU111": 0.2
                                         })

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
                    'Account': 'DU234',
                    'Action': 'SELL',
                    'OrderRef': 'buy-below-10',
                    # close open position
                    'TotalQuantity': 300.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.5 allocation * 0.5 weight * 85K / 8.50 - 400
                    'TotalQuantity': 2100.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U999',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.6 allocation * 0.5 weight * 56K / 8.5
                    'TotalQuantity': 1976.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'DU111',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.2 allocation * 0.5 weight * 150K / 8.50 - (-300)
                    'TotalQuantity': 2065.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_existing_open_orders(self):
        """
        Tests that the orders DataFrame is correct after running a long only
        strategy allocated to multiple accounts where some of the accounts
        have existing open orders.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

            def signals_to_target_weights(self, signals, prices):
                return self.allocate_fixed_weights(signals, 0.5)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
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
            balances = pd.DataFrame(dict(Account=["U123", "DU234", "U999", "DU111"],
                                         NetLiquidation=[85000, 450000, 56000, 150000],
                                         Currency=["USD", "USD", "USD", "USD"]))
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
            orders = [
                {
                    "Account": "U123",
                    "Action": "BUY",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Filled": 200,
                    "Remaining": 200,
                    "TotalQuantity": 400,
                    "Status": "Submitted"
                },
                # this is the position we want, so no order will be needed
                {
                    "Account": "DU234",
                    "Action": "BUY",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Filled": 0,
                    "Remaining": 7941,
                    "TotalQuantity": 7941,
                    "Status": "Submitted"
                },
                # Next two orders are for same conid/account, should be summed
                {
                    "Account": "DU234",
                    "Action": "SELL",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Filled": 0,
                    "Remaining": 100,
                    "TotalQuantity": 200,
                    "Status": "Submitted"
                },
                {
                    "Account": "DU234",
                    "Action": "BUY",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Filled": 0,
                    "Remaining": 200,
                    "TotalQuantity": 200,
                    "Status": "Submitted"
                },
                {
                    "Account": "DU111",
                    "Action": "SELL",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Filled": 0,
                    "Remaining": 600,
                    "TotalQuantity": 600,
                    "Status": "Submitted"
                },

            ]
            json.dump(orders, f)
            f.seek(0)

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10().trade(
                                        {"U123": 0.5,
                                         "DU234": 0.3,
                                         "U999": 0.6,
                                         "DU111": 0.2
                                         })

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
                    'Account': 'DU234',
                    'Action': 'SELL',
                    'OrderRef': 'buy-below-10',
                    # close open position
                    'TotalQuantity': 100.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.5 allocation * 0.5 weight * 85K / 8.50 - 200
                    'TotalQuantity': 2300.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U999',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.6 allocation * 0.5 weight * 56K / 8.5
                    'TotalQuantity': 1976.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'DU111',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.2 allocation * 0.5 weight * 150K / 8.50 - (-600)
                    'TotalQuantity': 2365.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )
    def test_existing_positions_and_open_orders(self):
        """
        Tests that the orders DataFrame is correct after running a long only
        strategy allocated to multiple accounts where some of the accounts
        have existing positions and open orders.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 10
                return signals.astype(int)

            def signals_to_target_weights(self, signals, prices):
                return self.allocate_fixed_weights(signals, 0.5)

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
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
            balances = pd.DataFrame(dict(Account=["U123", "DU234", "U999", "DU111"],
                                         NetLiquidation=[85000, 450000, 56000, 150000],
                                         Currency=["USD", "USD", "USD", "USD"]))
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
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": 400
                },
                # this is the position we want, so no order will be needed
                {
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": 7941
                },
                {
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 300
                },
                {
                    "Account": "DU111",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": -300
                },

            ]
            return positions

        def mock_download_order_statuses(f, **kwargs):
            orders = [
                {
                    "Account": "U123",
                    "Action": "SELL",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Filled": 200,
                    "Remaining": 200,
                    "TotalQuantity": 400,
                    "Status": "Submitted"
                },
                # Next two orders are for same conid/account, should be summed
                {
                    "Account": "DU234",
                    "Action": "SELL",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Filled": 0,
                    "Remaining": 100,
                    "TotalQuantity": 200,
                    "Status": "Submitted"
                },
                {
                    "Account": "DU234",
                    "Action": "BUY",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Filled": 0,
                    "Remaining": 200,
                    "TotalQuantity": 200,
                    "Status": "Submitted"
                },
                {
                    "Account": "DU111",
                    "Action": "SELL",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Filled": 0,
                    "Remaining": 600,
                    "TotalQuantity": 600,
                    "Status": "Submitted"
                },

            ]
            json.dump(orders, f)
            f.seek(0)

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                with patch("moonshot.strategies.base.get_db_config", new=mock_get_db_config):

                                    orders = BuyBelow10().trade(
                                        {"U123": 0.5,
                                         "DU234": 0.3,
                                         "U999": 0.6,
                                         "DU111": 0.2
                                         })

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
                    'Account': 'DU234',
                    'Action': 'SELL',
                    'OrderRef': 'buy-below-10',
                    # close open position (300) + pending open position (100)
                    'TotalQuantity': 400.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.5 allocation * 0.5 weight * 85K / 8.50 - 400 (position) + 200 (order)
                    'TotalQuantity': 2300.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U999',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.6 allocation * 0.5 weight * 56K / 8.5
                    'TotalQuantity': 1976.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'DU111',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.2 allocation * 0.5 weight * 150K / 8.50 - (-900)
                    'TotalQuantity': 2665.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_price_magnifier_and_multiplier(self):
        """
        Tests that the orders DataFrame is correct after running a
        strategy using conids with price magnifiers and multipliers.
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

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def order_stubs_to_orders(self, orders, prices):
                orders["Exchange"] = "GLOBEX"
                orders["OrderType"] = 'MKT'
                orders["Tif"] = "DAY"
                return orders

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/Chicago"), periods=3, normalize=True)
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
                    34567: [
                        # Close
                        19.89,
                        11,
                        11.50,
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
                        "America/Chicago",
                        "FUT",
                        "USD",
                        None,
                        20
                    ],
                    23456: [
                        "America/Chicago",
                        "FUT",
                        "USD",
                        1,
                        50,
                    ],
                    34567: [
                        "America/Chicago",
                        "FUT",
                        "USD",
                        10,
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
                                         NetLiquidation=[85000],
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

                                    orders = BuyBelow10ShortAbove10Overnight().trade({"U123": 0.5})

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
                    # 0.5 allocation * 0.25 weight * 85K / multiplier 20 / 10.50
                    'TotalQuantity': 51,
                    'Exchange': 'GLOBEX',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # 0.5 allocation * 0.25 weight * 85K / multiplier 50 / 10.50
                    'TotalQuantity': 25,
                    'Exchange': 'GLOBEX',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 34567,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    # 0.5 allocation * 0.25 weight * 85K * magnifier 10 / 11.50
                    'TotalQuantity': 9239,
                    'Exchange': 'GLOBEX',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_apply_exchange_rates(self):
        """
        Tests that the orders DataFrame is correct after running a long-short
        strategy with varying exchange rates that need to be applied.
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "long-short-10"
            TIMEZONE = "Europe/Berlin"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def order_stubs_to_orders(self, orders, prices):
                orders["Exchange"] = "SMART"
                orders["OrderType"] = 'MKT'
                orders["Tif"] = "DAY"
                return orders

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="Europe/Berlin"), periods=3, normalize=True)
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
                        "Europe/Berlin",
                        "STK",
                        "EUR",
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
            balances = pd.DataFrame(dict(Account=["U123", "DU234"],
                                         NetLiquidation=[85000, 450000],
                                         Currency=["USD", "CAD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD","USD","CAD","CAD"],
                                      QuoteCurrency=["USD","EUR","USD","EUR"],
                                         Rate=[1.0,0.75,0.8,0.7]))
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

                                    orders = BuyBelow10ShortAbove10Overnight().trade(
                                        {"U123": 0.75, "DU234": 0.4})

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
                    # 0.75 allocation * 0.25 weight * 85K USD * 0.75 USD.EUR / 10.50
                    'TotalQuantity': 1138,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 12345,
                    'Account': 'DU234',
                    'Action': 'SELL',
                    'OrderRef': 'long-short-10',
                    # 0.4 allocation * 0.25 weight * 450K CAD * 0.7 CAD.EUR / 10.50
                    'TotalQuantity': 3000,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }, {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # 0.75 allocation * 0.25 weight * 85K USD * 1.0 USD.USD / 8.50
                    'TotalQuantity': 1875,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'DU234',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    # 0.4 allocation * 0.25 weight * 450K CAD * 0.8 CAD.USD / 8.50
                    'TotalQuantity': 4235,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )


    def test_forex(self):
        """
        Tests that the orders DataFrame is correct after running a forex
        strategy. For forex, the contract value is always 1, regardless of
        the price (1 EUR.USD is always worth 1 EUR regardless of exchange
        rate). Exchange rates between the base currency and fx pair are based
        on the Symbol, not the Currency field (e.g. trading EUR.USD with base
        currency USD requires exchange rate to EUR, not USD)
        """

        class BuyBelow10ShortAbove10Overnight(Moonshot):
            """
            A basic test strategy that buys below 10 and shorts above 10.
            """
            CODE = "fx-long-short-10"

            def prices_to_signals(self, prices):
                long_signals = prices.loc["Open"] <= 10
                short_signals = prices.loc["Open"] > 10
                signals = long_signals.astype(int).where(long_signals, -short_signals.astype(int))
                return signals

            def signals_to_target_weights(self, signals, prices):
                weights = self.allocate_fixed_weights(signals, 0.25)
                return weights

            def order_stubs_to_orders(self, orders, prices):
                orders["Exchange"] = "IDEALPRO"
                orders["OrderType"] = 'MKT'
                orders["Tif"] = "GTC"
                return orders

        def mock_get_historical_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Open"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Open
                        1.2,
                        1.1,
                        1.25
                    ],
                    23456: [
                        # Open
                        100.89,
                        112.0,
                        118.50,
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
                        "USD",
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

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123"],
                                         NetLiquidation=[85000],
                                         Currency=["USD"]))
            balances.to_csv(f, index=False)
            f.seek(0)

        def mock_download_exchange_rates(f, **kwargs):
            rates = pd.DataFrame(dict(BaseCurrency=["USD","USD"],
                                      QuoteCurrency=["USD","EUR"],
                                         Rate=[1.0, 0.7]))
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

                                    orders = BuyBelow10ShortAbove10Overnight().trade({"U123": 0.5})

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
                    'OrderRef': 'fx-long-short-10',
                    # 0.5 allocation * 0.25 weight * 85K USD * 0.7 USD.EUR
                    'TotalQuantity': 7437,
                    'Exchange': 'IDEALPRO',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'fx-long-short-10',
                    # 0.5 allocation * 0.25 weight * 85K USD * 1 USD.USD
                    'TotalQuantity': 10625,
                    'Exchange': 'IDEALPRO',
                    'OrderType': 'MKT',
                    'Tif': 'GTC'
                }
            ]
        )

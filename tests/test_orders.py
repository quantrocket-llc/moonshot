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
from moonshot import Moonshot
from moonshot.exceptions import MoonshotError, MoonshotParameterError

class OrdersTestCase(unittest.TestCase):
    """
    Test cases related to creating orders.
    """

    def test_child_orders(self):
        """
        Tests that the orders DataFrame is correct when using orders_to_child_orders.
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
                orders["Tif"] = "Day"

                child_orders = self.orders_to_child_orders(orders)
                child_orders.loc[:, "OrderType"] = "MOC"

                orders = pd.concat([orders,child_orders])
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
             'OrderId',
             'ParentId',
             'OrderType',
             'Tif'}
        )
        # replace nan with 'nan' to allow equality comparisons
        orders = orders.where(orders.notnull(), 'nan')

        # strip timestamp from OrderId/ParentId field
        orders.loc[orders.OrderId.notnull(), "OrderId"] = orders.loc[orders.OrderId.notnull()].OrderId.str.split(".").str[0]
        orders.loc[orders.ParentId.notnull(), "ParentId"] = orders.loc[orders.ParentId.notnull()].ParentId.str.split(".").str[0]

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'Account': 'U123',
                    'Action': 'SELL',
                    'ConId': 12345,
                    'Exchange': 'SMART',
                    'OrderId': '0',
                    'OrderRef': 'long-short-10',
                    'OrderType': 'MKT',
                    'ParentId': 'nan',
                    'Tif': 'Day',
                    'TotalQuantity': 1012
                },
                {
                    'Account': 'U123',
                    'Action': 'BUY',
                    'ConId': 23456,
                    'Exchange': 'SMART',
                    'OrderId': '1',
                    'OrderRef': 'long-short-10',
                    'OrderType': 'MKT',
                    'ParentId': 'nan',
                    'Tif': 'Day',
                    'TotalQuantity': 1250
                },
                {
                    'Account': 'U123',
                    'Action': 'BUY',
                    'ConId': 12345,
                    'Exchange': 'SMART',
                    'OrderId': 'nan',
                    'OrderRef': 'long-short-10',
                    'OrderType': 'MOC',
                    'ParentId': '0',
                    'Tif': 'Day',
                    'TotalQuantity': 1012
                },
                {
                    'Account': 'U123',
                    'Action': 'SELL',
                    'ConId': 23456,
                    'Exchange': 'SMART',
                    'OrderId': 'nan',
                    'OrderRef': 'long-short-10',
                    'OrderType': 'MOC',
                    'ParentId': '1',
                    'Tif': 'Day',
                    'TotalQuantity': 1250
                }
            ]
        )

    def test_complain_if_reindex_like_orders_with_time_index_on_once_a_day_intraday_strategy(self):
        """
        Tests error handling when using reindex_like_orders on a once-a-day
        intraday strategy and passing Time in the index.
        """

        class ShortAbove10Intraday(Moonshot):

            CODE = "short-10"

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

            def order_stubs_to_orders(self, orders, prices):
                closes = prices.loc["Close"]
                prior_closes = closes.shift()
                prior_closes = self.reindex_like_orders(prior_closes, orders)

                orders["Exchange"] = "SMART"
                orders["OrderType"] = 'LMT'
                orders["LmtPrice"] = prior_closes
                orders["Tif"] = "Day"
                return orders

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

                                    with self.assertRaises(MoonshotError) as cm:
                                        ShortAbove10Intraday().trade({"U123": 0.5})

        self.assertIn("cannot reindex DataFrame like orders because DataFrame contains "
                      "'Time' in index, please take a cross-section first", repr(cm.exception))

    def test_reindex_like_orders(self):
        """
        Tests that the orders DataFrame is correct when using reindex_like_orders.
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
                closes = prices.loc["Close"]
                prior_closes = closes.shift()
                prior_closes = self.reindex_like_orders(prior_closes, orders)

                orders["Exchange"] = "SMART"
                orders["OrderType"] = 'LMT'
                orders["LmtPrice"] = prior_closes
                orders["Tif"] = "Day"
                return orders

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
                        11.25,
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
             'LmtPrice',
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
                    'TotalQuantity': 1012,
                    'Exchange': 'SMART',
                    'OrderType': 'LMT',
                    'LmtPrice': 11.0,
                    'Tif': 'Day'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'long-short-10',
                    'TotalQuantity': 1250,
                    'Exchange': 'SMART',
                    'OrderType': 'LMT',
                    'LmtPrice': 11.25,
                    'Tif': 'Day'
                }
            ]
        )

    def test_reindex_like_orders_continous_intraday(self):
        """
        Tests that the orders DataFrame is correct when using
        reindex_like_orders on a continuous intraday strategy.
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

            def order_stubs_to_orders(self, orders, prices):
                closes = prices.loc["Close"]
                prior_closes = closes.shift()
                prior_closes = self.reindex_like_orders(prior_closes, orders)

                orders["Exchange"] = "SMART"
                orders["OrderType"] = 'LMT'
                orders["LmtPrice"] = prior_closes
                orders["Tif"] = "Day"
                return orders

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
             'LmtPrice',
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
                    'TotalQuantity': 2439,
                    'Exchange': 'SMART',
                    'OrderType': 'LMT',
                    'LmtPrice': 8.67,
                    'Tif': 'Day'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'c-intraday-pivot-10',
                    'TotalQuantity': 4000,
                    'Exchange': 'SMART',
                    'OrderType': 'LMT',
                    'LmtPrice': 13.4,
                    'Tif': 'Day'
                }
            ]
        )
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
from moonshot.exceptions import MoonshotParameterError

class AllowRebalanceTestCase(unittest.TestCase):
    """
    Test cases for the ALLOW_REBALANCE param.
    """

    def test_allow_rebalance(self):
        """
        Tests that small rebalancing orders are allowed when there are
        existing positions and ALLOW_REBALANCE = True (the default).
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
                        9.50
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
            positions = [
                {
                    "Account": "U123",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 2240
                },
                {
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 7100
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
                    'Account': 'U123',
                    'Action': 'SELL',
                    'OrderRef': 'buy-below-10',
                    # 0.5 allocation * 0.5 weight * 85K / 9.50 - 2240
                    'TotalQuantity': 3,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 12345,
                    'Account': 'DU234',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.3 allocation * 0.5 weight * 450K / 9.50 - 7100
                    'TotalQuantity': 5,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_disable_rebalance(self):
        """
        Tests that rebalancing orders are not allowed when there are existing
        positions and ALLOW_REBALANCE = False. However, closing positions and
        switching sides is allowed.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"
            ALLOW_REBALANCE = False

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
                        9.50
                    ],
                    23456: [
                        # Close
                        8.9,
                        12,
                        10.50
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
                        None
                    ],
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
            positions = [
                {
                    # this position won't be rebalanced
                    "Account": "U123",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 200
                },
                {
                      # this position will switch sides
                      "Account": "DU234",
                      "OrderRef": "buy-below-10",
                      "ConId": 12345,
                      "Quantity": -4
                },
                {
                    # this position will be closed
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": -7
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
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.3 allocation * 0.5 weight * 450K / 9.50 - (-4)
                    'TotalQuantity': 7109.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'DU234',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0 - (-7)
                    'TotalQuantity': 7.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_min_rebalance(self):
        """
        Tests that rebalancing orders are only allowed when above the
        ALLOW_REBALANCE threshold, when there are existing positions and
        ALLOW_REBALANCE is a float.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"
            ALLOW_REBALANCE = 0.25

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
                        9.50
                    ],
                    23456: [
                        # Close
                        8.9,
                        12,
                        10.50
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
                        None
                    ],
                },
                index=master_fields
            )
            securities.columns.name = "ConId"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123", "DU234", "U999"],
                                         NetLiquidation=[85000, 450000, 200000],
                                         Currency=["USD", "USD", "USD"]))
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
                    # this position won't be rebalanced
                    "Account": "U123",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 2000
                },
                {
                    # this position will be rebalanced
                    "Account": "U999",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 3000
                    },
                {
                    # this position will switch sides
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": -4
                },
                {
                    # this position will be closed
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": -7
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
                                         "U999": 0.5
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
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.3 allocation * 0.5 weight * 450K / 9.50 - (-4)
                    'TotalQuantity': 7109.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 12345,
                    'Account': 'U999',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0.5 allocation * 0.5 weight * 200K / 9.50 - 3000
                    'TotalQuantity': 2263,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },

                {
                    'ConId': 23456,
                    'Account': 'DU234',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 0 - (-7)
                    'TotalQuantity': 7.0,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_complain_if_min_rebalance_not_float(self):
        """
        Tests error handling when ALLOW_REBALANCE is not a float or int.
        """

        class BuyBelow10(Moonshot):
            """
            A basic test strategy that buys below 10.
            """
            CODE = "buy-below-10"
            ALLOW_REBALANCE = "always"

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
                        9.50
                    ],
                    23456: [
                        # Close
                        8.9,
                        12,
                        10.50
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
                        None
                    ],
                },
                index=master_fields
            )
            securities.columns.name = "ConId"
            securities.T.to_csv(f, index=True, header=True)
            f.seek(0)

        def mock_download_account_balances(f, **kwargs):
            balances = pd.DataFrame(dict(Account=["U123", "DU234", "U999"],
                                         NetLiquidation=[85000, 450000, 200000],
                                         Currency=["USD", "USD", "USD"]))
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
                    # this position won't be rebalanced
                    "Account": "U123",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 2000
                },
                {
                    # this position will be rebalanced
                    "Account": "U999",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": 3000
                    },
                {
                    # this position will switch sides
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 12345,
                    "Quantity": -4
                },
                {
                    # this position will be closed
                    "Account": "DU234",
                    "OrderRef": "buy-below-10",
                    "ConId": 23456,
                    "Quantity": -7
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

                                    with self.assertRaises(MoonshotParameterError) as cm:
                                        BuyBelow10().trade(
                                            {"U123": 0.5,
                                             "DU234": 0.3,
                                             "U999": 0.5
                                             })

        self.assertIn(
            "invalid value for ALLOW_REBALANCE: always (should be a float)", repr(cm.exception))

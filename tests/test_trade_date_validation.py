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
import datetime
import pytz
from moonshot import Moonshot
from moonshot.exceptions import MoonshotError

class TradeDateValidationTestCase(unittest.TestCase):

    def test_complain_if_stale_data(self):
        """
        Tests error handling when data is older than today.
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
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
                index=idx
            )
            return pd.concat((prices, securities))

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):

            with self.assertRaises(MoonshotError) as cm:
                BuyBelow10().trade({"U123": 1.0})

        self.assertIn((
            "expected signal date {0} not found in weights DataFrame, is "
            "the underlying data up-to-date? (max date is 2018-05-03").format(
                pd.Timestamp.today(tz="America/New_York").date()), repr(cm.exception))


    def test_review_date(self):
        """
        Tests the use of review date to generate orders for earlier dates.
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

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
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
                index=idx
            )
            return pd.concat((prices, securities))

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

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):

                        orders_20180503 = BuyBelow10().trade({"U123": 1.0}, review_date="2018-05-03")
                        orders_20180501 = BuyBelow10().trade({"U123": 1.0}, review_date="2018-05-01")

        self.assertSetEqual(
            set(orders_20180503.columns),
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
            orders_20180503.to_dict(orient="records"),
            [
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef':
                    'buy-below-10',
                    # 1.0 allocation * 1.0 weight * 55K / 8.50
                    'TotalQuantity': 6471,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

        self.assertListEqual(
            orders_20180501.to_dict(orient="records"),
            [
                {
                    'ConId': 12345,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 1.0 allocation * 0.5 weight * 55K / 9
                    'TotalQuantity': 3056,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-10',
                    # 1.0 allocation * 0.5 weight * 55K / 9.89
                    'TotalQuantity': 2781,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_signal_date_from_timezone(self):
        """
        Tests that the signal date is derived from the TIMEZONE, if set.
        """

        class BuyBelow1(Moonshot):
            """
            A basic test strategy that buys below 1.
            """
            CODE = "buy-below-1"
            TIMEZONE = "America/Mexico_City"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 1
                return signals.astype(int)

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
                        0.99,
                        8.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
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
                index=idx
            )
            return pd.concat((prices, securities))

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

        def mock_pd_timestamp_now(tz=None):
            if tz == "America/Mexico_City":
                return pd.Timestamp("2018-05-02 10:40:00", tz=tz)
            elif tz:
                return datetime.datetime.now(tzinfo=pytz.timezone(tz))
            else:
                return datetime.datetime.now()

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.pd.Timestamp.now", new=mock_pd_timestamp_now):

                            orders = BuyBelow1().trade({"U123": 1.0})

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
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-1',
                    # 1.0 allocation * 1.0 weight * 55K / 0.99
                    'TotalQuantity': 55556,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_signal_date_from_inferred_timezone(self):
        """
        Tests that the signal date is derived from the inferred timezone.
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

            dt_idx = pd.DatetimeIndex(["2018-04-01", "2018-04-02", "2018-04-03"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        0.9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Close
                        0.89,
                        0.99,
                        8.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/Mexico_City",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "USD",
                        None,
                        None,
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

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

        def mock_pd_timestamp_now(tz=None):
            if tz == "America/Mexico_City":
                return pd.Timestamp("2018-04-01 10:40:00", tz=tz)
            elif tz:
                return datetime.datetime.now(tzinfo=pytz.timezone(tz))
            else:
                return datetime.datetime.now()

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.pd.Timestamp.now", new=mock_pd_timestamp_now):

                            orders = BuyBelow1().trade({"U123": 1.0})

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
                    'OrderRef': 'buy-below-1',
                    # 1.0 allocation * 0.5 weight * 55K / 0.9
                    'TotalQuantity': 30556,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                },
                {
                    'ConId': 23456,
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-1',
                    # 1.0 allocation * 0.5 weight * 55K / 0.89
                    'TotalQuantity': 30899,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    def test_complain_if_stale_data_and_suggest_CALENDAR(self):
        """
        Tests that the error message suggests setting CALENDAR when the data is stale by a single day.
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

            dt_idx = pd.DatetimeIndex(["2018-04-01", "2018-04-02", "2018-04-03"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    12345: [
                        # Close
                        0.9,
                        11,
                        10.50
                    ],
                    23456: [
                        # Close
                        0.89,
                        0.99,
                        8.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
            securities = pd.DataFrame(
                {
                    12345: [
                        "America/Mexico_City",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    23456: [
                        "America/Mexico_City",
                        "STK",
                        "USD",
                        None,
                        None,
                    ]
                },
                index=idx
            )
            return pd.concat((prices, securities))

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

        def mock_pd_timestamp_now(tz=None):
            if tz == "America/Mexico_City":
                return pd.Timestamp("2018-04-04 10:40:00", tz=tz)
            elif tz:
                return datetime.datetime.now(tzinfo=pytz.timezone(tz))
            else:
                return datetime.datetime.now()

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.pd.Timestamp.now", new=mock_pd_timestamp_now):

                            with self.assertRaises(MoonshotError) as cm:
                                BuyBelow1().trade({"U123": 1.0})

        self.assertIn((
            "expected signal date 2018-04-04 not found in weights DataFrame, is "
            "the underlying data up-to-date? (max date is 2018-04-03)"
            " If your strategy trades before the open and 2018-04-04 data "
            "is not expected, try setting CALENDAR = <exchange>").format(
                pd.Timestamp.today(tz="America/New_York")), repr(cm.exception))

    @patch("moonshot.strategies.base.list_calendar_statuses")
    def test_signal_date_from_calendar_timezone_if_open(self, mock_list_calendar_statuses):
        """
        Tests that the signal date is derived from the CALENDAR timezone, if
        set and the exchange is open.
        """

        class BuyBelow1(Moonshot):
            """
            A basic test strategy that buys below 1.
            """
            CODE = "buy-below-1"
            CALENDAR = "TSEJ"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 1
                return signals.astype(int)

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
                        0.99,
                        8.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
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
                index=idx
            )
            return pd.concat((prices, securities))

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

        def _mock_list_calendar_statuses():
            return {
                "TSEJ":{
                    "timezone": "Japan",
                    "status": "open",
                    "since": "2018-05-02T09:00:00",
                    "until": "2018-05-02T14:00:00"
                }
            }

        mock_list_calendar_statuses.return_value = _mock_list_calendar_statuses()

        def mock_pd_timestamp_now(tz=None):
            if tz == "Japan":
                return pd.Timestamp("2018-05-02 10:40:00", tz=tz)
            elif tz:
                return datetime.datetime.now(tzinfo=pytz.timezone(tz))
            else:
                return datetime.datetime.now()

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.pd.Timestamp.now", new=mock_pd_timestamp_now):

                            orders = BuyBelow1().trade({"U123": 1.0})

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
                    'Action': 'BUY',
                    'OrderRef': 'buy-below-1',
                    # 1.0 allocation * 1.0 weight * 55K / 0.99
                    'TotalQuantity': 55556,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

    @patch("moonshot.strategies.base.list_calendar_statuses")
    def test_signal_date_from_calendar_since_if_closed(self, mock_list_calendar_statuses):
        """
        Tests that the signal date is derived from the CALENDAR "since"
        value, if set and the exchange is closed (i.e. is derived from the
        exchange last open date).
        """

        class BuyBelow1(Moonshot):
            """
            A basic test strategy that buys below 1.
            """
            CODE = "buy-below-1"
            CALENDAR = "TSEJ"

            def prices_to_signals(self, prices):
                signals = prices.loc["Close"] < 1
                return signals.astype(int)

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
                        0.50
                    ],
                    23456: [
                        # Close
                        9.89,
                        0.99,
                        8.50,
                    ],
                 },
                index=idx
            )

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            idx = pd.MultiIndex.from_product((master_fields, [dt_idx[0]]), names=["Field", "Date"])
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
                index=idx
            )
            return pd.concat((prices, securities))

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

        # First, as a control, pretend the exchange is open; this should
        # raise an error
        def _mock_list_calendar_statuses():
            return {
                "TSEJ":{
                    "timezone": "Japan",
                    "status": "open",
                    "since": "2018-05-04T09:00:00",
                    "until": "2018-05-04T14:00:00"
                }
            }

        mock_list_calendar_statuses.return_value = _mock_list_calendar_statuses()

        def mock_pd_timestamp_now(tz=None):
            if tz == "Japan":
                return pd.Timestamp("2018-05-04 08:40:00", tz=tz)
            elif tz:
                return datetime.datetime.now(tzinfo=pytz.timezone(tz))
            else:
                return datetime.datetime.now()


        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.pd.Timestamp.now", new=mock_pd_timestamp_now):

                            with self.assertRaises(MoonshotError) as cm:
                                BuyBelow1().trade({"U123": 1.0})

        self.assertIn((
            "expected signal date 2018-05-04 not found in weights DataFrame, is "
            "the underlying data up-to-date? (max date is 2018-05-03"), repr(cm.exception))

        # Now pretend it's May 4 but the exchange was last open May 3
        def _mock_list_calendar_statuses():
            return {
                "TSEJ":{
                    "timezone": "Japan",
                    "status": "closed",
                    "since": "2018-05-03T14:00:00",
                    "until": "2018-05-04T09:00:00"
                }
            }

        mock_list_calendar_statuses.return_value = _mock_list_calendar_statuses()

        with patch("moonshot.strategies.base.get_historical_prices", new=mock_get_historical_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.pd.Timestamp.now", new=mock_pd_timestamp_now):

                            orders = BuyBelow1().trade({"U123": 1.0})

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
                    'OrderRef': 'buy-below-1',
                    # 1.0 allocation * 1.0 weight * 55K / 0.50
                    'TotalQuantity': 110000,
                    'Exchange': 'SMART',
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

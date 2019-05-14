# Copyright 2017 QuantRocket LLC - All Rights Reserved
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

import io
import pandas as pd
import numpy as np
import warnings
import time
import requests
import json
from moonshot.slippage import FixedSlippage
from moonshot.mixins import (
    WeightAllocationMixin,
    ReutersFundamentalsMixin
)
from moonshot.cache import Cache
from moonshot.exceptions import MoonshotError, MoonshotParameterError
from quantrocket.history import get_historical_prices, get_db_config
from quantrocket.master import list_calendar_statuses, download_master_file
from quantrocket.account import download_account_balances, download_exchange_rates
from quantrocket.blotter import list_positions, download_order_statuses

class Moonshot(
    WeightAllocationMixin,
    ReutersFundamentalsMixin):
    """
    Base class for Moonshot strategies.

    To create a strategy, subclass this class. Implement your trading logic in the class
    methods, and store your strategy parameters as class attributes.

    Class attributes include built-in Moonshot parameters which you can override, as well
    as your own custom parameters.

    To run a backtest, at minimum you must implement `prices_to_signals`, but in general you will
    want to implement the following methods (which are called in the order shown):

        `prices_to_signals` -> `signals_to_target_weights` -> `target_weights_to_positions` -> `positions_to_gross_returns`

    To trade (i.e. generate orders intended to be placed, but actually placed by other services
    than Moonshot), you must also implement `order_stubs_to_orders`. Order generation for trading
    follows the path shown below:

        `prices_to_signals` -> `signals_to_target_weights` -> `order_stubs_to_orders`

    Parameters
    ----------
    CODE : str, required
        the strategy code

    DB : str, required
        code of history db to pull data from

    DB_FIELDS : list of str, optional
        fields to retrieve from history db (defaults to ["Open", "High", "Low",
        "Close", "Volume"])

    DB_TIMES : list of str (HH:MM:SS), optional
        for intraday databases, only retrieve these times

    CONIDS : list of int, optional
        limit history db query to these conids

    UNIVERSES : list of str, optional
        limit history db query to these universes

    EXCLUDE_CONIDS : list of int, optional
        exclude these conids from history db query

    EXCLUDE_UNIVERSES : list of str, optional
        exclude these universes from history db query

    CONT_FUT : str, optional
        pass this cont_fut option to history db query (default None)

    LOOKBACK_WINDOW : int, optional
        get this many days additional data prior to the backtest start date or
        trade date to account for rolling windows. If set to None (the default),
        will use the largest value of any attributes ending with `*_WINDOW`, or
        252 if no such attributes, and will further pad window based on any
        `*_INTERVAL` attributes, which are interpreted as pandas offset aliases
        (for example `REBALANCE_INTERVAL = 'Q'`). Set to 0 to disable.

    MASTER_FIELDS : list of str, optional
        [DEPRECATED] get these fields from the securities master service. This
        parameter is deprecated and will be removed in a future release. For better
        performance, use `quantrocket.master.get_securities_reindexed_like` to get
        securities master data shaped like prices.

    NLV : dict, optional
        dict of currency:NLV for each currency represented in the strategy. Can
        alternatively be passed directly to backtest method.

    COMMISSION_CLASS : Class or dict of (sectype,exchange,currency):Class, optional
        the commission class to use. If strategy includes a mix of security types,
        exchanges, or currencies, you can pass a dict mapping tuples of
        (sectype,exchange,currency) to the different commission classes. By default
        no commission is applied.

    SLIPPAGE_CLASSES : iterable of slippage classes, optional
        one or more slippage classes. By default no slippage is applied.

    SLIPPAGE_BPS : float, optional
        amount on one-slippage to apply to each trade in BPS (for example, enter 5 to deduct
        5 BPS)

    BENCHMARK : int, optional
        the conid of a security in the historical data to use as the benchmark

    BENCHMARK_DB : str, optional
        the database containing the benchmark, if different from DB. BENCHMARK_DB
        should contain end-of-day data, not intraday (but can be used with intraday
        backtests).

    BENCHMARK_TIME : str (HH:MM:SS), optional
        use prices from this time of day as benchmark prices. Only applicable if
        benchmark prices originate in DB (not BENCHMARK_DB), DB contains intraday
        data, and backtest results are daily.

    TIMEZONE : str, optional
        convert timestamps to this timezone (if not provided, will be inferred
        from securities universe if possible)

    CALENDAR : str, optional
        use this exchange's trading calendar to determine which date's signals
        should be used for live trading. If the exchange is currently open,
        today's signals will be used. If currently closed, the signals corresponding
        to the last date the exchange was open will be used. If no calendar is specified,
        today's signals will be used.

    POSITIONS_CLOSED_DAILY : bool
        if True, positions in backtests that fall on adjacent days are assumed to
        be closed out and reopened each day rather than held continuously; this
        impacts commission and slippage calculations (default is False, meaning
        adjacent positions are assumed to be held continuously)

    ALLOW_REBALANCE: bool or float
        in live trading, whether to allow rebalancing of existing positions that
        are already on the correct side. If True (the default), allow rebalancing.
        If False, no rebalancing. If set to a positive decimal, allow rebalancing
        only when the existing position differs from the target position by at least
        this percentage. For example 0.5 means don't rebalance a position unless
        the position will change by +/-50%.

    Examples
    --------
    Example of a minimal strategy that runs on a history db called "mexi-stk-1d" and buys when
    the securities are above their 200-day moving average:

    >>> MexicoMovingAverage(Moonshot):
    >>>
    >>>     CODE = "mexi-ma"
    >>>     DB = "mexi-stk-1d"
    >>>     MAVG_WINDOW = 200
    >>>
    >>>     def prices_to_signals(self, prices):
    >>>         closes = prices.loc["Close"]
    >>>         mavgs = closes.rolling(self.MAVG_WINDOW).mean()
    >>>         signals = closes > mavgs.shift()
    >>>         return signals.astype(int)
    """
    CODE = None
    DB = None
    DB_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
    DB_TIMES = None
    CONIDS = None
    UNIVERSES = None
    EXCLUDE_CONIDS = None
    EXCLUDE_UNIVERSES = None
    CONT_FUT = None
    LOOKBACK_WINDOW = None
    MASTER_FIELDS = None
    NLV = None
    COMMISSION_CLASS = None
    SLIPPAGE_CLASSES = ()
    SLIPPAGE_BPS = 0
    BENCHMARK = None
    BENCHMARK_DB = None
    BENCHMARK_TIME = None
    TIMEZONE = None
    CALENDAR = None
    POSITIONS_CLOSED_DAILY = False
    ALLOW_REBALANCE = True

    def __init__(self):
        self.is_trade = False
        self.review_date = None # see trade() docstring
        self.is_backtest = False
        self._securities_master = None
        self._backtest_results = {}
        self._inferred_timezone = None
        self._signal_date = None # set by _weights_to_today_weights
        self._signal_time = None # set by _weights_to_today_weights

        if hasattr(self, "get_signals"):
            warnings.warn(
                "method name get_signals is deprecated and will be removed in a "
                "future release, please use prices_to_signals instead", DeprecationWarning)
            self.prices_to_signals = self.get_signals

        if hasattr(self, "allocate_weights"):
            warnings.warn(
                "method name allocate_weights is deprecated and will be removed in a "
                "future release, please use signals_to_target_weights instead", DeprecationWarning)
            self.signals_to_target_weights = self.allocate_weights

        if hasattr(self, "simulate_positions"):
            warnings.warn(
                "method name simulate_positions is deprecated and will be removed in a "
                "future release, please use target_weights_to_positions instead", DeprecationWarning)
            self.target_weights_to_positions = self.simulate_positions

        if hasattr(self, "simulate_gross_returns"):
            warnings.warn(
                "method name simulate_gross_returns is deprecated and will be removed in a "
                "future release, please use positions_to_gross_returns instead", DeprecationWarning)
            self.positions_to_gross_returns = self.simulate_gross_returns

    def prices_to_signals(self, prices):
        """
        From a DataFrame of prices, return a DataFrame of signals. By convention,
        signals should be 1=long, 0=cash, -1=short.

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            signals

        Examples
        --------
        Buy when the close is above yesterday's 50-day moving average:

        >>> def prices_to_signals(self, prices):
        >>>     closes = prices.loc["Close"]
        >>>     mavgs = closes.rolling(50).mean()
        >>>     signals = closes > mavgs.shift()
        >>>     return signals.astype(int)
        """
        raise NotImplementedError("strategies must implement prices_to_signals")

    def signals_to_target_weights(self, signals, prices):
        """
        From a DataFrame of signals, return a DataFrame of target weights.

        Whereas signals indicate the direction of the trades, weights
        indicate both the direction and size. For example, -0.5 means a short
        position equal to 50% of the equity allocated to the strategy.

        Weights are used to help create orders in live trading, and to help
        simulate executed positions in backtests.

        The default implemention of this method evenly divides allocated
        capital among the signals each period, but it is intended to be
        overridden by strategy subclasses.

        A variety of built-in weight allocation algorithms are provided by
        and documented under `moonshot.mixins.WeightAllocationMixin`.

        Parameters
        ----------
        signals : DataFrame, required
            a DataFrame of signals

        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame
            of price/market data

        Returns
        -------
        DataFrame
            weights

        Examples
        --------
        The default implementation is shown below:

        >>> def signals_to_target_weights(self, signals, prices):
        >>>     weights = self.allocate_equal_weights(signals) # provided by moonshot.mixins.WeightAllocationMixin
        >>>     return weights
        """
        weights = self.allocate_equal_weights(signals)
        return weights

    def target_weights_to_positions(self, weights, prices):
        """
        From a DataFrame of target weights, return a DataFrame of simulated
        positions.

        The positions should shift the weights based on when the weights
        would be filled in live trading.

        By default, assumes the position are taken in the period after the
        weights were allocated. Intended to be overridden by strategy
        subclasses.

        Parameters
        ----------
        weights : DataFrame, required
            a DataFrame of weights

        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            positions

        Examples
        --------
        The default implemention is shown below (enter position in the period after
        signal generation/weight allocation):

        >>> def target_weights_to_positions(self, weights, prices):
        >>>     positions = weights.shift()
        >>>     return positions
        """
        positions = weights.shift()
        return positions

    def positions_to_gross_returns(self, positions, prices):
        """
        From a DataFrame of positions, return a DataFrame of returns before
        commissions and slippage.

        By default, assumes entry on the close on the period the position is
        taken and calculates the return through the following period's close.
        Intended to be overridden by strategy subclasses.

        Parameters
        ----------
        positions : DataFrame, required
            a DataFrame of positions

        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            gross returns

        Examples
        --------
        The default implementation is shown below:

        >>> def positions_to_gross_returns(self, positions, prices):
        >>>     closes = prices.loc["Close"]
        >>>     gross_returns = closes.pct_change() * positions.shift()
        >>>     return gross_returns
        """
        closes = prices.loc["Close"]
        gross_returns = closes.pct_change() * positions.shift()
        return gross_returns

    def order_stubs_to_orders(self, orders, prices):
        """
        From a DataFrame of order stubs, creates a DataFrame of fully
        specified orders.

        Parameters
        ----------
        orders : DataFrame
            a DataFrame of order stubs, with columns ConId, Account, Action,
            OrderRef, and TotalQuantity

        prices : DataFrame
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            a DataFrame of fully specified orders, with (at minimum) columns
            Exchange, Tif, OrderType added

        Examples
        --------
        The orders DataFrame provided to this method resembles the following:

        >>> print(orders)
            ConId  Account Action     OrderRef  TotalQuantity
        0   12345   U12345   SELL  my-strategy            100
        1   12345   U55555   SELL  my-strategy             50
        2   23456   U12345    BUY  my-strategy            100
        3   23456   U55555    BUY  my-strategy             50
        4   34567   U12345    BUY  my-strategy            200
        5   34567   U55555    BUY  my-strategy            100

        The default implemention creates MKT DAY orders routed to SMART and is
        shown below:

        >>> def order_stubs_to_orders(self, orders, prices):
        >>>     orders["Exchange"] = "SMART"
        >>>     orders["OrderType"] = "MKT"
        >>>     orders["Tif"] = "DAY"
        >>>     return orders

        Set a limit price equal to the prior closing price:

        >>> closes = prices.loc["Close"]
        >>> prior_closes = closes.shift()
        >>> prior_closes = self.reindex_like_orders(prior_closes, orders)
        >>> orders["OrderType"] = "LMT"
        >>> orders["LmtPrice"] = prior_closes
        """
        orders["Exchange"] = "SMART"
        orders["OrderType"] = "MKT"
        orders["Tif"] = "DAY"
        return orders

    def reindex_like_orders(self, df, orders):
        """
        Reindexes a DataFrame (having ConIds as columns and dates as index)
        to match the shape of the orders DataFrame.

        Parameters
        ----------
        df : DataFrame, required
            a DataFrame of arbitrary values with ConIds as columns and
            dates as index

        orders : DataFrame, required
            an orders DataFrame with a ConId column

        Returns
        -------
        Series
            a Series with an index matching orders

        Examples
        --------
        Calculate prior closes (assuming daily bars) and reindex like
        orders:

        >>> closes = prices.loc["Close"]
        >>> prior_closes = closes.shift()
        >>> prior_closes = self.reindex_like_orders(prior_closes, orders)

        Calculate prior closes (assuming 30-min bars) and reindex like
        orders:

        >>> session_closes = prices.loc["Close"].xs("15:30:00", level="Time")
        >>> prior_closes = session_closes.shift()
        >>> prior_closes = self.reindex_like_orders(prior_closes, orders)

        """
        df = df.loc[self._signal_date]
        if "Time" in df.index.names:
            if not self._signal_time:
                raise MoonshotError(
                    "cannot reindex DataFrame like orders because DataFrame contains "
                    "'Time' in index, please take a cross-section first, for example: "
                    "`my_dataframe.xs('15:45:00', level='Time')`")
            df = df.loc[self._signal_time]

        df.name = "_MoonshotOther"
        df = orders.join(df, on="ConId")._MoonshotOther
        df.name = None
        return df

    def orders_to_child_orders(self, orders):
        """
        From a DataFrame of orders, returns a DataFrame of child orders
        (bracket orders) to be submitted if the parent orders fill.

        An OrderId column will be added to the orders DataFrame, and child
        orders will be linked to it via a ParentId column. The Action
        (BUY/SELL) will be reversed on the child orders but otherwise the
        child orders will be identical to the parent orders.

        Parameters
        ----------
        orders : DataFrame, required
            an orders DataFrame

        Returns
        -------
        DataFrame
            a DataFrame of child orders

        Examples
        --------
        >>> orders.head()
            ConId   Action  TotalQuantity Exchange OrderType  Tif
        0   12345      BUY            200    SMART       MKT  Day
        1   23456      BUY            400    SMART       MKT  Day
        >>> child_orders = self.orders_to_child_orders(orders)
        >>> child_orders.loc[:, "OrderType"] = "MOC"
        >>> orders = pd.concat([orders,child_orders])
        >>> orders.head()
            ConId   Action  TotalQuantity Exchange OrderType  Tif  OrderId  ParentId
        0   12345      BUY            200    SMART       MKT  Day        0       NaN
        1   23456      BUY            400    SMART       MKT  Day        1       NaN
        0   12345     SELL            200    SMART       MOC  Day      NaN         0
        1   23456     SELL            400    SMART       MOC  Day      NaN         1
        """
        if "OrderId" not in orders.columns:
            orders["OrderId"] = orders.index.astype(str) + ".{0}".format(time.time())
        child_orders = orders.copy()
        child_orders.rename(columns={"OrderId":"ParentId"}, inplace=True)
        child_orders.loc[orders.Action=="BUY", "Action"] = "SELL"
        child_orders.loc[orders.Action=="SELL", "Action"] = "BUY"
        return child_orders

    def _quantities_to_order_stubs(self, quantities):
        """
        From a DataFrame of quantities to be ordered (with ConIds as index,
        Accounts as columns), returns a DataFrame of order stubs.

        quantities in:

        Account   U12345  U55555
        ConId
        12345       -100     -50
        23456        100      50
        34567        200     100

        order_stubs out:

            ConId  Account Action     OrderRef  TotalQuantity
        0   12345   U12345   SELL  my-strategy            100
        1   12345   U55555   SELL  my-strategy             50
        2   23456   U12345    BUY  my-strategy            100
        3   23456   U55555    BUY  my-strategy             50
        4   34567   U12345    BUY  my-strategy            200
        5   34567   U55555    BUY  my-strategy            100

        """
        quantities.index.name = "ConId"
        quantities.columns.name = "Account"
        quantities = quantities.stack()
        quantities.name = "Quantity"
        order_stubs = quantities.to_frame().reset_index()
        order_stubs["Action"] = np.where(order_stubs.Quantity > 0, "BUY", "SELL")
        order_stubs = order_stubs.loc[order_stubs.Quantity != 0].copy()
        order_stubs["OrderRef"] = self.CODE
        order_stubs["TotalQuantity"] = order_stubs.Quantity.abs()
        order_stubs = order_stubs.drop("Quantity",axis=1)

        return order_stubs

    def _get_nlv(self):
        """
        Return a dict of currency:NLV for each currency in the strategy. By
        default simply returns the NLV class attribute.
        """
        return self.NLV

    def _positions_to_turnover(self, positions):
        """
        Given a dataframe of positions, returns a dataframe of turnover. 0
        indicates no turnover; 1 indicates going from 100% short to cash or
        cash to 100% long (for example), and vice versa; and 2 indicates
        going from 100% short to %100 long (for example).
        """
        # Intraday trades are opened and closed each day there's a position,
        # so the turnover is twice the positions.
        if self.POSITIONS_CLOSED_DAILY:
            turnover = positions * 2
        else:
            turnover = positions.fillna(0).diff()
        return turnover.abs()

    def _weights_to_today_weights(self, weights, prices):
        """
        From a DataFrame of target weights, extract the row that contains the
        weights that should be used for today's trading. Returns a Series of
        weights by conid:

        ConId
        12345  -0.2
        23456     0
        34567   0.1

        The date whose weights are selected is usually today, but if CALENDAR
        is used and the market is closed it will be the date when the market
        closed. Can also be overridden by review_date.

        For intraday strategies, the time whose weights are selected is the
        latest time that is earlier than the time at which the strategy is
        running.
        """

        # First, get the signal date

        # Use review_date if set
        if self.review_date:
            dt = pd.Timestamp(self.review_date)

        # Else use trading calendar if provided
        elif self.CALENDAR:
            status = list_calendar_statuses([self.CALENDAR])[self.CALENDAR]
            # If the exchange if closed, the signals should correspond to the
            # date the exchange was last open
            if status["status"] == "closed":
                dt = pd.Timestamp(status["since"])
            # If the exchange is open, the signals should correspond to
            # today's date
            else:
                dt = pd.Timestamp.now(tz=status["timezone"])

        # If no trading calendar, use today's date (in strategy timezone)
        else:
            tz = self.TIMEZONE or self._inferred_timezone
            dt = pd.Timestamp.now(tz=tz)

        # Keep only the date as the signal_date
        self._signal_date = pd.Timestamp(dt.date())

        # extract the current time (or review date time)
        trade_time = dt.strftime("%H:%M:%S")

        weights_is_intraday = "Time" in weights.index.names

        try:
            today_weights = weights.loc[self._signal_date]
        except KeyError:
            if weights_is_intraday:
                max_date = weights.index.get_level_values("Date").max()
            else:
                max_date = weights.index.max()

            msg = ("expected signal date {0} not found in target weights DataFrame, "
                   "is the underlying data up-to-date? (max date is {1})")
            if not self.CALENDAR and not weights_is_intraday and self._signal_date.date() - max_date.date() == pd.Timedelta(days=1):
                msg += (" If your strategy trades before the open and {0} data "
                        "is not expected, try setting CALENDAR = <exchange>")
            raise MoonshotError(msg.format(
                self._signal_date.date().isoformat(),
                max_date.date().isoformat()))

        if not weights_is_intraday:
            print("using target weights for {0} to create orders".format(self._signal_date.date().isoformat()))
            return today_weights

        # For intraday strategies, select the weights from the latest time
        # that is earlier than the trade time. Note that we select the
        # expected time from the entire weights DataFrame, which will result
        # in a failure if that time is missing for the trade date
        unique_times = weights.index.get_level_values("Time").unique()
        self._signal_time = unique_times[unique_times < trade_time].max()
        if pd.isnull(self._signal_time):
            msg = (
                "cannot determine which target weights to use for orders because "
                "target weights DataFrame contains no times earlier than trade time {0} "
                "for signal date {1}".format(
                    trade_time,
                    self._signal_date.date().isoformat()))

            if self.review_date:
                msg += ", please adjust the review_date"
            raise MoonshotError(msg)

        # get_historical_prices inserts all times into each day's index, thus
        # the signal_time will be in the weights DataFrame even if the data
        # is stale. Instead, to validate the data, we make sure that there is
        # at least one nonnull field in the prices DataFrame at the
        # signal_time on the signal_date
        today_prices = prices.xs(self._signal_date, level="Date")
        notnull_today_prices = today_prices[today_prices.notnull().any(axis=1)]

        try:
            no_signal_time_prices = notnull_today_prices.xs(self._signal_time, level="Time").empty
        except KeyError:
            no_signal_time_prices = True

        if no_signal_time_prices:
            msg = ("no {0} data found in prices DataFrame for signal date {1}, "
                   "is the underlying data up-to-date? (max time for {1} "
                   "is {2})")
            notnull_max_date = notnull_today_prices.iloc[-1].name[-1]
            raise MoonshotError(msg.format(
                self._signal_time,
                self._signal_date.date().isoformat(),
                notnull_max_date))

        today_weights = today_weights.loc[self._signal_time]

        print("using target weights for {0} at {1} to create orders".format(
            self._signal_date.date().isoformat(),
            self._signal_time))

        return today_weights

    def _get_commissions(self, positions, prices):
        """
        Returns the commissions to be subtracted from the returns.
        """
        if not self.COMMISSION_CLASS:
            return pd.DataFrame(0, index=positions.index, columns=positions.columns)

        turnover = self._positions_to_turnover(positions)
        contract_values = self._get_contract_values(prices)

        prices_is_intraday = "Time" in prices.index.names
        positions_is_intraday = "Time" in positions.index.names

        if prices_is_intraday and not positions_is_intraday:
            contract_values = contract_values.groupby(
                contract_values.index.get_level_values("Date")).first()

        fields = prices.index.get_level_values("Field").unique()
        if "Nlv" in self._securities_master.columns:
            nlvs = contract_values.apply(lambda x: self._securities_master.Nlv, axis=1)
        else:
            nlvs = None

        # handle the case of only one commission class
        if not isinstance(self.COMMISSION_CLASS, dict):
            commissions = self.COMMISSION_CLASS.get_commissions(contract_values, turnover=turnover, nlvs=nlvs)
            return commissions

        # handle multiple commission classes per sectype/exchange/currency

        # first, tuple-ize the dict keys in case they are lists
        commission_classes = {}
        for sec_group, commission_cls in self.COMMISSION_CLASS.items():
            commission_classes[tuple(sec_group)] = commission_cls

        defined_sec_groups = set([tuple(k) for k in commission_classes.keys()])

        # Reindex master fields like contract_values
        sec_types = contract_values.apply(lambda x: self._securities_master.SecType, axis=1)
        exchanges = contract_values.apply(lambda x: self._securities_master.PrimaryExchange, axis=1)
        currencies = contract_values.apply(lambda x: self._securities_master.Currency, axis=1)

        required_sec_groups = set([
            tuple(s.split("|")) for s in (sec_types+"|"+exchanges+"|"+currencies).iloc[-1].unique()])
        missing_sec_groups = required_sec_groups - defined_sec_groups
        if missing_sec_groups:
            raise MoonshotParameterError("expected a commission class for each combination of (sectype,exchange,currency) "
                                         "but none is defined for {0}".format(
                                             ", ".join(["({0})".format(",".join(t)) for t in missing_sec_groups])))

        all_commissions = pd.DataFrame(None, index=positions.index, columns=positions.columns)

        for sec_group in required_sec_groups:
            commission_cls = commission_classes[sec_group]
            sec_type, exchange, currency = sec_group

            sec_group_commissions = commission_cls.get_commissions(
                contract_values, turnover=turnover, nlvs=nlvs)

            in_sec_group = (sec_types == sec_type) & (exchanges == exchange) & (currencies == currency)
            all_commissions = sec_group_commissions.where(in_sec_group, all_commissions)

        return all_commissions

    def _get_slippage(self, positions, prices):
        """
        Returns the slippage to be subtracted from the returns.
        """
        turnover = self._positions_to_turnover(positions)
        slippage = pd.DataFrame(0, index=turnover.index, columns=turnover.columns)

        slippage_classes = self.SLIPPAGE_CLASSES or ()
        if not isinstance(slippage_classes, (list, tuple)):
            slippage_classes = [slippage_classes]
        for slippage_class in slippage_classes:
            slippage += slippage_class().get_slippage(turnover, positions, prices)

        if self.SLIPPAGE_BPS:
            slippage += FixedSlippage(self.SLIPPAGE_BPS/10000.0).get_slippage(turnover, positions, prices)

        return slippage.fillna(0)

    def _constrain_weights(self, weights, prices):
        """
        Constrains the weights by the quantity constraints defined in
        limit_position_sizes.
        """

        max_quantities_for_longs, max_quantities_for_shorts = self.limit_position_sizes(prices)
        if max_quantities_for_longs is None and max_quantities_for_shorts is None:
            return weights

        if "Nlv" not in self._securities_master.columns:
            raise MoonshotParameterError("must provide NLVs if using limit_position_sizes")

        contract_values = self._get_contract_values(prices)
        contract_values = contract_values.fillna(method="ffill")
        nlvs_in_trade_currency = contract_values.apply(lambda x: self._securities_master.Nlv, axis=1)

        prices_is_intraday = "Time" in prices.index.names
        weights_is_intraday = "Time" in weights.index.names

        if prices_is_intraday and not weights_is_intraday:
            # we somewhat arbitrarily pick the contract value as of the
            # earliest time of day; this contract value might be somewhat
            # stale but it avoids the possible lookahead bias of using, say,
            # the contract value as of the latest time of day. We could ask
            # the user to supply a time but that is rather clunky.
            earliest_time = prices.index.get_level_values("Time").unique().min()
            contract_values = contract_values.xs(earliest_time, level="Time")
            nlvs_in_trade_currency = nlvs_in_trade_currency.xs(earliest_time, level="Time")

        # Convert weights to quantities
        trade_values_in_trade_currency = weights * nlvs_in_trade_currency
        quantities = trade_values_in_trade_currency / contract_values.where(contract_values > 0)
        quantities = quantities.round().fillna(0).astype(int)

        # Constrain quantities
        if max_quantities_for_longs is not None:
            max_quantities_for_longs = max_quantities_for_longs.abs()
            quantities = max_quantities_for_longs.where(
                quantities > max_quantities_for_longs, quantities)
        if max_quantities_for_shorts is not None:
            max_quantities_for_shorts = -max_quantities_for_shorts.abs()
            quantities = max_quantities_for_shorts.where(
                quantities < max_quantities_for_shorts, quantities)

        # Convert quantities back to weights
        target_trade_values_in_trade_currency = quantities * contract_values
        weights = target_trade_values_in_trade_currency / nlvs_in_trade_currency

        return weights

    def limit_position_sizes(self, prices):
        """
        This method should return a tuple of DataFrames::

            return max_quantities_for_longs, max_quantities_for_shorts

        where the DataFrames define the maximum number of shares/contracts
        that can be held long and short, respectively. Maximum limits might
        be based on available liquidity (recent volume), shortable shares
        available, etc.

        The shape and alignment of the returned DataFrames should match that of the
        target_weights returned by `signals_to_target_weights`. Target weights will be
        reduced, if necessary, based on max_quantities_for_longs and max_quantities_for_shorts.

        Return None for one or both DataFrames to indicate "no limits."
        For example to limit shorts but not longs::

            return None, max_quantities_for_shorts

        Within a DataFrame, any None or NaNs will be treated as "no limit" for that
        particular security and date.

        Note that max_quantities_for_shorts can equivalently be represented with
        positive or negative numbers. This is OK::

                        AAPL
            2018-05-18   100
            2018-05-19   100

        This is also OK::

                        AAPL
            2018-05-18  -100
            2018-05-19  -100

        Both of the above DataFrames would mean: short no more than 100 shares of
        AAPL.

        Parameters
        ----------
        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        tuple of (DataFrame, DataFrame)
            max quantities for long, max quantities for shorts

        Examples
        --------
        Limit quantities to 1% of 15-day average daily volume:

        >>> def limit_position_sizes(self, prices):
        >>>     # assumes end-of-day bars, for intraday bars, use `.xs` to
        >>>     # select a time of day
        >>>     volumes = prices.loc["Volume"]
        >>>     mean_volumes = volumes.rolling(15).mean()
        >>>     max_shares = (mean_volumes * 0.01).round()
        >>>     max_quantities_for_longs = max_quantities_for_shorts = max_shares
        >>>     return max_quantities_for_longs, max_quantities_for_shorts
        """
        max_quantities_for_longs = None
        max_quantities_for_shorts = None
        return max_quantities_for_longs, max_quantities_for_shorts

    @classmethod
    def _get_lookback_window(cls):
        """
        Returns cls.LOOKBACK_WINDOW if set, otherwise infers the lookback
        window from `_WINDOW`, defaulting to 252. Then increases the lookback
        based on `_INTERVAL` attributes, which are interpreted as pandas
        frequencies (for example `REBALANCE_INTERVAL` = 'Q'). This ensures the
        lookback is sufficient when resampling to quarterly etc. for periodic
        rebalancing.
        """
        if cls.LOOKBACK_WINDOW is not None:
            return cls.LOOKBACK_WINDOW

        window_attrs = [getattr(cls, attr) for attr in dir(cls) if attr.endswith("_WINDOW")]
        windows = [attr for attr in window_attrs if isinstance(attr, int)]
        lookback_window = max(windows) if windows else 252

        # Add _INTERVAL if any
        offset_aliases = [getattr(cls, attr) for attr in dir(cls) if attr.endswith("_INTERVAL")]
        intervals = []
        for freq in offset_aliases:
            if not freq:
                continue
            try:
                periods = pd.date_range(start=pd.to_datetime('today'),
                                        freq=freq, periods=2)
            except ValueError:
                continue

            # Use the period date range to count bdays in period
            bdays = len(pd.bdate_range(start=periods[0], end=periods[1]))
            intervals.append(bdays)

        if intervals:
            lookback_window += max(intervals)

        return lookback_window

    def _load_master_file(self, conids, nlv=None):
        """
        Loads master file from cache or master service.
        """
        securities = None

        fields = [
            "Currency", "Multiplier", "PriceMagnifier",
            "PrimaryExchange", "SecType", "Symbol", "Timezone"]

        if self.is_backtest:
            # try to load from cache
            securities = Cache.get(conids, prefix="_master")

        if securities is None:

            # determine domain (all DBs must have same domain, which is check
            # by get_historical_prices, so not checked here)
            dbs = self.DB
            if not isinstance(dbs, (list, tuple)):
                dbs = [dbs]
            first_db = dbs[0]
            db_config = get_db_config(first_db)
            domain = db_config.get("domain", "main")

            # query master
            f = io.StringIO()
            download_master_file(
                f,
                conids=conids,
                fields=fields,
                domain=domain)

            securities = pd.read_csv(f, index_col="ConId")

            if self.is_backtest:
                Cache.set(conids, securities, prefix="_master")

        if not self.TIMEZONE:
            timezones = securities.Timezone.unique()

            if len(timezones) > 1:
                raise MoonshotParameterError(
                    "cannot infer timezone because multiple timezones are present "
                    "in data, please specify TIMEZONE explicitly (timezones: {0})".format(
                        ", ".join(timezones)))

            self._inferred_timezone = timezones[0]

        # Append NLV if applicable
        nlvs = nlv or self._get_nlv()
        if nlvs:

            # For Forex, store NLV based on the Symbol not Currency (100
            # EUR.USD = 100 EUR, not 100 USD)
            currencies = securities.Symbol.where(securities.SecType=="CASH", securities.Currency)

            missing_nlvs = set(currencies) - set(nlvs.keys())
            if missing_nlvs:
                raise MoonshotParameterError(
                    "NLV dict is missing values for required currencies: {0}".format(
                        ", ".join(missing_nlvs)))

            securities["Nlv"] = currencies.apply(lambda currency: nlvs.get(currency, None))

        self._securities_master = securities.sort_index()

    @classmethod
    def _get_start_date_with_lookback(cls, start_date):
        """
        Returns the start_date adjusted to incorporate the LOOKBACK_WINDOW,
        plus a buffer. LOOKBACK_WINDOW is measured in trading days, but we
        query the db in calendar days. Convert from weekdays (260 per year)
        to calendar days, assuming 25 holidays (NYSE has ~9 per year, TSEJ
        has ~19), plus a 10 day buffer to be safe.
        """
        lookback_window = cls._get_lookback_window()

        start_date = pd.Timestamp(start_date) - pd.Timedelta(
            days=lookback_window*365.0/(260 - 25) + 10)
        return start_date.date().isoformat()

    def get_historical_prices(self, start_date, end_date=None, nlv=None):
        """
        Downloads historical prices from a history db. Downloads security
        details from the master db.
        """
        if start_date:
            start_date = self._get_start_date_with_lookback(start_date)

        codes = self.DB
        if not isinstance(codes, (list, tuple)):
            codes = [self.DB]

        if not self.DB_TIMES and getattr(self, "DB_TIME_FILTERS", None):
            warnings.warn(
                "DB_TIME_FILTERS is deprecated and will be removed in a "
                "future release, please use DB_TIMES instead", DeprecationWarning)
            self.DB_TIMES = self.DB_TIME_FILTERS

        kwargs = dict(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            universes=self.UNIVERSES,
            conids=self.CONIDS,
            exclude_universes=self.EXCLUDE_UNIVERSES,
            exclude_conids=self.EXCLUDE_CONIDS,
            times=self.DB_TIMES,
            cont_fut=self.CONT_FUT,
            fields=self.DB_FIELDS,
            master_fields=self.MASTER_FIELDS,
            timezone=self.TIMEZONE
        )

        if not self.TIMEZONE:
            kwargs["infer_timezone"] = True

        prices = None

        if self.is_backtest:
            # try to load from cache
            prices = Cache.get(kwargs, prefix="_history")

        if prices is None:
            prices = get_historical_prices(**kwargs)
            if self.is_backtest:
                Cache.set(kwargs, prices, prefix="_history")

        self._load_master_file(prices.columns.tolist(), nlv=nlv)

        return prices

    def _prices_to_signals(self, prices):
        """
        Converts a prices DataFrame to a DataFrame of signals. This private
        method, which simply calls the user-modified public method
        `prices_to_signals`, exists for the benefit of the MoonshotML
        subclass, which overrides it.
        """
        return self.prices_to_signals(prices)

    def backtest(self, start_date=None, end_date=None, nlv=None, allocation=1.0,
                 label_conids=False):
        """
        Backtest a strategy and return a DataFrame of results.

        Parameters
        ----------
        start_date : str (YYYY-MM-DD), optional
            the backtest start date (default is to include all history in db)

        end_date : str (YYYY-MM-DD), optional
            the backtest end date (default is to include all history in db)

        nlv : dict
            dict of currency:nlv. Should contain a currency:nlv pair for
            each currency represented in the strategy

        allocation : float
            how much to allocate to the strategy

        label_conids : bool
            replace <ConId> with <Symbol>(<ConId>) in columns in output
            for better readability (default True)

        Returns
        -------
        DataFrame
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            backtest results
        """
        self.is_backtest = True
        allocation = allocation or 1.0

        prices = self.get_historical_prices(start_date, end_date, nlv=nlv)

        signals = self._prices_to_signals(prices)
        weights = self.signals_to_target_weights(signals, prices)
        weights = weights * allocation
        weights = self._constrain_weights(weights, prices)
        positions = self.target_weights_to_positions(weights, prices)
        gross_returns = self.positions_to_gross_returns(positions, prices)
        commissions = self._get_commissions(positions, prices)
        slippages = self._get_slippage(positions, prices)
        returns = gross_returns.fillna(0) - commissions - slippages
        turnover = self._positions_to_turnover(positions)

        total_holdings = (positions.fillna(0) != 0).astype(int)

        results_are_intraday = "Time" in signals.index.names

        all_results = dict(
            Signal=signals,
            Weight=weights,
            AbsWeight=weights.abs(),
            AbsExposure=positions.abs(),
            NetExposure=positions,
            Turnover=turnover,
            TotalHoldings=total_holdings,
            Commission=commissions,
            Slippage=slippages,
            Return=returns)

        # validate that custom backtest results are daily if results are
        # daily
        for custom_name, custom_df in self._backtest_results.items():

            if "Time" in custom_df.index.names and not results_are_intraday:
                raise MoonshotParameterError(
                    "custom DataFrame '{0}' won't concat properly with 'Time' in index, "
                    "please take a cross-section first, for example: "
                    "`my_dataframe.xs('15:45:00', level='Time')`".format(custom_name))

        all_results.update(self._backtest_results)

        if self.BENCHMARK:
            all_results["Benchmark"] = self._get_benchmark(prices, daily=not results_are_intraday)

        results = pd.concat(all_results)

        names = ["Field","Date"]
        if results.index.nlevels == 3:
            names.append("Time")

        results.index.set_names(names, inplace=True)

        if label_conids:
            symbols = self._securities_master.Symbol
            sec_types = self._securities_master.SecType
            currencies = self._securities_master.Currency
            symbols = symbols.astype(str).str.cat(currencies, sep=".").where(sec_types=="CASH", symbols)
            symbols_with_conids = symbols.astype(str) + "(" + symbols.index.astype(str) + ")"
            results.rename(columns=symbols_with_conids.to_dict(), inplace=True)

        # truncate at requested start_date
        if start_date:
            results = results.iloc[
                results.index.get_level_values("Date") >= pd.Timestamp(start_date)]

        return results

    def _get_benchmark(self, prices, daily=True):
        """
        Returns a 1-column DataFrame of benchmark prices, either extracted
        from prices or queried from BENCHMARK_DB if defined.

        BENCHMARK_DB, if used, must contain end-of-day prices.

        If prices are intraday and daily=True, the returned benchmark prices
        will be daily; if this is the case and benchmark prices are extracted
        from the prices DataFrame, BENCHMARK_TIME will be used to extract
        daily prices.

        If prices are intraday and daily=False, intraday benchmark prices
        will be returned; if this is the case and BENCHMARK_DB is used, the
        daily benchmark prices will be broadcast across the entire intraday
        timeframe.
        """

        if self.BENCHMARK_DB:
            try:
                benchmark_prices = get_historical_prices(
                    self.BENCHMARK_DB,
                    conids=self.BENCHMARK,
                    start_date=prices.index.get_level_values("Date").min(),
                    end_date=prices.index.get_level_values("Date").max(),
                    fields="Close"
                )
            except requests.HTTPError as e:
                raise MoonshotError("error querying BENCHMARK_DB {0}: {1}".format(
                    self.BENCHMARK_DB, repr(e)
                ))

            benchmark_prices = benchmark_prices.loc["Close"]

            if "Time" in benchmark_prices.index.names:
                raise MoonshotParameterError(
                    "only end-of-day databases are supported for BENCHMARK_DB but {0} is intraday".format(
                        self.BENCHMARK_DB))

            # Reindex benchmark prices like prices
            first_prices_field = prices.loc[prices.index.get_level_values("Field")[0]]

            # either reindex daily to daily (end-of-day backtests)
            if "Time" not in first_prices_field.index.names:
                benchmark_prices = benchmark_prices.reindex(index=first_prices_field.index)
            else:
                # or reindex daily to intraday daily (continuous intraday backtests)
                benchmark_prices = benchmark_prices.reindex(index=first_prices_field.index, level="Date")

                # and possibly back to daily (once-a-day intraday backtests)
                if daily:
                    benchmark_prices = benchmark_prices.groupby(
                        benchmark_prices.index.get_level_values("Date")).last()

            benchmark_db = self.BENCHMARK_DB
        else:
            benchmark_prices = prices
            benchmark_db = self.DB

            field = None
            fields = benchmark_prices.index.get_level_values("Field").unique()
            candidate_fields = ("Close", "Open", "Bid", "Ask", "High", "Low")
            for candidate in candidate_fields:
                if candidate in fields:
                    field = candidate
                    break
                else:
                    raise MoonshotParameterError("Cannot extract BENCHMARK {0} from {1} data without one of {2}".format(
                        self.BENCHMARK, benchmark_db, ", ".join(candidate_fields)))

            benchmark_prices = benchmark_prices.loc[field]

        try:
            benchmark = benchmark_prices[self.BENCHMARK]
        except KeyError:
            raise MoonshotError("BENCHMARK ConId {0} is not in {1} data".format(
                self.BENCHMARK, benchmark_db))

        # to avoid inserting an extra column in the results DataFrame,
        # store the benchmark prices under the first column
        if self.BENCHMARK_DB:
            benchmark.name = prices.columns[0]

        if "Time" in benchmark_prices.index.names and daily:
            if not self.BENCHMARK_TIME:
                raise MoonshotParameterError(
                    "Cannot extract BENCHMARK {0} from {1} data because prices contains intraday "
                    "prices but no BENCHMARK_TIME specified".format(self.BENCHMARK, benchmark_db))
            try:
                benchmark = benchmark.xs(self.BENCHMARK_TIME, level="Time")
            except KeyError:
                raise MoonshotError("BENCHMARK_TIME {0} is not in {1} data".format(
                    self.BENCHMARK_TIME, benchmark_db))

        return pd.DataFrame(benchmark)

    def save_to_results(self, name, df):
        """
        Saves the DataFrame to the backtest results output.

        DataFrame should have a Date or (Date, Time) index with
        ConIds as columns.

        Parameters
        ----------
        name : str, required
            the name to assign to the DataFrame

        df : DataFrame, required
            the DataFrame to save

        Returns
        -------
        None

        Examples
        --------
        Save moving averages of closing prices to results:

        >>> closes = prices.loc["Close"]
        >>> mavgs = closes.rolling(50).mean()
        >>> self.save_to_results("MAvg", mavgs)
        """

        # No-op if trading
        if self.is_trade:
            return

        reserved_names = [
            "Signal",
            "Weight",
            "AbsWeight",
            "AbsExposure",
            "NetExposure",
            "Turnover",
            "TotalHolding",
            "Commission",
            "Slippage",
            "Return",
            "Benchmark"
        ]
        if name in reserved_names:
            raise ValueError("name {0} is a reserved name".format(name))

        index_levels = df.index.names

        if "Date" not in index_levels:
            raise MoonshotParameterError(
                "custom DataFrame '{0}' must have index called 'Date' to concat properly, but has {1}".format(
                    name, ",".join([str(level_name) for level_name in index_levels])))

        if not hasattr(df.index.get_level_values("Date"), "date"):
            raise MoonshotParameterError("custom DataFrame '{0}' must have a DatetimeIndex to concat properly".format(name))

        self._backtest_results[name] = df

    def trade(self, allocations, review_date=None):
        """
        Run the strategy and create orders.

        Parameters
        ----------
        allocations : dict, required
            dict of account:allocation to strategy (expressed as a percentage of NLV)

        review_date : str (YYYY-MM-DD [HH:MM:SS]), optional
            generate orders as if it were this date, rather than using the latest date.
            For end-of-day strategies, provide a date; for intraday strategies a date
            and time

        Returns
        -------
        DataFrame
            orders
        """
        self.is_trade = True
        self.review_date = review_date

        start_date = review_date or pd.Timestamp.today()

        prices = self.get_historical_prices(start_date)
        prices_is_intraday = "Time" in prices.index.names

        signals = self._prices_to_signals(prices)

        weights = self.signals_to_target_weights(signals, prices)

        weights = self._weights_to_today_weights(weights, prices)

        allocations = pd.Series(allocations)
        # Out:
        # U12345    0.25
        # U55555    0.50

        # Multiply weights times allocations
        weights = weights.apply(lambda x: x * allocations)

        # Out:
        #        U12345  U55555
        # 12345  -0.050   -0.10
        # 23456   0.000    0.00
        # 34567   0.025    0.05
        contract_values = self._get_contract_values(prices)
        contract_values = contract_values.fillna(method="ffill").loc[self._signal_date]
        if prices_is_intraday:
            if self._signal_time:
                contract_values = contract_values.loc[self._signal_time]
            else:
                contract_values = contract_values.iloc[-1]
        contract_values = allocations.apply(lambda x: contract_values).T
        # Out:
        #          U12345    U55555
        # 12345     95.68     95.68
        # 23456   1500.00   1500.00
        # 34567   3600.00   3600.00

        currencies = self._securities_master.Currency
        sec_types = self._securities_master.SecType

        # For FX, exchange rate conversions should be based on the symbol,
        # not the currency (i.e. 100 EUR.USD = 100 EUR, not 100 USD)
        if (sec_types == "CASH").any():
            symbols = self._securities_master.Symbol
            currencies = currencies.where(sec_types != "CASH", symbols)

        f = io.StringIO()
        download_account_balances(
            f,
            latest=True,
            accounts=list(allocations.index),
            fields=["NetLiquidation"])

        balances = pd.read_csv(f, index_col="Account")

        f = io.StringIO()
        download_exchange_rates(
            f, latest=True,
            base_currencies=list(balances.Currency.unique()),
            quote_currencies=list(currencies.unique()))
        exchange_rates = pd.read_csv(f)

        nlvs = balances.NetLiquidation.reindex(allocations.index)
        # Out:
        # U12345 1000000
        # U55555  500000

        nlvs = weights.apply(lambda x: nlvs, axis=1)
        # Out:
        #        U12345 U55555
        # 12345 1000000 500000
        # 23456 1000000 500000
        # 34567 1000000 500000

        base_currencies = balances.Currency.reindex(allocations.index)
        # Out:
        # U12345 USD
        # U55555 EUR

        base_currencies = weights.apply(lambda x: base_currencies, axis=1)
        # Out:
        #        U12345 U55555
        # 12345     USD    EUR
        # 23456     USD    EUR
        # 34567     USD    EUR

        trade_currencies = allocations.apply(lambda x: currencies).T
        # Out:
        #        U12345 U55555
        # 12345     USD    USD
        # 23456     JPY    JPY
        # 34567     JPY    JPY

        base_currencies = base_currencies.stack()
        trade_currencies= trade_currencies.stack()
        base_currencies.name = "BaseCurrency"
        trade_currencies.name = "QuoteCurrency"
        currencies = pd.concat((base_currencies,trade_currencies), axis=1)
        # Out:
        #              BaseCurrency QuoteCurrency
        # 12345 U12345          USD           USD
        #       U55555          EUR           USD
        # 23456 U12345          USD           JPY
        #       U55555          EUR           JPY
        # 34567 U12345          USD           JPY
        #       U55555          EUR           JPY

        exchange_rates = pd.merge(currencies, exchange_rates, how="left",
                                  on=["BaseCurrency","QuoteCurrency"])
        exchange_rates.index = currencies.index
        exchange_rates.loc[(exchange_rates.BaseCurrency == exchange_rates.QuoteCurrency), "Rate"] = 1
        exchange_rates = exchange_rates.Rate.unstack()
        # Out:
        #        U12345  U55555
        # 12345    1.00    1.15
        # 23456  107.02  127.12
        # 34567  107.02  127.12

        # Convert weights to quantities
        target_trade_values_in_base_currency = weights * nlvs
        target_trade_values_in_trade_currency = target_trade_values_in_base_currency * exchange_rates
        target_quantities = target_trade_values_in_trade_currency / contract_values.where(contract_values > 0)
        target_quantities = target_quantities.round().fillna(0).astype(int)

        # Constrain quantities (we do this before applying the position diff in order to
        # mirror backtesting)
        max_quantities_for_longs, max_quantities_for_shorts = self.limit_position_sizes(prices)

        if max_quantities_for_longs is not None:
            max_quantities_for_longs_is_intraday = "Time" in max_quantities_for_longs.index.names
            max_quantities_for_longs = max_quantities_for_longs.loc[self._signal_date]
            if max_quantities_for_longs_is_intraday:
                max_quantities_for_longs = max_quantities_for_longs.loc[self._signal_time]
            max_quantities_for_longs = allocations.apply(lambda x: max_quantities_for_longs.abs()).T
            target_quantities = max_quantities_for_longs.where(
                target_quantities > max_quantities_for_longs, target_quantities)

        if max_quantities_for_shorts is not None:
            max_quantities_for_shorts_is_intraday = "Time" in max_quantities_for_shorts.index.names
            max_quantities_for_shorts = max_quantities_for_shorts.loc[self._signal_date]
            if max_quantities_for_shorts_is_intraday:
                max_quantities_for_shorts = max_quantities_for_shorts.loc[self._signal_time]
            max_quantities_for_shorts = allocations.apply(lambda x: -max_quantities_for_shorts.abs()).T
            target_quantities = max_quantities_for_shorts.where(
                target_quantities < max_quantities_for_shorts, target_quantities)

        # Adjust quantities based on existing positions_and_orders
        positions_and_orders = self._get_positions_and_orders(
            accounts=list(allocations.index),
            conids=list(target_quantities.index))

        if positions_and_orders.empty:
            net_quantities = target_quantities
        else:
            positions_and_orders = positions_and_orders.set_index(["ConId","Account"]).Quantity
            target_quantities = target_quantities.stack()
            target_quantities.index.set_names(["ConId","Account"], inplace=True)
            positions_and_orders = positions_and_orders.reindex(target_quantities.index).fillna(0)
            net_quantities = target_quantities - positions_and_orders

            # disable rebalancing as per ALLOW_REBALANCE
            if self.ALLOW_REBALANCE is not True:
                is_rebalance = (
                    ((target_quantities > 0) & (positions_and_orders > 0))
                    |
                    ((target_quantities < 0) & (positions_and_orders < 0))
                )
                zeroes = pd.Series(0, index=net_quantities.index)
                # ALLOW_REBALANCE = False: no rebalancing
                if not self.ALLOW_REBALANCE:
                    net_quantities = zeroes.where(is_rebalance, net_quantities)
                # ALLOW_REBALANCE = <float>: only rebalance if it changes the position
                # at least this much
                else:
                    if not isinstance(self.ALLOW_REBALANCE, (int, float)):
                        raise MoonshotParameterError(
                            "invalid value for ALLOW_REBALANCE: {0} (should be a float)".format(
                                self.ALLOW_REBALANCE))

                    rebalance_pcts = net_quantities/positions_and_orders.where(is_rebalance)
                    net_quantities = zeroes.where(
                        is_rebalance & (rebalance_pcts.abs() < self.ALLOW_REBALANCE),
                        net_quantities)

            net_quantities = net_quantities.unstack()

        if (net_quantities == 0).all().all():
            return

        order_stubs = self._quantities_to_order_stubs(net_quantities)
        orders = self.order_stubs_to_orders(order_stubs, prices)

        return orders

    def _get_positions_and_orders(self, accounts, conids):
        """
        Returns a DataFrame of current positions and open orders, for the
        purpose of generating an order diff in live trading.
        """
        # query positions
        positions = list_positions(
            order_refs=[self.CODE],
            accounts=accounts,
            conids=conids
        )

        if positions:
            positions = pd.DataFrame(positions)
        else:
            positions = pd.DataFrame(columns=["ConId","Account","Quantity"])

        # query open orders
        f = io.StringIO()
        download_order_statuses(
            f,
            order_refs=[self.CODE],
            accounts=accounts,
            conids=conids,
            open_orders=True,
            fields=["ConId","Account","OrderRef","Remaining","Action"],
            output="json")

        if f.getvalue():
            orders = json.load(f)
            orders = pd.DataFrame(orders)
            orders.loc[orders.Action == "SELL", "Remaining"] = -orders.loc[orders.Action == "SELL"].Remaining
            orders = orders.groupby([orders.ConId, orders.Account]).Remaining.sum().reset_index()
        else:
            orders = pd.DataFrame(columns=["ConId","Account","Remaining"])

        positions_and_orders = pd.merge(positions, orders, how="outer", on=["ConId","Account"])
        positions_and_orders.loc[:, "Quantity"] = positions_and_orders.Quantity.fillna(0) + positions_and_orders.Remaining.fillna(0)

        positions_and_orders = positions_and_orders[["ConId","Account","Quantity"]]

        return positions_and_orders

    def _get_contract_values(self, prices):
        """
        Return a DataFrame of contract values by multiplying prices times
        multipliers and dividing by price magnifiers.
        """
        # Find a price field we can use
        field = None
        fields = prices.index.get_level_values("Field").unique()
        candidate_fields = ("Close", "Open", "Bid", "Ask", "High", "Low")
        for candidate in candidate_fields:
            if candidate in fields:
                field = candidate
                break
        else:
            raise ValueError("Can't calculate contract values without one of {0}".format(
                ", ".join(candidate_fields)))

        closes = prices.loc[field]

        # For Forex, the value of the contract is simply 1 (1 EUR.USD = 1
        # EUR; 1 EUR.JPY = 1 EUR)
        if "CASH" in self._securities_master.SecType.values:
            sec_types = closes.apply(lambda x: self._securities_master.SecType, axis=1)
            closes = closes.where(sec_types != "CASH", 1)

        price_magnifiers = closes.apply(lambda x: self._securities_master.PriceMagnifier.fillna(1), axis=1)
        multipliers = closes.apply(lambda x: self._securities_master.Multiplier.fillna(1), axis=1)
        contract_values = closes / price_magnifiers * multipliers
        return contract_values

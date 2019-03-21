# Copyright 2019 QuantRocket LLC - All Rights Reserved
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

import pickle
try:
    import joblib
except ImportError:
    pass
import pandas as pd
import numpy as np
from moonshot.strategies.base import Moonshot
from moonshot.exceptions import MoonshotError, MoonshotParameterError
from moonshot.cache import Cache

class MoonshotML(Moonshot):
    """
    Base class for Moonshot machine learning strategies.

    To create a strategy, subclass this class. Implement your trading logic in the class
    methods, and store your strategy parameters as class attributes.

    Class attributes include built-in Moonshot parameters which you can override, as well
    as your own custom parameters.

    To run a backtest, at minimum you must implement `prices_to_features` and
    `predictions_to_signals`, but in general you will want to implement the
    following methods (which are called in the order shown):

        `prices_to_features` -> `predictions_to_signals` -> `signals_to_target_weights` -> `target_weights_to_positions` -> `positions_to_gross_returns`

    To trade (i.e. generate orders intended to be placed, but actually placed by other services
    than Moonshot), you must also implement `order_stubs_to_orders`. Order generation for trading
    follows the path shown below:

        `prices_to_features` -> `predictions_to_signals` -> `signals_to_target_weights` -> `order_stubs_to_orders`

    Parameters
    ----------
    CODE : str, required
        the strategy code

    MODEL : str, optional
        path of machine learning model to load (for scikit-learn models, a joblib or
        pickle file); alternatively model can be passed as a parameter to backtest
        method, in which case the MODEL parameter is ignored

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
    Example of a minimal strategy that runs on a history db called "usa-stk-1d", trains
    the model using the 1-day and 2-day returns, and buys when the machine learning
    model predicts a positive 1-day forward return:

    >>> class DemoMLStrategy(MoonshotML):
    >>>
    >>>     CODE = "demo-ml"
    >>>     DB = "usa-stk-1d"
    >>>     MODEL = "my_ml_model.pkl"
    >>>
    >>>     def prices_to_features(self, prices):
    >>>         closes = prices.loc["Close"]
    >>>         features = {}
    >>>         features["returns_1d"]= closes.pct_change()
    >>>         features["returns_2d"] = (closes - closes.shift(2)) / closes.shift(2)
    >>>         features["target"] = closes.pct_change().shift(-1)
    >>>         return features
    >>>
    >>>     def predictions_to_signals(self, predictions, prices):
    >>>         signals = predictions > 0
    >>>         return signals.astype(int)
    """

    MODEL = None

    def __init__(self, *args, **kwargs):
        super(MoonshotML, self).__init__(*args, **kwargs)
        self.model = None

    def _load_model(self):
        """
        Loads a model from file, either using joblib or pickle or keras.
        """
        if not self.MODEL:
            raise MoonshotParameterError("please specify a model file")

        if "joblib" in self.MODEL:
            self.model = joblib.load(self.MODEL)
        elif "keras.h5" in self.MODEL:
            from keras.models import load_model
            self.model = load_model(self.MODEL)
        else:
            with open(self.MODEL, "rb") as f:
                self.model = pickle.load(f)

    def prices_to_features(self, prices):
        """
        From a DataFrame of prices, return a dict of DataFrames of
        features to be provided to the machine learning model.

        Each DataFrame should have the same shape, with a Date or (Date, Time)
        index and conids as columns. (Moonshot will convert the DataFrames to
        the format expected by the machine learning model).

        If the features dict contains a key called "target", this will
        be interpreted as the targets (a.k.a. labels) to use in training
        and will be ignored for prediction. (Model training is not handled
        by the MoonshotML class.)

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        dict of DataFrames
            features

        Examples
        --------
        Predict next-day returns based on 1-day and 2-day returns:

        >>> def prices_to_features(self, prices):
        >>>     closes = prices.loc["Close"]
        >>>     features = {}
        >>>     features["returns_1d"]= closes.pct_change()
        >>>     features["returns_2d"] = (closes - closes.shift(2)) / closes.shift(2)
        >>>     features["target"] = closes.pct_change().shift(-1)
        >>>     return features
        """
        raise NotImplementedError("strategies must implement prices_to_features")

    def predictions_to_signals(self, predictions, prices):
        """
        From a DataFrame of predictions produced by a machine learning model,
        return a DataFrame of signals. By convention, signals should be
        1=long, 0=cash, -1=short.

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        predictions : DataFrame, required
            DataFrame of machine learning predictions

        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            signals

        Examples
        --------
        Buy when prediction is above zero.

        >>> def predictions_to_signals(self, predictions, prices):
        >>>     signals = predictions > 0
        >>>     return signals.astype(int)
        """
        raise NotImplementedError("strategies must implement predictions_to_signals")

    def backtest(self, model=None, start_date=None, end_date=None, nlv=None,
                allocation=1.0, label_conids=False):
        """
        Backtest a strategy and return a DataFrame of results.

        Parameters
        ----------
        model : object, optional
            machine learning model to use for predictions; if not specified,
            model will be loaded from file based on MODEL class attribute

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

        if model:
            self.model = model
        else:
            self._load_model()

        return super(MoonshotML, self).backtest(
            start_date=start_date, end_date=end_date, nlv=nlv,
            allocation=allocation, label_conids=label_conids)

    def _prices_to_signals(self, prices):
        """
        Converts a prices DataFrame to a DataFrame of signals, by:

        - converting prices to features
        - using the ML model to create predictions from the features
        - creating signals from the predictions
        """
        features = None

        # serve features from cache in backtests if possible. The features are cached
        # based on the index and columns of prices. If this file has been
        # edited more recently than the features were cached, the cache is
        # not used.
        if self.is_backtest:
            cache_key = [prices.index.tolist(), prices.columns.tolist()]
            features = Cache.get(cache_key, prefix="_features", unless_modified=self)

        if features is None:
            features = self.prices_to_features(prices)
            if self.is_backtest:
                Cache.set(cache_key, features, prefix="_features")

        # Don't use the target/label for predictions
        if "target" in features:
            del features["target"]

        all_features = []

        for i, feature in enumerate(features.values()):
            feature = feature.stack(dropna=False).fillna(0)
            if i == 0:
                # save stacked index for predictions output
                predictions_series_idx = feature.index
            all_features.append(feature.values)

        features = np.stack(all_features, axis=-1)
        del all_features

        # get predictions
        predictions = self.model.predict(features)
        del features

        # squeeze if needed (needed for Keras output)
        if len(predictions.shape) == 2 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(axis=-1)

        predictions = pd.Series(predictions, index=predictions_series_idx)
        predictions = predictions.unstack(level="ConId")

        # predictions to signals
        signals = self.predictions_to_signals(predictions, prices)
        return signals

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
        self._load_model()
        return super(MoonshotML, self).trade(allocations, review_date=review_date)

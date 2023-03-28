# Copyright 2017-2023 QuantRocket LLC - All Rights Reserved
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
from typing import Union, Any
try:
    import joblib
except ImportError:
    pass
import pandas as pd
import numpy as np
from moonshot.strategies.base import Moonshot
from moonshot.exceptions import MoonshotError, MoonshotParameterError
from moonshot._cache import Cache

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
        code of db to pull data from

    DB_FIELDS : list of str, optional
        fields to retrieve from db (defaults to ["Open", "Close", "Volume"])

    DB_TIMES : list of str (HH:MM:SS), optional
        for intraday databases, only retrieve these times

    DB_DATA_FREQUENCY : str, optional
        Only applicable when DB specifies a Zipline bundle. Whether to query minute or
        daily data.  If omitted, defaults to minute data for minute bundles and to daily
        data for daily bundles. This parameter only needs to be set to request daily data
        from a minute bundle. Possible choices: daily, minute (or aliases d, m).

    SIDS : list of str, optional
        limit db query to these sids

    UNIVERSES : list of str, optional
        limit db query to these universes

    EXCLUDE_SIDS : list of str, optional
        exclude these sids from db query

    EXCLUDE_UNIVERSES : list of str, optional
        exclude these universes from db query

    CONT_FUT : str, optional
        pass this cont_fut option to db query (default None)

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

    BENCHMARK : str, optional
        the sid of a security in the historical data to use as the benchmark

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

    CONTRACT_VALUE_REFERENCE_FIELD : str, optional
        the price field to use for determining contract values for the purpose of
        applying commissions and constraining weights in backtests and calculating
        order quantities in trading. Defaults to the first available of Close, Open,
        MinuteCloseClose, SecondCloseClose, LastPriceClose, BidPriceClose, AskPriceClose,
        TimeSalesLastPriceClose, TimeSalesFilteredLastPriceClose, LastPriceMean,
        BidPriceMean, AskPriceMean, TimeSalesLastPriceMean, TimeSalesFilteredLastPriceMean,
        MinuteOpenOpen, SecondOpenOpen, LastPriceOpen, BidPriceOpen, AskPriceOpen,
        TimeSalesLastPriceOpen, TimeSalesFilteredLastPriceOpen.

    ACCOUNT_BALANCE_FIELD : str or list of str, optional
        the account field to use for calculating order quantities as a percentage of
        account equity. Applies to trading only, not backtesting. Default is
        NetLiquidation. If a list of fields is provided, the minimum value is used.
        For example, ['NetLiquidation', 'PreviousEquity'] means to use the lesser of
        NetLiquidation or PreviousEquity to determine order quantities.

    Notes
    -----
    Usage Guide:

    * MoonshotML: https://qrok.it/dl/ms/moonshot-ml

    Examples
    --------
    Example of a minimal strategy that runs on a history db called "usa-stk-1d", trains
    the model using the 1-day and 2-day returns, and buys when the machine learning
    model predicts a positive 1-day forward return::

        import pandas as pd

        class DemoMLStrategy(MoonshotML):

            CODE = "demo-ml"
            DB = "usa-stk-1d"
            MODEL = "my_ml_model.pkl"

            def prices_to_features(self, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                features = {}
                features["returns_1d"]= closes.pct_change()
                features["returns_2d"] = (closes - closes.shift(2)) / closes.shift(2)
                targets = closes.pct_change().shift(-1)
                return features, targets

            def predictions_to_signals(self, predictions: pd.DataFrame, prices: pd.DataFrame):
                signals = predictions > 0
                return signals.astype(int)
    """

    MODEL: str = None
    """path of machine learning model to load (for scikit-learn models, a joblib or
    pickle file); alternatively model can be passed as a parameter to backtest
    method, in which case the MODEL parameter is ignored"""

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

    def prices_to_features(
        self,
        prices: pd.DataFrame
        ) -> tuple[
            Union[
                list[Union[pd.DataFrame, 'pd.Series[float]']],
                dict[str, Union[pd.DataFrame, 'pd.Series[float]']]
            ],
            Union[pd.DataFrame, 'pd.Series[float]']]:
        """
        From a DataFrame of prices, return a tuple of features and targets to be
        provided to the machine learning model.

        The returned features can be a list or dict of DataFrames, where each
        DataFrame is a feature and should have the same shape, with a Date or
        (Date, Time) index and sids as columns. (Moonshot will convert the
        DataFrames to the format expected by the machine learning model).

        Alternatively, a list or dict of Series can be provided, which is
        suitable if using multiple securities to make predictions for a
        single security (for example, an index).

        The returned targets should be a DataFrame or Series with an index
        matching the index of the features DataFrames or Series. Targets are
        used in training and are ignored for prediction. (Model training is
        not handled by the MoonshotML class.) Alternatively return None if
        using an already trained model.

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        tuple of (dict or list of DataFrames or Series, and DataFrame or Series)
            features and targets

        Notes
        -----
        Usage Guide:

        * MoonshotML: https://qrok.it/dl/ms/moonshot-ml

        Examples
        --------
        Predict next-day returns based on 1-day and 2-day returns::

            def prices_to_features(self, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                features = {}
                features["returns_1d"]= closes.pct_change()
                features["returns_2d"] = (closes - closes.shift(2)) / closes.shift(2)
                targets = closes.pct_change().shift(-1)
                return features, targets

        Predict next-day returns for a single security in the prices
        DataFrame using another security's returns::

            def prices_to_features(self, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                closes_to_predict = closes[12345]
                closes_to_predict_with = closes[23456]
                features = {}
                features["returns_1d"]= closes_to_predict_with.pct_change()
                features["returns_2d"] = (closes_to_predict_with - closes_to_predict_with.shift(2)) / closes_to_predict_with.shift(2)
                targets = closes_to_predict.pct_change().shift(-1)
                return features, targets
        """
        raise NotImplementedError("strategies must implement prices_to_features")

    def predictions_to_signals(
        self,
        predictions: Union[pd.DataFrame, 'pd.Series[float]'],
        prices: pd.DataFrame
        ) -> pd.DataFrame:
        """
        From a DataFrame of predictions produced by a machine learning model,
        return a DataFrame of signals. By convention, signals should be
        1=long, 0=cash, -1=short.

        The index of predictions will match the index of the features
        DataFrames or Series returned in prices_to_features.

        Must be implemented by strategy subclasses.

        Parameters
        ----------
        predictions : DataFrame or Series, required
            DataFrame of machine learning predictions

        prices : DataFrame, required
            multiindex (Field, Date) or (Field, Date, Time) DataFrame of
            price/market data

        Returns
        -------
        DataFrame
            signals

        Notes
        -----
        Usage Guide:

        * MoonshotML: https://qrok.it/dl/ms/moonshot-ml

        Examples
        --------
        Buy when prediction (a DataFrame) is above zero::

            def predictions_to_signals(self, predictions: pd.DataFrame, prices: pd.DataFrame):
                signals = predictions > 0
                return signals.astype(int)

        Buy a single security when the predictions (a Series) is above zero::

            def predictions_to_signals(self, predictions: pd.Series, prices: pd.DataFrame):
                closes = prices.loc["Close"]
                signals = pd.DataFrame(False, index=closes.index, columns=closes.columns)
                signals.loc[:, 12345] = predictions > 0
                return signals.astype(int)
        """
        raise NotImplementedError("strategies must implement predictions_to_signals")

    def backtest(
        self,
        model: Any = None,
        start_date: str = None,
        end_date: str = None,
        nlv: dict[str, float] = None,
        allocation: float = 1.0,
        label_sids: bool = False,
        no_cache: bool = False
        ) -> pd.DataFrame:
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

        label_sids : bool
            replace <Sid> with <Symbol>(<Sid>) in columns in output
            for better readability (default True)

        no_cache : bool
            don't use cached files even if available. Using cached files speeds
            up backtests but may be undesirable if underlying data has changed.
            See http://qrok.it/h/mcache to learn more about caching in Moonshot.

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
            allocation=allocation, label_sids=label_sids,
            no_cache=no_cache)

    def _prices_to_signals(self, prices, no_cache=False):
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
        cache_key = [self.CODE, prices.index.tolist(), prices.columns.tolist()]
        if self.is_backtest and not no_cache:
            features = Cache.get(cache_key, prefix="_features", unless_file_modified=self)

        if features is None:
            features = self.prices_to_features(prices)
            if self.is_backtest:
                Cache.set(cache_key, features, prefix="_features")

        # validate features
        if not isinstance(features, tuple) or len(features) != 2:
            raise MoonshotError("prices_to_features should return a tuple of (features, targets)")

        features, targets = features

        # Don't use the targets/labels for predictions
        del targets

        if not isinstance(features, (dict, list, tuple, pd.DataFrame)):
            raise MoonshotError("features should either be a DataFrame or a dict, list, or tuple of DataFrames or Series")

        predictions_series_idx = None
        unstack_predictions_series = False

        # a single DataFrame is interpreted as a ready-made DataFrame of features
        if isinstance(features, pd.DataFrame):
            predictions_series_idx = features.index
            features = features.values

        # Convert iteratable of DataFrames or Series to np array
        else:

            if isinstance(features, dict):
                features = features.values()

            all_features = []

            has_df = False
            has_series = False

            for i, feature in enumerate(features):

                if isinstance(feature, pd.DataFrame):
                    has_df = True
                    unstack_predictions_series = True
                    if has_series:
                        raise MoonshotError("features should be either all DataFrames or all Series, not a mix of both")
                    # stack DataFrame to Series
                    feature = feature.stack(dropna=False)
                else:
                    has_series = True
                    if has_df:
                        raise MoonshotError("features should be either all DataFrames or all Series, not a mix of both")

                feature = feature.fillna(0)
                if i == 0:
                    # save stacked index for predictions output
                    predictions_series_idx = feature.index
                all_features.append(feature.values)
                del feature

            features = np.stack(all_features, axis=-1)
            del all_features

        # get predictions
        predictions = self.model.predict(features)
        del features

        if len(predictions.shape) == 2:
            # Keras output has (n_samples,1) shape and needs to be squeezed
            if predictions.shape[-1] == 1:
                predictions = predictions.squeeze(axis=-1)

            # predict_proba has (n_samples,2) shape where first col is probablity of
            # 0 (False) and second col is probability of 1 (True); we just want the
            # second col (https://datascience.stackexchange.com/a/22821)
            elif (
                hasattr(self.model, "classes_")
                and len(self.model.classes_) == 2
                and list(self.model.classes_) == [0,1]):
                predictions = predictions[:,-1]

            else:
                raise NotImplementedError("Don't know what to do with predictions having shape {}".format(predictions.shape))

        predictions = pd.Series(predictions, index=predictions_series_idx)
        if unstack_predictions_series:
            predictions = predictions.unstack(level="Sid")

        # predictions to signals
        signals = self.predictions_to_signals(predictions, prices)
        return signals

    def trade(
        self,
        allocations: dict[str, float],
        review_date: str = None
        ) -> pd.DataFrame:
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

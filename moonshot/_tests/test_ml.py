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

# To run: python3 -m unittest discover -s _tests/ -p test_*.py -t . -v

import os
import unittest
from unittest.mock import patch
import glob
import pickle
import joblib
import platform
import inspect
import pandas as pd
import numpy as np
from moonshot import MoonshotML
from moonshot._cache import TMP_DIR
from moonshot.exceptions import MoonshotError
from sklearn.tree import DecisionTreeClassifier

is_aarch64 = platform.machine() == "aarch64"

class SKLearnMachineLearningTestCase(unittest.TestCase):


    def setUp(self):
        """
        Trains a scikit-learn model.
        """
        self.model = DecisionTreeClassifier()
        # Predict Y will be same as X
        X = np.array([[1,1],[0,0]])
        Y = np.array([1,0])
        self.model.fit(X, Y)
        self.pickle_path = "{0}/decision_tree_model.pkl".format(TMP_DIR)
        self.joblib_path = "{0}/decision_tree_model.joblib".format(TMP_DIR)

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

        for file in (self.pickle_path, self.joblib_path):
            if os.path.exists(file):
                os.remove(file)

    def test_complain_if_mix_dataframe_and_series(self):
        """
        Tests error handling when the features list contains a mix of
        DataFrames and Series.
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML1(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = []
                # DataFrame then Series
                features.append(prices.loc["Close"] > 10)
                features.append(prices.loc["Close"]["FI12345"] > 10)
                return features, None

        class DecisionTreeML2(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = []
                # Series then DataFrame
                features.append(prices.loc["Close"]["FI12345"] > 10)
                features.append(prices.loc["Close"] > 10)
                return features, None

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                with self.assertRaises(MoonshotError) as cm:
                    results = DecisionTreeML1().backtest()

                    self.assertIn(
                        "features should be either all DataFrames or all Series, not a mix of both",
                        repr(cm.exception))

                    # clear cache
                    for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
                        os.remove(file)

                    with self.assertRaises(MoonshotError) as cm:
                        results = DecisionTreeML2().backtest()

                        self.assertIn(
                            "features should be either all DataFrames or all Series, not a mix of both",
                            repr(cm.exception))

    def test_complain_if_no_targets(self):
        """
        Tests error handling when prices_to_features doesn't return a two-tuple.
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = []
                features.append(prices.loc["Close"] > 10)
                features.append(prices.loc["Close"] > 100)
                return features

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                with self.assertRaises(MoonshotError) as cm:
                        results = DecisionTreeML().backtest()

        self.assertIn(
            "prices_to_features should return a tuple of (features, targets)", repr(cm.exception))

    def test_backtest_from_pickle(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy and loading the model from a pickle.
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     0.0]})

    def test_backtest_from_joblib(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy and loading the model from joblib.
        """

        # save model
        joblib.dump(self.model, self.joblib_path)

        class DecisionTreeML(MoonshotML):

            MODEL = self.joblib_path

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     0.0]})

    def test_predict_proba(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy and using predict_proba instead of predict.
        """

        # save model
        joblib.dump(self.model, self.joblib_path)

        class DecisionTreeML(MoonshotML):

            MODEL = self.joblib_path

            def prices_to_features(self, prices):

                self.model.predict = self.model.predict_proba

                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature

                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when <50% probability of being in class 1 (class 1 = price > 10)
                signals = predictions < 0.5
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     0.0]})

    def test_backtest_pass_model(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy and passing the model directly.
        """

        class DecisionTreeML(MoonshotML):

            MODEL = "nosuchpath.pkl" # should be ignored

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest(model=self.model)

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     0.0]})

    def test_backtest_features_list_of_dataframes(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy which produces a list of DataFrames of features.
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = []
                features.append(prices.loc["Close"] > 10)
                features.append(prices.loc["Close"] > 10) # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                             0.0]})

    def test_backtest_features_single_dataframe(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy which produces a single DataFrame of features.
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                feature1 = prices.loc["Close"] > 10
                feature2 = prices.loc["Close"] > 10 # silly, duplicate feature

                feature1 = feature1.stack()
                feature2 = feature2.stack()
                features = pd.concat((feature1, feature2), axis=1)
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions == 0
                signals = signals.unstack(level="Sid").astype(int)
                return signals

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest()

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

        self.maxDiff = None
        signals = results.loc["Signal"].reset_index()
        signals.loc[:, "Date"] = signals.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            signals.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                             0.0]})

    def test_backtest_dict_of_series(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        machine learning strategy which produces a dict of Series of features
        (for predicting a single series).
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML(MoonshotML):

            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"]["FI12345"] > 10
                features["feature2"] = prices.loc["Close"]["FI12345"] > 10 # silly duplicate feature to make model predictable

                targets = prices.loc["Close"]["FI12345"]
                return features, targets

            def predictions_to_signals(self, predictions, prices):
                # Go long on FI12345 when price is predicted to be below 10
                signals = pd.DataFrame(0, index=prices.loc["Close"].index, columns=prices.columns)
                signals.loc[:, "FI12345"] = (predictions == 0).astype(int)
                return signals

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DecisionTreeML().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     1.0,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.0,
                     0.0,
                     0.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     1.0,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.0,
                     0.0,
                     0.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0.0],
             "FI23456": [0,
                     0.0,
                     0,
                     0.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     1.0,
                     1.0,
                     0.0],
             "FI23456": ["nan",
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0454545, # (10.50 - 11)/11 * 1.0
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.0,
                     0.0]})

    def test_trade(self):
        """
        Tests that the resulting orders DataFrame is correct after running a basic
        machine learning strategy.
        """

        # pickle model
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.model, f)

        class DecisionTreeML(MoonshotML):

            CODE = "tree-ml"
            MODEL = self.pickle_path

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

            return prices

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/New_York",
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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                                orders = DecisionTreeML().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'Sid',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'OrderType',
             'Tif'}
        )

        # expected quantity for FI23456:
        # 1.0 weight * 1.0 allocation * 55K / 8.50 = 6471

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'Sid': "FI23456",
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef':'tree-ml',
                    'TotalQuantity': 6471,
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

@unittest.skipIf(is_aarch64, "keras predictions are not as expected on aarch64, they're all the same!")
class KerasMachineLearningTestCase(unittest.TestCase):


    def setUp(self):
        """
        Loads a Keras model fixture.

        Model created with:

        model = Sequential()
        model.add(Dense(1, input_dim=2))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Predict Y will be same as X
        X = np.array([[1,1],[0,0]])
        Y = np.array([1,0])
        model.fit(X, Y)

        model.save("fixtures/test_model.keras.h5")
        """
        thisfile = inspect.getfile(self.__class__)
        thisdir = os.path.dirname(thisfile)
        self.model_path = "{0}/fixtures/test_model.keras.h5".format(thisdir)

    def tearDown(self):
        """
        Remove cached files.
        """
        for file in glob.glob("{0}/moonshot*.pkl".format(TMP_DIR)):
            os.remove(file)

    def test_backtest_from_h5(self):
        """
        Tests that the resulting DataFrames are correct after running a basic
        Keras strategy and loading the model from disk.
        """

        class DeepLearningML(MoonshotML):

            MODEL = self.model_path

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions.round() == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.DatetimeIndex(["2018-05-01","2018-05-02","2018-05-03", "2018-05-04"])
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50,
                        9.99,
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                        10.50,

                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                results = DeepLearningML().backtest()

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
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [1.0,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [1.0,
                     0.0,
                     1.0,
                     0.0]}
        )

        weights = results.loc["Weight"].reset_index()
        weights.loc[:, "Date"] = weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        abs_weights = results.loc["AbsWeight"].reset_index()
        abs_weights.loc[:, "Date"] = abs_weights.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_weights.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.5,
                     0.0,
                     0.0,
                     1.0],
             "FI23456": [0.5,
                     0.0,
                     1.0,
                     0.0]}
        )

        net_positions = results.loc["NetExposure"].reset_index()
        net_positions.loc[:, "Date"] = net_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            net_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        abs_positions = results.loc["AbsExposure"].reset_index()
        abs_positions.loc[:, "Date"] = abs_positions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            abs_positions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.0,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.0,
                     1.0]}
        )

        total_holdings = results.loc["TotalHoldings"].reset_index()
        total_holdings.loc[:, "Date"] = total_holdings.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            total_holdings.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0,
                     1.0,
                     0,
                     0],
             "FI23456": [0,
                     1.0,
                     0,
                     1.0]}
        )

        turnover = results.loc["Turnover"].reset_index()
        turnover.loc[:, "Date"] = turnover.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            turnover.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": ["nan",
                     0.5,
                     0.5,
                     0.0],
             "FI23456": ["nan",
                     0.5,
                     0.5,
                     1.0]}
        )

        commissions = results.loc["Commission"].reset_index()
        commissions.loc[:, "Date"] = commissions.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            commissions.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        slippage = results.loc["Slippage"].reset_index()
        slippage.loc[:, "Date"] = slippage.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            slippage.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     0.0,
                     0.0],
             "FI23456": [0.0,
                     0.0,
                     0.0,
                     0.0]}
        )

        returns = results.loc["Return"]
        returns = returns.reset_index()
        returns.loc[:, "Date"] = returns.Date.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.assertDictEqual(
            returns.to_dict(orient="list"),
            {'Date': [
                '2018-05-01T00:00:00',
                '2018-05-02T00:00:00',
                '2018-05-03T00:00:00',
                '2018-05-04T00:00:00'],
             "FI12345": [0.0,
                     0.0,
                     -0.0227273, # (10.50 - 11)/11 * 0.5
                     -0.0],
             "FI23456": [0.0,
                     0.0,
                     -0.1136364, # (8.50 - 11)/11 * 0.5
                     0.0]})

    def test_trade(self):
        """
        Tests that the resulting orders DataFrame is correct after running a basic
        Keras machine learning strategy.
        """
        class DeepLearningML(MoonshotML):

            CODE = "deep-ml"
            MODEL = self.model_path

            def prices_to_features(self, prices):
                features = {}
                features["feature1"] = prices.loc["Close"] > 10
                features["feature2"] = prices.loc["Close"] > 10 # silly, duplicate feature
                return features, None

            def predictions_to_signals(self, predictions, prices):
                # Go long when price is predicted to be below 10
                signals = predictions.round() == 0
                return signals.astype(int)

        def mock_get_prices(*args, **kwargs):

            dt_idx = pd.date_range(end=pd.Timestamp.today(tz="America/New_York"), periods=3, normalize=True)
            fields = ["Close"]
            idx = pd.MultiIndex.from_product([fields, dt_idx], names=["Field", "Date"])

            prices = pd.DataFrame(
                {
                    "FI12345": [
                        # Close
                        9,
                        11,
                        10.50
                    ],
                    "FI23456": [
                        # Close
                        9.89,
                        11,
                        8.50,
                    ],
                 },
                index=idx
            )
            prices.columns.name = "Sid"

            return prices

        def mock_download_master_file(f, *args, **kwargs):

            master_fields = ["Timezone", "SecType", "Currency", "PriceMagnifier", "Multiplier"]
            securities = pd.DataFrame(
                {
                    "FI12345": [
                        "America/New_York",
                        "STK",
                        "USD",
                        None,
                        None
                    ],
                    "FI23456": [
                        "America/New_York",
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

        with patch("moonshot.strategies.base.get_prices", new=mock_get_prices):
            with patch("moonshot.strategies.base.download_account_balances", new=mock_download_account_balances):
                with patch("moonshot.strategies.base.download_exchange_rates", new=mock_download_exchange_rates):
                    with patch("moonshot.strategies.base.list_positions", new=mock_list_positions):
                        with patch("moonshot.strategies.base.download_order_statuses", new=mock_download_order_statuses):
                            with patch("moonshot.strategies.base.download_master_file", new=mock_download_master_file):
                                                orders = DeepLearningML().trade({"U123": 1.0})

        self.assertSetEqual(
            set(orders.columns),
            {'Sid',
             'Account',
             'Action',
             'OrderRef',
             'TotalQuantity',
             'OrderType',
             'Tif'}
        )

        # expected quantity for FI23456:
        # 1.0 weight * 1.0 allocation * 55K / 8.50 = 6471

        self.assertListEqual(
            orders.to_dict(orient="records"),
            [
                {
                    'Sid': "FI23456",
                    'Account': 'U123',
                    'Action': 'BUY',
                    'OrderRef':'deep-ml',
                    'TotalQuantity': 6471,
                    'OrderType': 'MKT',
                    'Tif': 'DAY'
                }
            ]
        )

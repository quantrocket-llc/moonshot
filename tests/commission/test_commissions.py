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

# To run: python3 -m unittest discover -s tests/ -p test_*.py -t . -v

import unittest
import pandas as pd
from moonshot.commission import (
    FuturesCommission, PerShareCommission, NoCommission)
from moonshot.commission.fx import SpotForexCommission

class TestFuturesCommission(FuturesCommission):
    IB_COMMISSION_PER_CONTRACT = 0.85
    EXCHANGE_FEE_PER_CONTRACT = 1.20
    CARRYING_FEE_PER_CONTRACT = 0

class FuturesCommissionTestCase(unittest.TestCase):

    def test_commissions(self):

        turnover = pd.DataFrame(
            {"ES201609": [0.1, 0.2],
            "NQ201609": [0.18, 0.32]},
            )
        contract_values = pd.DataFrame(
            {"ES201609": [70000, 70000],
            "NQ201609": [105000, 105000]},
            )
        commissions = TestFuturesCommission.get_commissions(
            contract_values,
            turnover)

        # expected commission
        # ES 0: $2.05 / $70000 * .1  = 0.000002929
        # ES 1: $2.05 / $70000 * .2  = 0.000005857
        # NQ 0: $2.05 / $105000 * .18  = 0.000003514
        # NQ 1: $2.05 / $105000 * .32  = 0.000006248
        self.assertEqual(round(commissions.loc[0, "ES201609"], 9), 0.000002929)
        self.assertEqual(round(commissions.loc[1, "ES201609"], 9), 0.000005857)
        self.assertEqual(round(commissions.loc[0, "NQ201609"], 9), 0.000003514)
        self.assertEqual(round(commissions.loc[1, "NQ201609"], 9), 0.000006248)

class TestStockCommission(PerShareCommission):
    IB_COMMISSION_PER_SHARE = 0.0035 # IB commission per share
    EXCHANGE_FEE_PER_SHARE = 0.0003
    MAKER_FEE_PER_SHARE = -0.002 # exchange rebate
    TAKER_FEE_PER_SHARE = 0.00118 # exchange fee
    MAKER_RATIO = 0.4
    MIN_COMMISSION = 0.35
    COMMISSION_PERCENTAGE_FEE_RATE = 0.056
    PERCENTAGE_FEE_RATE = 0.00002

class PerShareCommissionTestCase(unittest.TestCase):

    def test_min_commission(self):

        # only buy 50 shares
        turnover_pct = 50*250/220000
        turnover = pd.DataFrame(
            {"LVS": [turnover_pct]},
        )
        contract_values = pd.DataFrame(
            {"LVS": [250.00]},
        )
        nlvs = pd.DataFrame(
            {"LVS": [220000]},
        )
        commission = TestStockCommission.get_commissions(
            contract_values,
            turnover,
            nlvs=nlvs)

        # expected commission
        # exchange fees = 0.0003 + (0.4 * -0.002) + (0.6 * 0.00118) = 0.000208 * 50 shares = 0.0104
        # percentage fees = 0.00002 * 50 * 250 = 0.25
        # commission based fees = 0.00056 * 0.35 = 0.0196
        # 0.35 IB min commission + 0.0104 + 0.25 + 0.0196 = 0.63 / 220000 = 0.000002864
        self.assertEqual(round(commission.loc[0, "LVS"], 9), 0.000002864)

    def test_maker_commissions(self):

        class TestMakerCommission(TestStockCommission):
            MAKER_RATIO = 1

        turnover = pd.DataFrame(
            {"AAPL": [0.1]},
        )
        contract_values = pd.DataFrame(
            {"AAPL": [90]},
        )
        nlvs = pd.DataFrame(
            {"AAPL": [500000]},
        )

        commissions = TestMakerCommission.get_commissions(
            contract_values,
            turnover,
            nlvs=nlvs)

        # expected commission
        # (0.0035 - 0.002 + 0.0003 + (0.0035 * 0.056)) * 500000 * 0.1 / 90 = 1.108888889 + (500000 * 0.1 * 0.00002) = $2.108888 / $500000 = 0.000004218
        self.assertEqual(round(commissions.loc[0, "AAPL"], 9), 0.000004218)

    def test_taker_commissions(self):

        class TestTakerCommission(TestStockCommission):
            MAKER_RATIO = 0

        turnover = pd.DataFrame(
            {"AAPL": [0.1]},
        )
        contract_values = pd.DataFrame(
            {"AAPL": [90]},
        )
        nlvs = pd.DataFrame(
            {"AAPL": [500000]},
        )

        commissions = TestTakerCommission.get_commissions(
            contract_values,
            turnover,
            nlvs=nlvs)

        # expected commission
        # (0.0035 + 0.00118 + 0.0003 + (0.0035 * 0.056)) * 500000 * 0.1 / 90 = 2.87555 + (500000 * 0.1 * 0.00002) = $3.8755 / $500000 = 0.000007751
        self.assertEqual(round(commissions.loc[0, "AAPL"], 9), 0.000007751)

    def test_maker_taker_commissions(self):

        class TestMakerTakerCommission(TestStockCommission):
            MAKER_RATIO = 0.60

        turnover = pd.DataFrame(
            {"AAPL": [0.1]},
        )
        contract_values = pd.DataFrame(
            {"AAPL": [90]},
        )
        nlvs = pd.DataFrame(
            {"AAPL": [500000]},
        )

        commissions = TestMakerTakerCommission.get_commissions(
            contract_values,
            turnover,
            nlvs=nlvs)

        # expected commission
        # (0.0035 + (0.00118*0.4) + (-0.002 * 0.6) + 0.0003 + (0.0035 * 0.056)) * 500000 * 0.1 / 90 = $1.8155 + (500000 * 0.1 * 0.00002) = $2.8155 / $500000 = 0.000005631
        self.assertEqual(round(commissions.loc[0, "AAPL"], 9), 0.000005631)

class NoCommissionTestCase(unittest.TestCase):

    def test_no_commissions(self):
        turnover = pd.DataFrame(
            {"WYNN": [0.15]},
        )
        contract_values = pd.DataFrame(
            {"WYNN": [79.56]},
        )
        commissions = NoCommission.get_commissions(
            contract_values,
            turnover)

        self.assertEqual(commissions.loc[0, "WYNN"], 0)

class ForexCommissionTestCase(unittest.TestCase):

    def test_commissions_cadhkd(self):

        turnover = pd.DataFrame(
            {"CAD.HKD": [0.25]})
        contract_values = None # not used for fixed rate
        nlvs = pd.DataFrame(
            {"CAD.HKD": [700000]})
        commissions = SpotForexCommission.get_commissions(
            contract_values,
            turnover,
            nlvs=nlvs)

        # expected commission
        # .2 bps x 0.25 x 700K USD = $3.50 / 700K USD = 0.000005
        commissions = commissions["CAD.HKD"]
        self.assertEqual(commissions.iloc[0], 0.000005)

    # Spot fx min commissions aren't currently supported
    @unittest.expectedFailure
    def test_min_commissions_cadhkd(self):

        turnover = pd.DataFrame(
            {"CAD.HKD": [0.01, 0.05]})
        contract_values = None # not used for fixed rate
        nlvs = pd.DataFrame(
            {"CAD.HKD": [700000,700000]})
        commissions = SpotForexCommission.get_commissions(
            contract_values,
            turnover,
            nlvs=nlvs)

        # expected commission
        # 0: .2 bps x 0.01 x 700K USD = $0.14 = $2 min / 700K USD = 0.000002857
        # 1: .2 bps x 0.05 x 700K USD = $0.70 = $2 min / 700K USD = 0.000002857
        commissions = commissions["CAD.HKD"]
        self.assertEqual(round(commissions.iloc[0], 9), 0.000002857)
        self.assertEqual(round(commissions.iloc[1], 9), 0.000002857)

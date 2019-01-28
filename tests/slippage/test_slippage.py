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
from moonshot.slippage import FixedSlippage, BorrowFees
from moonshot import Moonshot

class TestFixedSlippage(FixedSlippage):
    ONE_WAY_SLIPPAGE = 0.0010

class FixedSlippageTestCase(unittest.TestCase):

    def test_fixed_slippage(self):

        turnover = pd.DataFrame(
            {"ES201609": [0.1, 0.2],
            "NQ201609": [0.17, 0.32]},
            )
        slippage = TestFixedSlippage().get_slippage(turnover)

        self.assertListEqual(
            slippage.to_dict(orient="records"),
            [{'ES201609': 0.0001, 'NQ201609': 0.00017},
             {'ES201609': 0.0002, 'NQ201609': 0.00032}]
        )

class BorrowFeesSlippageTestCase(unittest.TestCase):

    @patch("moonshot.slippage.borrowfee.get_borrow_fees_reindexed_like")
    def test_borrow_fees_slippage(self, mock_get_borrow_fees_reindexed_like):

        positions = pd.DataFrame(
            {12345: [0.1, 0, -0.2, -0.2, -0.1, 0.5, -0.25],
            23456: [-0.17, 0.32, 0.23, 0, -0.4, -0.4, -0.4]},
            index=pd.DatetimeIndex(["2018-06-01", "2018-06-02", "2018-06-03",
                                   "2018-06-04", "2018-06-05", "2018-06-08",
                                   "2018-06-09"]))

        borrow_fee_rates = pd.DataFrame(
            {12345: [1.75, 1.75, 1.75, 1.85, 1.85, 1.85, 1.2],
            23456: [8.0, 8.0, 8.23, 8.5, 0, 0, None]},
            index=pd.DatetimeIndex(["2018-06-01", "2018-06-02", "2018-06-03",
                                   "2018-06-04", "2018-06-05", "2018-06-08",
                                   "2018-06-09"]))

        mock_get_borrow_fees_reindexed_like.return_value = borrow_fee_rates

        turnover = prices = None
        fees = BorrowFees().get_slippage(turnover, positions, prices)

        mock_get_borrow_fees_reindexed_like.assert_called_with(positions, time=None)

        fees.index.name = "Date"
        fees.index = fees.index.strftime("%Y-%m-%d")
        fees = fees.to_dict(orient="dict")

        self.assertAlmostEqual(fees[12345]["2018-06-01"], 0)
        self.assertAlmostEqual(fees[12345]["2018-06-02"], 0)
        self.assertAlmostEqual(fees[12345]["2018-06-03"], 0.000013889, 9)
        self.assertAlmostEqual(fees[12345]["2018-06-04"], 0.000014683, 9)
        self.assertAlmostEqual(fees[12345]["2018-06-05"], 0.000007341, 9)
        self.assertAlmostEqual(fees[12345]["2018-06-08"], 0)
        self.assertAlmostEqual(fees[12345]["2018-06-09"], 0.000011905, 9)

        self.assertAlmostEqual(fees[23456]["2018-06-01"], 0.000053968, 9)
        self.assertAlmostEqual(fees[23456]["2018-06-02"], 0)
        self.assertAlmostEqual(fees[23456]["2018-06-03"], 0)
        self.assertAlmostEqual(fees[23456]["2018-06-04"], 0)
        self.assertAlmostEqual(fees[23456]["2018-06-05"], 0)
        self.assertAlmostEqual(fees[23456]["2018-06-08"], 0)
        self.assertAlmostEqual(fees[23456]["2018-06-09"], 0)

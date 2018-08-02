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

import os
import unittest
from unittest.mock import patch
import glob
import pandas as pd
from moonshot import Moonshot
from moonshot.cache import TMP_DIR

class WeightAllocationsTestCase(unittest.TestCase):

    def test_allocate_equal_weights(self):
        """
        Tests that the allocate_equal_weights returns the expected
        DataFrames.
        """
        signals = pd.DataFrame(
            data={
                12345: [1, 1, 1, 0, 0],
                23456: [0, -1, 1, 0, -1],
            }
        )

        target_weights = Moonshot().allocate_equal_weights(signals, cap=1.0)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [1.0, 0.5, 0.5, 0.0, 0.0],
             23456: [0.0, -0.5, 0.5, 0.0, -1.0]}
        )

        target_weights = Moonshot().allocate_equal_weights(signals, cap=0.5)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [0.5, 0.25, 0.25, 0.0, 0.0],
             23456: [0.0, -0.25, 0.25, 0.0, -0.5]}
        )

    def test_allocate_fixed_weights(self):
        """
        Tests that the allocate_fixed_weights returns the expected
        DataFrames.
        """
        signals = pd.DataFrame(
            data={
                12345: [1, 1, 1, 0, 0],
                23456: [0, -1, 1, 0, -1],
                34567: [1, 1, 1, -1, -1]
            }
        )

        target_weights = Moonshot().allocate_fixed_weights(signals, 0.34)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [0.34, 0.34, 0.34, 0.0, 0.0],
             23456: [0.0, -0.34, 0.34, 0.0, -0.34],
             34567: [0.34, 0.34, 0.34, -0.34, -0.34]}
        )

    def test_allocate_fixed_weights_capped(self):
        """
        Tests that the allocate_fixed_weights_capped returns the expected
        DataFrames.
        """
        signals = pd.DataFrame(
            data={
                12345: [1, 1, 1, 0, 0],
                23456: [0, -1, 1, 0, -1],
                34567: [1, 1, 1, -1, -1]
            }
        )

        target_weights = Moonshot().allocate_fixed_weights_capped(signals, 0.34, cap=1.5)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [0.34, 0.34, 0.34, 0.0, 0.0],
             23456: [0.0, -0.34, 0.34, 0.0, -0.34],
             34567: [0.34, 0.34, 0.34, -0.34, -0.34]}
        )

        target_weights = Moonshot().allocate_fixed_weights_capped(signals, 0.34, cap=0.81)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [0.34, 0.27, 0.27, 0.0, 0.0],
             23456: [0.0, -0.27, 0.27, 0.0, -0.34],
             34567: [0.34, 0.27, 0.27, -0.34, -0.34]}
        )

    def test_allocate_market_neutral_fixed_weights_capped(self):
        """
        Tests that the allocate_market_neutral_fixed_weights_capped returns
        the expected DataFrames.
        """
        signals = pd.DataFrame(
            data={
                12345: [1, 1, 1, 0, 0],
                23456: [0, -1, 1, 1, -1],
                34567: [1, 1, -1, -1, -1]
            }
        )

        target_weights = Moonshot().allocate_market_neutral_fixed_weights_capped(
            signals, 0.34, cap=1.2, neutralize_weights=False)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [0.3, 0.3, 0.3, 0.0, 0.0],
             23456: [0.0, -0.34, 0.3, 0.34, -0.3],
             34567: [0.3, 0.3, -0.34, -0.34, -0.3]}
        )

        target_weights = Moonshot().allocate_market_neutral_fixed_weights_capped(
            signals, 0.34, cap=1.2, neutralize_weights=True)

        self.assertDictEqual(
            target_weights.to_dict(orient="list"),
            {12345: [0.0, 0.17, 0.17, 0.0, 0.0],
             23456: [0.0, -0.34, 0.17, 0.34, -0.0],
             34567: [0.0, 0.17, -0.34, -0.34, -0.0]}
        )

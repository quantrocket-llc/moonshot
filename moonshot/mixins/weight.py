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

import pandas as pd
import numpy as np

class WeightAllocationMixin(object):
    """
    Mixin class with utilities for turning signals into weights.
    """
    def allocate_equal_weights(self, signals, cap=1.0):
        """
        For multi-security strategies. Given a dataframe of whole number
        signals (-1, 0, 1), reduces the position size so that the absolute
        sum of all weights is never greater than the cap.
        """
        # Count active signals for the day
        signals_count = signals.abs().sum(axis=1)
        # If no signals, divide by 1 to leave the signal as-is (can't divide by 0)
        divisor = np.where(signals_count != 0, signals_count, 1)
        return signals.div(divisor, axis=0) * cap / 1.0

    def allocate_fixed_weights(self, signals, weight):
        """
        Applies the specified fixed weight to the signals.
        """
        return signals * weight

    def allocate_fixed_weights_capped(self, signals, weight, cap=1.0):
        """
        Applies fixed weights, but if the sum of the weights exceeds the cap,
        applies equal weights.
        """
        equal_weighted = self.allocate_equal_weights(signals, cap=cap)
        fixed_weighted = self.allocate_fixed_weights(signals, weight)
        fixed_sum = fixed_weighted.abs().sum(axis=1)
        fixed_sum = pd.DataFrame(dict(
            [(column, fixed_sum.copy()) for column in signals.columns]),
            columns=signals.columns, index=signals.index)
        return pd.DataFrame(
            np.where(fixed_sum > cap, equal_weighted, fixed_weighted),
            index=signals.index, columns=signals.columns)

    def allocate_market_neutral_fixed_weights_capped(self, signals, weight, cap=1.0,
                                                  neutralize_weights=True):
        """
        Applies fixed capped weights to the long and short side separately to
        ensure the strategy is hedged.
        """
        long_signals = signals.where(signals > 0, 0)
        short_signals = signals.where(signals < 0, 0)
        cap_per_side = cap * 0.5
        long_weights = self.allocate_fixed_weights_capped(long_signals, weight, cap=cap_per_side)
        short_weights = self.allocate_fixed_weights_capped(short_signals, weight, cap=cap_per_side)
        weights = long_weights.where(long_weights > 0, short_weights)
        if neutralize_weights:
            weights = self.neutralize_weights(weights)
        return weights

    def neutralize_weights(self, weights):
        """
        If the long or short side has a greater total weight than the
        opposite side, proportionately reduces the overweight side.
        """
        long_weights = weights.where(weights > 0, 0)
        short_weights = weights.where(weights < 0, 0)

        total_long_weights = long_weights.sum(axis=1)
        total_long_weights = pd.DataFrame(dict((column, total_long_weights.copy()) for column in weights.columns),
                                          index=weights.index, columns=weights.columns)
        total_short_weights = short_weights.abs().sum(axis=1)
        total_short_weights = pd.DataFrame(dict((column, total_short_weights.copy()) for column in weights.columns),
                                           index=weights.index, columns=weights.columns)

        long_weights = long_weights.where(
            total_long_weights <= total_short_weights,
            long_weights * total_short_weights / total_long_weights.replace(0, 1))

        short_weights = short_weights.where(
            total_short_weights <= total_long_weights,
            short_weights * total_long_weights / total_short_weights.replace(0, 1))

        weights = long_weights.where(long_weights > 0, short_weights)
        return weights

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

from .base import Slippage
import pandas as pd
from quantrocket.fundamental import get_ibkr_borrow_fees_reindexed_like

class IBKRBorrowFees(Slippage):
    """
    Apply borrow fees to each short position.

    Examples
    --------
    Use this on your strategy:

    >>>  class MyStrategy(Moonshot):
    >>>      SLIPPAGE_CLASSES = IBKRBorrowFees
    """

    def get_slippage(
        self,
        turnover: pd.DataFrame,
        positions: pd.DataFrame,
        prices: pd.DataFrame
        ) -> pd.DataFrame:

        borrow_fees = get_ibkr_borrow_fees_reindexed_like(positions)

        # convert to decimals
        borrow_fees = borrow_fees / 100
        # convert to daily rates
        daily_borrow_fees = borrow_fees / 360 # industry convention is to divide annual fee by 360, not 365

        # account for weekends, which are assessed the borrow fee x 3 days
        dates = borrow_fees.apply(lambda x: borrow_fees.index)
        days_held = (dates - dates.shift()).fillna(pd.Timedelta('1d')).apply(lambda x: x.dt.days)
        daily_borrow_fees *= days_held

        # by industry convention, collateral amount is 102% of borrow amount
        assessed_fees = positions.where(positions < 0, 0).abs() * 1.02 * daily_borrow_fees

        return assessed_fees

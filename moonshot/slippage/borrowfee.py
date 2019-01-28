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

from quantrocket.fundamental import get_borrow_fees_reindexed_like

class BorrowFees(object):
    """
    Applies borrow fees to each position.

    Parameters
    ----------
    TIME : str (HH:MM:SS[ TZ]), optional
         Query and assess borrow fees as of this time of day. See
         `quantrocket.fundamental.get_borrow_fees_reindexed_like`
         for more details.
    """

    TIME = None

    def get_slippage(self, turnover, positions, prices):

        borrow_fees = get_borrow_fees_reindexed_like(positions, time=self.TIME)
        borrow_fees = borrow_fees.fillna(0) / 100
        # Fees are assessed daily but the dataframe is expected to only
        # includes trading days, thus use 252 instead of 365. In reality the
        # borrow fee is greater for weekend positions than weekday positions,
        # but this implementation doesn't model that.
        daily_borrow_fees = borrow_fees / 252
        assessed_fees = positions.where(positions < 0, 0).abs() * daily_borrow_fees
        return assessed_fees

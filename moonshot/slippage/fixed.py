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

class FixedSlippage(object):
    """
    Applies a fixed pct slippage to each trade.

    This slippage class can be used on strategies indirectly (and more
    easily) by simply specifying SLIPPAGE_BPS on the strategy.

    Parameters
    ----------
    ONE_WAY_SLIPPAGE : float
        the slippage to apply to each trade (default 0.0005 = 5 basis points);
        overridden if `one_way_slippage` is passed to __init__
    """
    ONE_WAY_SLIPPAGE = 0.0005

    def __init__(self, one_way_slippage=None):
        if one_way_slippage is not None:
            self.one_way_slippage = one_way_slippage
        else:
            self.one_way_slippage = self.ONE_WAY_SLIPPAGE

    def get_slippage(self, turnover, *args, **kwargs):
        """
        Apply the fix pct slippage to each trade.

        Parameters
        ----------
        turnover : DataFrame, required
            a DataFrame of turnover

        Returns
        -------
        DataFrame
            slippages
        """
        return turnover * self.one_way_slippage

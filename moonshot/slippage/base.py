# Copyright 2017-2024 QuantRocket LLC - All Rights Reserved
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

class Slippage:
    """
    Base class for slippage classes.

    A slippage class must implement a method called `get_slippage` that
    accepts a DataFrame of turnover, a DataFrame of positions, and a
    DataFrame of prices, and returns a DataFrame of slippage.
    """
    def get_slippage(
        self,
        turnover: pd.DataFrame,
        positions: pd.DataFrame,
        prices: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Apply slippage to each trade.

        Parameters
        ----------
        turnover : DataFrame, required
            a DataFrame of turnover

        positions : DataFrame, required
            a DataFrame of positions

        prices : DataFrame, required
            a DataFrame of prices

        Returns
        -------
        DataFrame
            a DataFrame of slippages
        """
        raise NotImplementedError()

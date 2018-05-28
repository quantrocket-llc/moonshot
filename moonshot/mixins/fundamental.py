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

import warnings
from quantrocket.fundamental import get_reuters_financials_reindexed_like

class ReutersFundamentalsMixin(object):
    """
    Moonshot mixin class providing utility functions for working with Reuters
    fundamentals.

    NOTE: This class is deprecated, please use
    `quantrocket.fundamental.get_reuters_financials_reindexed_like` instead.
    """

    def get_reuters_financials(self, coa_codes, reindex_like, fields=["Amount"], interim=False):
        """
        Return a multiindex (CoaCode, Field, Date) DataFrame of point-in-time
        Reuters financial statements for one or more Chart of Account (COA)
        codes. The resulting DataFrame is reindexed to match the index
        (dates) and columns (conids) of `reindex_like`. Financial values
        are forward-filled in order to provide the latest reading at any
        given date. Financials are indexed to the SourceDate field, i.e. the
        date on which the financial statement was released.

        Parameters
        ----------
        coa_codes : list of str, required
            the Chart of Account (COA) code(s) to query

        reindex_like : DataFrame, required
            a DataFrame (usually of prices) with dates for the index and conids
            for the columns, to which the shape of the resulting DataFrame will
            be conformed

        fields : list of str
            a list of fields to include in the resulting DataFrame. Defaults to
            simply including the Amount field.

        interim : bool
            query interim/quarterly reports (default is to query annual reports,
            which provide deeper history)

        Returns
        -------
        DataFrame
            a multiindex (CoaCode, Field, Date) DataFrame of financials,
            shaped like the input DataFrame

        Examples
        --------
        Let's calculate book value per share, defined as:

            (Total Assets - Total Liabilities) / Number of shares outstanding

        The COA codes for these metrics are 'ATOT' (Total Assets), 'LTLL' (Total
        Liabilities), and 'QTCO' (Total Common Shares Outstanding).


        >>> closes = prices.loc["Close"]
        >>> financials = self.get_reuters_financials(["ATOT", "LTLL", "QTCO"], closes)
        >>> tot_assets = financials.loc["ATOT"].loc["Amount"]
        >>> tot_liabilities = financials.loc["LTLL"].loc["Amount"]
        >>> shares_out = financials.loc["QTCO"].loc["Amount"]
        >>> book_values_per_share = (tot_assets - tot_liabilities)/shares_out

        """
        warnings.warn(
            "this method has been deprecated and will be removed in a "
            "future release, please use quantrocket.fundamental.get_reuters_financials_reindexed_like "
            "instead", DeprecationWarning)

        return get_reuters_financials_reindexed_like(
            reindex_like, coa_codes, fields=fields, interim=interim)

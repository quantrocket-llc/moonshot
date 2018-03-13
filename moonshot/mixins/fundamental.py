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

import io
import pandas as pd
from quantrocket.fundamental import download_reuters_financials

class ReutersFundamentalsMixin(object):
    """
    Moonshot mixin class providing utility functions for working with Reuters
    fundamentals.
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
        conids = list(reindex_like.columns)
        start_date = reindex_like.index.min().date().isoformat()
        end_date = reindex_like.index.max().date().isoformat()

        f = io.StringIO()
        download_reuters_financials(
            coa_codes, f, conids=conids, start_date=start_date, end_date=end_date,
            fields=fields, interim=interim)
        financials = pd.read_csv(
            f, parse_dates=["SourceDate","FiscalPeriodEndDate"])

        if self.TIMEZONE:
            source_dates = financials.SourceDate.dt.tz_localize("UTC")
            period_end_dates = financials.FiscalPeriodEndDate.dt.tz_localize("UTC")
            source_dates = source_dates.dt.tz_convert(self.TIMEZONE)
            period_end_dates = period_end_dates.dt.tz_convert(self.TIMEZONE)
            source_dates = source_dates.dt.tz_localize(None)
            period_end_dates = period_end_dates.dt.tz_localize(None)

        financials.loc[:, "SourceDate"] = source_dates
        financials.loc[:, "FiscalPeriodEndDate"] = period_end_dates

        # Rename SourceDate to match price history index name
        financials = financials.rename(columns={"SourceDate": "Date"})

        # Drop any fields we don't need
        needed_fields = set(fields)
        needed_fields.update(set(("ConId", "Date", "CoaCode")))
        unneeded_fields = set(financials.columns) - needed_fields
        if unneeded_fields:
            financials = financials.drop(unneeded_fields, axis=1)

        # Create a unioned index of input DataFrame and statement SourceDates
        source_dates = pd.to_datetime(financials.Date.unique())
        union_date_idx = reindex_like.index.union(source_dates).sort_values()

        all_financials = {}
        for code in coa_codes:
            financials_for_code = financials.loc[financials.CoaCode == code]
            if "CoaCode" not in fields:
                financials_for_code = financials_for_code.drop("CoaCode", axis=1)
            financials_for_code = financials_for_code.pivot(index="ConId",columns="Date").T
            multiidx = pd.MultiIndex.from_product(
                (financials_for_code.index.get_level_values(0).unique(), union_date_idx),
                names=["Field", "Date"])
            financials_for_code = financials_for_code.reindex(index=multiidx, columns=reindex_like.columns)

            # financial values are sparse so ffill
            financials_for_code = financials_for_code.fillna(method="ffill")

            # In cases the statements included dates not in the input
            # DataFrame, drop those now that we've ffilled
            extra_dates = union_date_idx.difference(reindex_like.index)
            if not extra_dates.empty:
                financials_for_code.drop(extra_dates, axis=0, level="Date", inplace=True)

            all_financials[code] = financials_for_code

        financials = pd.concat(all_financials, names=["CoaCode", "Field", "Date"])

        return financials

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

class BaseCommission(object):
    """
    Base class for all commission classes.
    """
    MIN_COMMISSION = 0

    @classmethod
    def get_commissions(cls, contract_values, turnover, nlvs=None):
        """
        Returns a DataFrame of commissions.


        Parameters
        ----------
        contract_values : DataFrame, required
            a DataFrame of contract values (price * multipler / price_magnifier)

        turnover : DataFrame of floats, required
            a DataFrame of turnover, expressing the percentage of account equity that
            is turning over

        nlvs : DataFrame of nlvs, optional
            a DataFrame of NLVs (account balance), which is used to calculate and
            enforce min commissions. NLVs should be expressed in the currency of the
            contract, which should also be the currency of the commission class. If
            not provided, min commissions won't be calculated or enforced.

        Returns
        -------
        DataFrame
            a DataFrame of commissions, expressed as percentages of account equity
        """
        raise NotImplementedError()

    @classmethod
    def _enforce_min_commissions(cls, commissions, nlvs):
        """
        Return a DataFrame of commissions after enforcing the min commission.
        """
        # Express the min commission as a percentage of NLV
        min_commissions = cls.MIN_COMMISSION / nlvs
        must_pay_min_commissions = (commissions > 0) & (commissions < min_commissions)
        commissions = commissions.where(must_pay_min_commissions == False, min_commissions)
        return commissions

class PercentageCommission(BaseCommission):
    """
    Base class for commissions which are a fixed percentage of the trade
    value. These commissions consist of an IB commission percentage rate
    (which might vary based on monthly trade volume) plus a fixed exchange
    fee percentage rate.

    This class can't be used directly but should be subclassed with the
    appropriate parameters.

    Parameters
    ----------
    IB_COMMISSION_RATE : float, required
        the commission rate (as a percentage of trade value) charged by IB

    IB_COMMISSION_RATE_TIER_2 : float, optional
        the commission rate (as a percentage of trade value) charged by IB
        at monthly volume tier 2

    TIER_2_RATIO : float, optional
        the ratio of monthly trades at volume tier 2 (default 0)

    EXCHANGE_FEE_RATE : float, required
        the exchange fee as a percentage of trade value

    MIN_COMMISSION : float, optional
        the minimum commission charged by IB. Only enforced if NLVs are passed
        by the backtest.

    Examples
    --------
    Example commission subclass for Tokyo Stock Exchange:

    >>> class JapanStockCommission(PercentageCommission):
    >>>     IB_COMMISSION_RATE = 0.0005
    >>>     EXCHANGE_FEE_RATE = 0.000004
    >>>     MIN_COMMISSION = 80.00 # JPY
    >>>
    >>>  # then, use this on your strategy:
    >>>  class MyJapanStrategy(Moonshot):
    >>>      COMMISSION_CLASS = JapanStockCommission
    """
    IB_COMMISSION_RATE = 0
    IB_COMMISSION_RATE_TIER_2 = None
    TIER_2_RATIO = None
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 0

    @classmethod
    def get_commissions(cls, contract_values, turnover, nlvs=None):
        """
        Returns a DataFrame of commissions.


        Parameters
        ----------
        contract_values : DataFrame, required
            a DataFrame of contract values (price * multipler / price_magnifier)

        turnover : DataFrame of floats, required
            a DataFrame of turnover, expressing the percentage of account equity that
            is turning over

        nlvs : DataFrame of nlvs, optional
            a DataFrame of NLVs (account balance), which is used to calculate and
            enforce min commissions. NLVs should be expressed in the currency of the
            contract, which should also be the currency of the commission class. If
            not provided, min commissions won't be calculated or enforced.

        Returns
        -------
        DataFrame
            a DataFrame of commissions, expressed as percentages of account equity
        """
        if cls.TIER_2_RATIO:
            ib_commission_rate = (
                ((1 - cls.TIER_2_RATIO) * cls.IB_COMMISSION_RATE)
                + (cls.TIER_2_RATIO * cls.IB_COMMISSION_RATE_TIER_2)
            )
        else:
            ib_commission_rate = cls.IB_COMMISSION_RATE

        ib_commissions = turnover * ib_commission_rate

        if nlvs is not None and cls.MIN_COMMISSION:
            ib_commissions = cls._enforce_min_commissions(ib_commissions, nlvs=nlvs)

        exchange_commissions = turnover * cls.EXCHANGE_FEE_RATE

        commissions = ib_commissions + exchange_commissions

        return commissions

class NoCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 0

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

from moonshot.commission.base import BaseCommission, PercentageCommission

class BaseCostPlusStockCommission(BaseCommission):
    """
    Base class for IB Cost-Plus stock commissions.

    IB Cost-Plus commissions/fees consist of a transaction fee, which is a
    percentage of the trade value, and various per share costs.

    This class can't be used directly but should be subclassed with the
    appropriate parameters.

    Parameters
    ----------
    IB_COMMISSION_PER_SHARE : float, required
        the IB commission per share at the lowest volume tier

    IB_COMMISSION_PER_SHARE_TIER_2 : float, optional
        the IB commission per share at volume tier 2

    TIER_2_RATIO : float, optional
        ratio of monthly trades at volume tier 2

    MAKER_FEE_PER_SHARE : float, optional
        the "maker" fee from the exchange for adding liquidity. Use a negative value
        to indicate a rebate

    TAKER_FEE_PER_SHARE : float, optional
        the "taker" fee paid to the exchange for removing liquidity

    MAKER_RATIO : float, optional
        the ratio of trades that earn the maker fee (for example if 75% of trades add
        liquidty and 25% remove liqudity, this value should be 0.75)

    MIN_COMMISSION : float, optional
        the minimum commission charged by IB. Only enforced if NLVs are passed
        by the backtest.

    TRANSACTION_FEE_RATE : float, optional
        the transaction fee rate as a percentage of trade value

    Examples
    --------
    Example subclass for US stock comissions:

    >>> class CostPlusUSStockCommission(BaseCostPlusStockCommission):
    >>>     IB_COMMISSION_PER_SHARE = (
    >>>         0.0035 * # IB commission per share
    >>>         (1
    >>>          + 0.000175 # NYSE pass-through (% of IB commission)
    >>>          + 0.00056 # FINRA pass-through (% of IB Commission)
    >>>         )
    >>>         + 0.0002 # clearing fee per share
    >>>         + (0.000119/2) # FINRA activity fee (per share sold)
    >>>     )
    >>>     MAKER_FEE_PER_SHARE = -0.002 # exchange rebate (varies)
    >>>     TAKER_FEE_PER_SHARE = 0.00118 # exchange fee (varies)
    >>>     MAKER_RATIO = 0 # assume we always take liquidity for this strategy
    >>>     MIN_COMMISSION = 0.35
    >>>     TRANSACTION_FEE_RATE = 0.0000221
    >>>
    >>>  # then, use this on your strategy:
    >>>  class MyUSAStrategy(Moonshot):
    >>>      COMMISSION_CLASS = CostPlusUSStockCommission
    """

    IB_COMMISSION_PER_SHARE = None
    IB_COMMISSION_PER_SHARE_TIER_2 = None
    TIER_2_RATIO = 0
    MAKER_FEE_PER_SHARE = 0
    TAKER_FEE_PER_SHARE = 0
    MAKER_RATIO = 0 # ratio of maker trades, between 0 and 1
    MIN_COMMISSION = 0
    TRANSACTION_FEE_RATE = 0

    @classmethod
    def get_commissions(cls, contract_values, trades, nlvs=None):
        """
        Returns a DataFrame of commissions.


        Parameters
        ----------
        contract_values : DataFrame, required
            a DataFrame of contract values (price * multipler / price_magnifier)

        trades : DataFrame of floats, required
            a DataFrame of trades, expressing the percentage of account equity that
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
        trades = trades.abs()

        taker_ratio = 1 - cls.MAKER_RATIO
        exchange_fee_per_share = (cls.MAKER_RATIO * cls.MAKER_FEE_PER_SHARE) + (taker_ratio * cls.TAKER_FEE_PER_SHARE)

        # Calculate commissions as a percent of the share price.
        if cls.TIER_2_RATIO:
            commission_per_share = (
                ((1 - cls.TIER_2_RATIO) * cls.IB_COMMISSION_PER_SHARE)
                + (cls.TIER_2_RATIO * cls.IB_COMMISSION_PER_SHARE_TIER_2)
            )
        else:
            commission_per_share = cls.IB_COMMISSION_PER_SHARE

        share_based_commission_rates = float(commission_per_share)/contract_values

        # Multiply the commissions by the trades.
        share_based_commissions = share_based_commission_rates * trades

        if nlvs is not None and cls.MIN_COMMISSION:
            share_based_commissions = cls._enforce_min_commissions(share_based_commissions, nlvs=nlvs)

        share_based_exchange_fee_rates = exchange_fee_per_share/contract_values
        share_based_exchange_fees = share_based_exchange_fee_rates * trades

        value_based_fees = cls.TRANSACTION_FEE_RATE * trades

        commissions = share_based_commissions + share_based_exchange_fees + value_based_fees

        return commissions

class CostPlusUSStockCommission(BaseCostPlusStockCommission):

    IB_COMMISSION_PER_SHARE = (
        0.0035 * # IB commission per share
        (1
         + 0.000175 # NYSE pass-through (% of IB commission)
         + 0.00056 # FINRA pass-through (% of IB Commission)
        )
        + 0.0002 # clearing fee per share
        + (0.000119/2) # FINRA activity fee (per share sold)
    )
    MAKER_FEE_PER_SHARE = -0.002 # exchange rebate (varies)
    TAKER_FEE_PER_SHARE = 0.00118 # exchange fee (varies)
    MAKER_RATIO = 0
    MIN_COMMISSION = 0.35
    TRANSACTION_FEE_RATE = 0.0000221

class AustraliaStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0008
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 5.00

class CanadaStockCommission(BaseCostPlusStockCommission):

    IB_COMMISSION_PER_SHARE = (
        0.008 # IB commission per share
        + 0.00017 # clearing fee per share
        + 0.00011 # transaction fee per share
        )
    MAKER_FEE_PER_SHARE = -0.0019 # varies
    TAKER_FEE_PER_SHARE = 0.003 # varies
    MAKER_RATIO = 0
    MIN_COMMISSION = 1.00
    TRANSACTION_FEE_RATE = 0

class FranceStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0008
    EXCHANGE_FEE_RATE = 0.000095 # 0.95 bps exchange fee
    MIN_COMMISSION = 1.25 # EUR

class GermanyStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0008
    EXCHANGE_FEE_RATE = 0.000048 + 0.00001 # 0.48 bps exchange fee + 0.1 bps clearing fee
    MIN_COMMISSION = 1.25 # EUR

class HongKongStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0008
    EXCHANGE_FEE_RATE = (
          0.00005 # exchange fee
        + 0.00002 # clearing fee (2 HKD min)
        + 0.001 # Stamp duty
        + 0.000027 # SFC Transaction Levy
    )
    MIN_COMMISSION = 18.00 # HKD

class JapanStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0005
    EXCHANGE_FEE_RATE = 0.000004
    MIN_COMMISSION = 80.00 # JPY

class MexicoStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0010
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 60.00 # MXN

class SingaporeStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0008
    EXCHANGE_FEE_RATE = 0.00034775 + 0.00008025 # transaction fee + access fee
    MIN_COMMISSION = 2.50 # SGD

class UKStockCommission(PercentageCommission):

    IB_COMMISSION_RATE = 0.0008
    EXCHANGE_FEE_RATE = 0.000045 + 0.0025 # 0.45 bps + 0.5% stamp tax on purchases > 1000 GBP
    MIN_COMMISSION = 1.00 # GBP

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

import pandas as pd
from moonshot.commission.base import Commission, PercentageCommission

class PerShareCommission(Commission):
    """
    Base class for commissions which are primarily based on the number of
    shares.

    This class can't be used directly but should be subclassed with the
    appropriate parameters.

    Parameters
    ----------
    BROKER_COMMISSION_PER_SHARE : float, required
        the broker commission per share at the lowest volume tier

    BROKER_COMMISSION_PER_SHARE_TIER_2 : float, optional
        the broker commission per share at volume tier 2

    TIER_2_RATIO : float, optional
        ratio of monthly trades at volume tier 2

    EXCHANGE_FEE_PER_SHARE : float, optional
        the sum of all exchange fees which are assessed per share (excluding maker-taker
        fees, if defined separately)

    MAKER_FEE_PER_SHARE : float, optional
        the "maker" fee from the exchange for adding liquidity. Use a negative value
        to indicate a rebate

    TAKER_FEE_PER_SHARE : float, optional
        the "taker" fee paid to the exchange for removing liquidity

    MAKER_RATIO : float, optional
        the ratio of trades that earn the maker fee (for example if 75% of trades add
        liquidity and 25% remove liquidity, this value should be 0.75)

    PERCENTAGE_FEE_RATE : float, optional
        the sum of all fees which are assessed as a percentage of trade value

    COMMISSION_PERCENTAGE_FEE_RATE : float, optional
        the sum of all fees which are assessed as a percentage of the broker commission

    MIN_COMMISSION : float, optional
        the minimum commission charged by the broker. Only enforced if NLVs are passed
        by the backtest.

    Notes
    -----
    Usage Guide:

    * Moonshot commissions and slippage: https://qrok.it/dl/ms/moonshot-commissions-slippage

    Examples
    --------
    Example subclass for US stock comission with fixed pricing:

    >>> class USStockCommission(PerShareCommission):
    >>>     BROKER_COMMISSION_PER_SHARE = 0.005
    >>>     MIN_COMMISSION = 1.00
    >>>
    >>>  # then, use this on your strategy:
    >>>  class MyUSAStrategy(Moonshot):
    >>>      COMMISSION_CLASS = USStockCommission

    Example subclass for US Cost-Plus stock comissions:

    >>> class CostPlusUSStockCommission(PerShareCommission):
    >>>     BROKER_COMMISSION_PER_SHARE = 0.0035
    >>>     EXCHANGE_FEE_PER_SHARE = (0.0002 # clearing fee per share
    >>>                              + (0.000119/2)) # FINRA activity fee (per share sold)
    >>>     MAKER_FEE_PER_SHARE = -0.002 # exchange rebate (varies)
    >>>     TAKER_FEE_PER_SHARE = 0.00118 # exchange fee (varies)
    >>>     MAKER_RATIO = 0.25 # assume 25% of our trades add liquidity, 75% take liquidity
    >>>     COMMISSION_PERCENTAGE_FEE_RATE = (0.000175 # NYSE pass-through (% of broker commission)
    >>>                                      + 0.00056) # FINRA pass-through (% of broker commission)
    >>>     PERCENTAGE_FEE_RATE = 0.0000231 # Transaction fees
    >>>     MIN_COMMISSION = 0.35
    >>>
    >>>  # then, use this on your strategy:
    >>>  class MyUSAStrategy(Moonshot):
    >>>      COMMISSION_CLASS = CostPlusUSStockCommission
    """

    BROKER_COMMISSION_PER_SHARE: float = None
    """the broker commission per share at the lowest volume tier"""
    BROKER_COMMISSION_PER_SHARE_TIER_2: float = None
    """the broker commission per share at volume tier 2"""
    TIER_2_RATIO: float = 0
    """ratio of monthly trades at volume tier 2"""
    EXCHANGE_FEE_PER_SHARE: float = 0
    """the sum of all exchange fees which are assessed per share (excluding maker-taker
    fees, if defined separately)"""
    MAKER_FEE_PER_SHARE: float = 0
    """the "maker" fee from the exchange for adding liquidity. Use a negative value
    to indicate a rebate"""
    TAKER_FEE_PER_SHARE: float = 0
    """the "taker" fee paid to the exchange for removing liquidity"""
    MAKER_RATIO: float = 0
    """the ratio of trades that earn the maker fee (for example if 75% of trades add
    liquidity and 25% remove liquidity, this value should be 0.75)"""
    PERCENTAGE_FEE_RATE: float = 0
    """the sum of all fees which are assessed as a percentage of trade value"""
    COMMISSION_PERCENTAGE_FEE_RATE: float = 0
    """the sum of all fees which are assessed as a percentage of the broker commission"""
    MIN_COMMISSION: float = 0
    """the minimum commission charged by the broker. Only enforced if NLVs are passed
    by the backtest."""

    @classmethod
    def get_commissions(
        cls,
        contract_values: pd.DataFrame,
        turnover: pd.DataFrame,
        nlvs: pd.DataFrame = None
        ) -> pd.DataFrame:
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
        taker_ratio = 1 - cls.MAKER_RATIO
        exchange_fee_per_share = cls.EXCHANGE_FEE_PER_SHARE + (cls.MAKER_RATIO * cls.MAKER_FEE_PER_SHARE) + (taker_ratio * cls.TAKER_FEE_PER_SHARE)

        # Calculate commissions as a percent of the share price.
        if cls.TIER_2_RATIO:
            broker_commission_per_share = (
                ((1 - cls.TIER_2_RATIO) * cls.BROKER_COMMISSION_PER_SHARE)
                + (cls.TIER_2_RATIO * cls.BROKER_COMMISSION_PER_SHARE_TIER_2)
            )
        else:
            broker_commission_per_share = cls.BROKER_COMMISSION_PER_SHARE

        commission_per_share_with_fees = broker_commission_per_share * (1 + cls.COMMISSION_PERCENTAGE_FEE_RATE)

        # Note: we take abs() of contract_values because combos can have
        # negative prices which would cause a negative commission rate
        broker_commission_rates = float(broker_commission_per_share)/contract_values.where(contract_values != 0).abs()

        # Multiply the commissions by the turnover.
        broker_commissions = broker_commission_rates * turnover

        if nlvs is not None and cls.MIN_COMMISSION:
            broker_commissions = cls._enforce_min_commissions(broker_commissions, nlvs=nlvs)

        share_based_exchange_fee_rates = exchange_fee_per_share/contract_values.where(contract_values != 0).abs()
        share_based_exchange_fees = share_based_exchange_fee_rates * turnover

        value_based_fees = cls.PERCENTAGE_FEE_RATE * turnover

        commission_based_fees = cls.COMMISSION_PERCENTAGE_FEE_RATE * broker_commissions

        commissions = broker_commissions + share_based_exchange_fees + value_based_fees + commission_based_fees

        return commissions

class DemoUSStockCommission(PerShareCommission):

    BROKER_COMMISSION_PER_SHARE: float = 0.005
    MIN_COMMISSION: float = 1.00

class DemoCostPlusUSStockCommission(PerShareCommission):

    BROKER_COMMISSION_PER_SHARE: float = 0.0035
    EXCHANGE_FEE_PER_SHARE: float = (0.0002 # clearing fee per share
                              + (0.000119/2)) # FINRA activity fee (per share sold)
    MAKER_FEE_PER_SHARE: float = -0.002 # exchange rebate (varies)
    TAKER_FEE_PER_SHARE: float = 0.00118 # exchange fee (varies)
    MAKER_RATIO: float = 0
    COMMISSION_PERCENTAGE_FEE_RATE: float = (0.000175 # NYSE pass-through (% of broker commission)
                                      + 0.00056) # FINRA pass-through (% of broker commission)
    PERCENTAGE_FEE_RATE: float = 0.0000231 # Transaction fees
    MIN_COMMISSION: float = 0.35


class DemoCostPlusCanadaStockCommission(PerShareCommission):

    BROKER_COMMISSION_PER_SHARE: float = 0.008
    EXCHANGE_FEE_PER_SHARE: float = (
        0.00017 # clearing fee per share
        + 0.00011 # transaction fee per share
        )
    MAKER_FEE_PER_SHARE: float = -0.0019 # varies
    TAKER_FEE_PER_SHARE: float = 0.003 # varies
    MAKER_RATIO: float = 0
    MIN_COMMISSION: float = 1.00
    TRANSACTION_FEE_RATE: float = 0

class DemoAustraliaStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0008
    EXCHANGE_FEE_RATE: float = 0
    MIN_COMMISSION: float = 5.00

class DemoFranceStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0008
    EXCHANGE_FEE_RATE: float = 0.000095 # 0.95 bps exchange fee
    MIN_COMMISSION: float = 1.25 # EUR

class DemoGermanyStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0008
    EXCHANGE_FEE_RATE: float = 0.000048 + 0.00001 # 0.48 bps exchange fee + 0.1 bps clearing fee
    MIN_COMMISSION: float = 1.25 # EUR

class DemoHongKongStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0008
    EXCHANGE_FEE_RATE: float = (
          0.00005 # exchange fee
        + 0.00002 # clearing fee (2 HKD min)
        + 0.001 # Stamp duty
        + 0.000027 # SFC Transaction Levy
    )
    MIN_COMMISSION: float = 18.00 # HKD

class DemoJapanStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0005
    EXCHANGE_FEE_RATE: float = 0.000004
    MIN_COMMISSION: float = 80.00 # JPY

class DemoMexicoStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0010
    EXCHANGE_FEE_RATE: float = 0
    MIN_COMMISSION: float = 60.00 # MXN

class DemoSingaporeStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0008
    EXCHANGE_FEE_RATE: float = 0.00034775 + 0.00008025 # transaction fee + access fee
    MIN_COMMISSION: float = 2.50 # SGD

class DemoUKStockCommission(PercentageCommission):

    BROKER_COMMISSION_RATE: float = 0.0008
    EXCHANGE_FEE_RATE: float = 0.000045 + 0.0025 # 0.45 bps + 0.5% stamp tax on purchases > 1000 GBP
    MIN_COMMISSION: float = 1.00 # GBP

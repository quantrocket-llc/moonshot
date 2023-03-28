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

from typing import Any
import pandas as pd
from moonshot.commission.base import Commission, PercentageCommission

class FuturesCommission(Commission):
    """
    Base class for futures commissions.

    Parameters
    ----------
    BROKER_COMMISSION_PER_CONTRACT : float
        the commission per contract

    EXCHANGE_FEE_PER_CONTRACT : float
        the exchange and regulatory fees per contract

    CARRYING_FEE_PER_CONTRACT : float
        the overnight carrying fee per contract (depends on equity in excess of
        margin requirement)

    Notes
    -----
    Usage Guide:

    * Moonshot commissions and slippage: https://qrok.it/dl/ms/moonshot-commissions-slippage

    Examples
    --------
    Example subclass for CME E-Mini commissions:

    >>> class CMEEquityEMiniFixedCommission(FuturesCommission):
    >>>     BROKER_COMMISSION_PER_CONTRACT = 0.85
    >>>     EXCHANGE_FEE_PER_CONTRACT = 1.18
    >>>     CARRYING_FEE_PER_CONTRACT = 0 # Depends on equity in excess of margin requirement
    >>>
    >>>  # then, use this on your strategy:
    >>>  class MyEminiStrategy(Moonshot):
    >>>      COMMISSION_CLASS = CMEEquityEMiniFixedCommission
    """
    BROKER_COMMISSION_PER_CONTRACT: float = 0
    """the commission per contract"""
    EXCHANGE_FEE_PER_CONTRACT: float = 0
    """the exchange and regulatory fees per contract"""
    CARRYING_FEE_PER_CONTRACT: float = 0
    """the overnight carrying fee per contract (depends on equity in excess of
    margin requirement)"""

    @classmethod
    def get_commissions(
        cls,
        contract_values: pd.DataFrame,
        turnover: pd.DataFrame,
        **kwargs: Any
        ) -> pd.DataFrame:
        """
        Return a DataFrame of commissions as percentages of account equity.
        """
        cost_per_contract = cls.BROKER_COMMISSION_PER_CONTRACT + cls.EXCHANGE_FEE_PER_CONTRACT + cls.CARRYING_FEE_PER_CONTRACT

        # Express the commission as a percent of contract value
        commission_rates = float(cost_per_contract)/contract_values

        # Multipy the commission rates by the turnover
        commissions = commission_rates * turnover

        return commissions

class DemoCMEEquityEMiniFixedCommission(FuturesCommission):
    """
    Fixed commission for CME Equity E-Minis.
    """
    BROKER_COMMISSION_PER_CONTRACT: float = 0.85
    EXCHANGE_FEE_PER_CONTRACT: float = 1.18
    CARRYING_FEE_PER_CONTRACT: float = 0

class DemoCanadaCADFuturesTieredCommission(FuturesCommission):
    """
    Tiered/Cost-Plus commission for Canada futures denominated in CAD, for US
    customers.
    """

    BROKER_COMMISSION_PER_CONTRACT: float = 0.85
    EXCHANGE_FEE_PER_CONTRACT: float = (
        1.12   # exchange fee
        + 0.03 # regulatory fee
        + 0.01 # NFA assessment fee
    )
    CARRYING_FEE_PER_CONTRACT: float = 0

class DemoKoreaFuturesCommission(PercentageCommission):
    """
    Fixed rate commission for Korea futures excluding stock futures.
    """
    # 0.4 bps fixed rate, excludes stock futures and KWY futures (US dollar)

    BROKER_COMMISSION_RATE: float = 0.00004
    EXCHANGE_FEE_RATE: float = 0
    MIN_COMMISSION: float = 0

class DemoKoreaStockFuturesCommission(PercentageCommission):
    """
    Fixed rate commission for Korea stock futures.
    """
    # 4 bps fixed rate for stock futures

    BROKER_COMMISSION_RATE: float = 0.0004
    EXCHANGE_FEE_RATE: float = 0
    MIN_COMMISSION: float = 0

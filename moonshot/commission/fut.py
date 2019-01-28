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

class FuturesCommission(BaseCommission):
    """
    Base class for futures commissions.

    Parameters
    ----------
    IB_COMMISSION_PER_CONTRACT : float
        the commission per contract

    EXCHANGE_FEE_PER_CONTRACT : float
        the exchange and regulatory fees per contract

    CARRYING_FEE_PER_CONTRACT : float
        the overnight carrying fee per contract (depends on equity in excess of
        margin requirement)

    Examples
    --------
    Example subclass for Globex E-Mini commissions:

    >>> class GlobexEquityEMiniFixedCommission(FuturesCommission):
    >>>     IB_COMMISSION_PER_CONTRACT = 0.85
    >>>     EXCHANGE_FEE_PER_CONTRACT = 1.18
    >>>     CARRYING_FEE_PER_CONTRACT = 0 # Depends on equity in excess of margin requirement
    >>>
    >>>  # then, use this on your strategy:
    >>>  class MyEminiStrategy(Moonshot):
    >>>      COMMISSION_CLASS = GlobexEquityEMiniFixedCommission
    """
    IB_COMMISSION_PER_CONTRACT = 0
    EXCHANGE_FEE_PER_CONTRACT = 0
    CARRYING_FEE_PER_CONTRACT = 0 # Depends on equity in excess of margin requirement

    @classmethod
    def get_commissions(cls, contract_values, turnover, **kwargs):
        """
        Return a DataFrame of commissions as percentages of account equity.
        """
        cost_per_contract = cls.IB_COMMISSION_PER_CONTRACT + cls.EXCHANGE_FEE_PER_CONTRACT + cls.CARRYING_FEE_PER_CONTRACT

        # Express the commission as a percent of contract value
        commission_rates = float(cost_per_contract)/contract_values

        # Multipy the commission rates by the turnover
        commissions = commission_rates * turnover

        return commissions

class DemoGlobexEquityEMiniFixedCommission(FuturesCommission):
    """
    Fixed commission for Globex Equity E-Minis.
    """
    IB_COMMISSION_PER_CONTRACT = 0.85
    EXCHANGE_FEE_PER_CONTRACT = 1.18
    CARRYING_FEE_PER_CONTRACT = 0

class DemoCanadaCADFuturesTieredCommission(FuturesCommission):
    """
    Tiered/Cost-Plus commission for Canada futures denominated in CAD, for US
    customers.
    """

    IB_COMMISSION_PER_CONTRACT = 0.85
    EXCHANGE_FEE_PER_CONTRACT = (
        1.12   # exchange fee
        + 0.03 # regulatory fee
        + 0.01 # NFA assessment fee
    )
    CARRYING_FEE_PER_CONTRACT = 0

class DemoKoreaFuturesCommission(PercentageCommission):
    """
    Fixed rate commission for Korea futures excluding stock futures.
    """
    # 0.4 bps fixed rate, excludes stock futures and KWY futures (US dollar)

    IB_COMMISSION_RATE = 0.00004
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 0

class DemoKoreaStockFuturesCommission(PercentageCommission):
    """
    Fixed rate commission for Korea stock futures.
    """
    # 4 bps fixed rate for stock futures

    IB_COMMISSION_RATE = 0.0004
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 0

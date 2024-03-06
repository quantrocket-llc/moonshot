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

from moonshot.commission.base import PercentageCommission

class SpotFXCommission(PercentageCommission):
    """
    Commission class for spot FX. This class can be used as-is.

    Notes
    -----
    Min commissions are not modeled for spot FX. This is because min
    commissions for spot FX are in USD ($2), regardless of the quote
    currency. The Moonshot class passes NLVs in the quote currency (the
    Currency field). To accurately model min commissions, these NLVs would need
    to be converted to USD.

    Usage Guide:

    * Moonshot commissions and slippage: https://qrok.it/dl/ms/moonshot-commissions-slippage

    Examples
    --------
    Use this on your strategy:

    >>>  class MyFXStrategy(Moonshot):
    >>>      COMMISSION_CLASS = SpotFXCommission

    """

    BROKER_COMMISSION_RATE: float = 0.00002 # 0.2 bps
    """the commission rate (as a percentage of trade value) charged by the broker"""
    EXCHANGE_FEE_RATE: float = 0
    """the exchange fee as a percentage of trade value"""
    MIN_COMMISSION: float = 0
    """NOTE: min commissions are not modeled for spot FX. This is because min
    commissions for spot FX are in USD ($2), regardless of the quote
    currency. The Moonshot class passes NLVs in the quote currency (the
    Currency field). To accurately model min commissions, these NLVs would need
    to be converted to USD."""

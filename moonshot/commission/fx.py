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

from moonshot.commission import PercentageCommission

class SpotForexCommission(PercentageCommission):
    """
    Commission class for spot forex. This class can be used as-is.

    NOTE: min commissions are not modeled for spot forex. This is because min
    commissions for spot forex are in USD ($2), regardless of the quote
    currency. The Moonshot class passes NLVs in the quote currency (the
    Currency field). To accurately model min commissions, these NLVs would need
    to be converted to USD.

    Examples
    --------
    Use this on your strategy:

    >>>  class MyForexStrategy(Moonshot):
    >>>      COMMISSION_CLASS = SpotForexCommission

    """

    IB_COMMISSION_RATE = 0.00002 # 0.2 bps
    EXCHANGE_FEE_RATE = 0
    MIN_COMMISSION = 0 # see NOTE in docstring

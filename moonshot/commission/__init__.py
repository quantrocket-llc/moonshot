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
"""
Moonshot commission classes. Except as noted below, all commission classes
must be subclassed to be used, filling in specific parameters for your
commission structure.

Classes
-------
Commission
    Base class for all commission classes.

FuturesCommission
    Base class for futures commissions.

PercentageCommission
    Base class for commissions which are a fixed percentage of the trade
    value.

NoCommission
    Commission class for strategies that don't pay commissions.
    This class can be used as-is.

PerShareCommission
    Base class for commissions which are primarily based on the number of
    shares.

SpotFXCommission
    Commission class for spot FX. This class can be used as-is.

Notes
-----
Usage Guide:

* Moonshot commissions and slippage: https://qrok.it/dl/ms/moonshot-commissions-slippage
"""
from .fut import FuturesCommission
from .base import Commission, PercentageCommission, NoCommission
from .stk import PerShareCommission
from .fx import SpotFXCommission

# alias
SpotForexCommission = SpotFXCommission


__all__ = [
    'Commission',
    'FuturesCommission',
    'PercentageCommission',
    'NoCommission',
    'PerShareCommission',
    'SpotFXCommission',
]

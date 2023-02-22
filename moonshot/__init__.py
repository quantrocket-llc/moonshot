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
Vectorized backtesting and trading engine.

Classes
-------
Moonshot
    Base class for Moonshot strategies.

MoonshotML
    Base class for Moonshot machine learning strategies.

Modules
-------
commission
    Moonshot commission classes.

slippage
    Moonshot slippage classes.
"""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .strategies import Moonshot, MoonshotML
from . import slippage
from . import commission

__all__ = [
    'Moonshot',
    'MoonshotML',
    'slippage',
    'commission'
]

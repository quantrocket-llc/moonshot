"""
Moonshot slippage classes.

Classes
-------
Slippage
    Base class for slippage classes. This class must be subclassed
    to be used.

FixedSlippage
    Apply a fixed pct slippage to each trade.

IBKRBorrowFees
    Apply borrow fees to each short position.

Notes
-----
Usage Guide:

* Moonshot commissions and slippage: https://qrok.it/dl/ms/moonshot-commissions-slippage

"""

from .base import Slippage
from .fixed import FixedSlippage
from .borrowfee import IBKRBorrowFees

__all__ = [
    'Slippage',
    'FixedSlippage',
    'IBKRBorrowFees',
]
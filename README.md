# Moonshot

Moonshot is a backtester designed for data scientists, created by and for [QuantRocket](https://www.quantrocket.com).

## Key features

**Pandas-based**: Moonshot is based on Pandas, the centerpiece of the Python data science stack. If you love Pandas you'll love Moonshot. Moonshot can be thought of as a set of conventions for organizing Pandas code for the purpose of running backtests.

**Lightweight**: Moonshot is simple and lightweight because it relies on the power and flexibility of Pandas and doesn't attempt to re-create functionality that Pandas can already do. No bloated codebase full of countless indicators and models to import and learn. Most of Moonshot's code is contained in a single `Moonshot` class.

**Fast**: Moonshot is fast because Pandas is fast. No event-driven backtester can match Moonshot's speed. Speed promotes alpha discovery by facilitating rapid experimentation and research iteration.

**Multi-asset class, multi-time frame**: Moonshot supports end-of-day and intraday strategies using equities, futures, and forex.

**Live trading**: Live trading with Moonshot can be thought of as running a backtest on up-to-date historical data and generating a batch of orders based on the latest signals produced by the backtest.

**No black boxes, no magic**: Moonshot provides many conveniences to make backtesting easier, but it eschews hidden behaviors and complex, under-the-hood simulation rules that are hard to understand or audit. What you see is what you get.

## Example

A basic Moonshot strategy is shown below:

```python
from moonshot import Moonshot

class DualMovingAverageStrategy(Moonshot):

    CODE = "dma-tech"
    DB = "tech-giants-1d"
    LMAVG_WINDOW = 300
    SMAVG_WINDOW = 100

    def prices_to_signals(self, prices):
        closes = prices.loc["Close"]

        # Compute long and short moving averages
        lmavgs = closes.rolling(self.LMAVG_WINDOW).mean()
        smavgs = closes.rolling(self.SMAVG_WINDOW).mean()

        # Go long when short moving average is above long moving average
        signals = smavgs > lmavgs

        return signals.astype(int)

    def signals_to_target_weights(self, signals, prices):
        # spread our capital equally among our trades on any given day
        daily_signal_counts = signals.abs().sum(axis=1)
        weights = signals.div(daily_signal_counts, axis=0).fillna(0)
        return weights

    def target_weights_to_positions(self, weights, prices):
        # we'll enter in the period after the signal
        positions = weights.shift()
        return positions

    def positions_to_gross_returns(self, positions, prices):
        # Our return is the security's close-to-close return, multiplied by
        # the size of our position
        closes = prices.loc["Close"]
        gross_returns = closes.pct_change() * positions.shift()
        return gross_returns
```

See the [QuantRocket docs](https://www.quantrocket.com/docs/#moonshot-backtesting) for a fuller discussion.

## FAQ

### Can I use Moonshot without QuantRocket?

Moonshot depends on QuantRocket for querying historical data in backtesting and for live trading. In the future we hope to add support for running Moonshot on a CSV of data to allow backtesting outside of QuantRocket.

## See also

[Moonchart](https://github.com/quantrocket-llc/moonchart) is a companion library for creating performance tear sheets from a Moonshot backtest.

## License

Moonshot is distributed under the Apache 2.0 License. See the LICENSE file in the release for details.

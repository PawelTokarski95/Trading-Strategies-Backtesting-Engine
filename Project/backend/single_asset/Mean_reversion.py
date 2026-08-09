import backtrader as bt
import math


class RollingZScore(bt.Indicator):
    lines = ("zscore",)

    params = (
        ("period", 60),
        ("epsilon", 1e-12),
    )

    def __init__(self):
        self.mean = bt.indicators.SMA(
            self.data,
            period=self.p.period,
        )

        self.std = bt.indicators.StandardDeviation(
            self.data,
            period=self.p.period,
        )

    def next(self):
        value = float(self.data[0])
        mean = float(self.mean[0])
        std = float(self.std[0])

        if not all(
            math.isfinite(x)
            for x in (value, mean, std)
        ):
            self.lines.zscore[0] = float("nan")
            return

        if std <= self.p.epsilon:
            self.lines.zscore[0] = 0.0
            return

        self.lines.zscore[0] = (
            value - mean
        ) / std


class MeanReversionStrategy(bt.Strategy):

    params = (
        # Signal
        ("lookback", 60),
        ("entry_z", 2.0),
        ("exit_z", 0.5),
        ("stop_z", 3.5),

        # Risk
        ("max_exposure", 0.20),
        ("risk_per_trade", 0.01),
        ("atr_stop_mult", 2.0),
        ("max_holding_bars", 50),
        ("max_drawdown", 0.15),

        # Volatility filter
        ("atr_period", 14),
        ("max_atr_pct", 0.08),

        # Short selling
        ("allow_short", True),
    )

    def __init__(self):
        self.price = self.data.close

        self.mean = bt.indicators.SMA(
            self.price,
            period=self.p.lookback,
        )

        self.std = bt.indicators.StandardDeviation(
            self.price,
            period=self.p.lookback,
        )

        self.zscore = RollingZScore(
            self.price,
            period=self.p.lookback,
        )

        self.atr = bt.indicators.ATR(
            self.data,
            period=self.p.atr_period,
        )

        self.order = None

        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None

        self.position_state = "flat"

        self.peak_equity = None
        self.trading_disabled = False

    # =====================================================
    # ORDER MANAGEMENT
    # =====================================================

    def notify_order(self, order):
        if order.status in (
            order.Submitted,
            order.Accepted,
        ):
            return

        if order.status == order.Completed:
            position = self.getposition(self.data)

            # Entry completed
            if (
                self.position_state == "opening_long"
                and position.size > 0
            ):
                self.position_state = "long"
                self.entry_price = float(order.executed.price)
                self.entry_bar = len(self)

                atr = float(self.atr[0])

                if math.isfinite(atr) and atr > 0:
                    self.entry_atr = atr
                    self.stop_price = (
                        self.entry_price
                        - self.p.atr_stop_mult * atr
                    )

            elif (
                self.position_state == "opening_short"
                and position.size < 0
            ):
                self.position_state = "short"
                self.entry_price = float(order.executed.price)
                self.entry_bar = len(self)

                atr = float(self.atr[0])

                if math.isfinite(atr) and atr > 0:
                    self.entry_atr = atr
                    self.stop_price = (
                        self.entry_price
                        + self.p.atr_stop_mult * atr
                    )

            # Exit completed
            elif position.size == 0:
                self.reset_trade_state()

            self.order = None

        elif order.status in (
            order.Canceled,
            order.Margin,
            order.Rejected,
        ):
            if self.position_state in (
                "opening_long",
                "opening_short",
            ):
                self.reset_trade_state()

            self.order = None

    def reset_trade_state(self):
        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.position_state = "flat"

    # =====================================================
    # PORTFOLIO DRAWDOWN
    # =====================================================

    def calculate_drawdown(self):
        equity = self.broker.getvalue()

        if self.peak_equity is None:
            self.peak_equity = equity

        self.peak_equity = max(
            self.peak_equity,
            equity,
        )

        if self.peak_equity <= 0:
            return 0.0

        return (
            self.peak_equity - equity
        ) / self.peak_equity

    # =====================================================
    # VOLATILITY FILTER
    # =====================================================

    def volatility_ok(self):
        price = float(self.price[0])
        atr = float(self.atr[0])

        if not all(
            math.isfinite(x)
            for x in (price, atr)
        ):
            return False

        if price <= 0 or atr <= 0:
            return False

        atr_pct = atr / price

        return atr_pct <= self.p.max_atr_pct

    # =====================================================
    # POSITION SIZING
    # =====================================================

    def calculate_position_size(self):
        equity = self.broker.getvalue()
        price = float(self.price[0])
        atr = float(self.atr[0])

        if not all(
            math.isfinite(x)
            for x in (
                equity,
                price,
                atr,
            )
        ):
            return 0

        if (
            equity <= 0
            or price <= 0
            or atr <= 0
            or self.p.atr_stop_mult <= 0
        ):
            return 0

        # Capital loss allowed if the ATR stop is hit.
        risk_budget = (
            equity
            * self.p.risk_per_trade
        )

        stop_distance = (
            atr
            * self.p.atr_stop_mult
        )

        risk_size = (
            risk_budget
            / stop_distance
        )

        max_capital = (
            equity
            * self.p.max_exposure
        )

        exposure_size = (
            max_capital
            / price
        )

        size = min(
            risk_size,
            exposure_size,
        )

        return max(
            int(size),
            0,
        )

    # =====================================================
    # ENTRY
    # =====================================================

    def open_long(self):
        if self.order:
            return

        size = self.calculate_position_size()

        if size <= 0:
            return

        self.position_state = "opening_long"

        self.order = self.buy(
            size=size,
        )

    def open_short(self):
        if not self.p.allow_short:
            return

        if self.order:
            return

        size = self.calculate_position_size()

        if size <= 0:
            return

        self.position_state = "opening_short"

        self.order = self.sell(
            size=size,
        )

    # =====================================================
    # EXIT
    # =====================================================

    def close_position(self):
        if self.order:
            return

        if not self.position:
            return

        self.order = self.close()

    # =====================================================
    # TIME STOP
    # =====================================================

    def holding_period_exceeded(self):
        if self.entry_bar is None:
            return False

        bars_held = (
            len(self)
            - self.entry_bar
        )

        return (
            bars_held
            >= self.p.max_holding_bars
        )

    # =====================================================
    # ATR STOP
    # =====================================================

    def atr_stop_triggered(self):
        if self.stop_price is None:
            return False

        if not math.isfinite(self.stop_price):
            return False

        if self.position.size > 0:
            low = float(self.data.low[0])

            if not math.isfinite(low):
                return False

            return low <= self.stop_price

        if self.position.size < 0:
            high = float(self.data.high[0])

            if not math.isfinite(high):
                return False

            return high >= self.stop_price

        return False

    # =====================================================
    # MAIN STRATEGY
    # =====================================================

    def next(self):
        if self.order:
            return

        z = float(self.zscore[0])
        price = float(self.price[0])
        mean = float(self.mean[0])

        if not all(
            math.isfinite(x)
            for x in (
                z,
                price,
                mean,
            )
        ):
            return

        # Portfolio drawdown protection
        drawdown = self.calculate_drawdown()

        if drawdown >= self.p.max_drawdown:
            if self.position:
                self.close_position()

            self.trading_disabled = True
            return

        if self.trading_disabled:
            return

        # No position
        if not self.position:
            if self.position_state not in (
                "opening_long",
                "opening_short",
            ):
                self.position_state = "flat"

            if not self.volatility_ok():
                return

            if z <= -self.p.entry_z:
                self.open_long()
                return

            if (
                z >= self.p.entry_z
                and self.p.allow_short
            ):
                self.open_short()
                return

            return

        # Real ATR-based risk stop.
        # Note: because this is checked bar-by-bar, execution still occurs
        # according to Backtrader's order execution model rather than
        # necessarily exactly at stop_price after a gap.
        if self.atr_stop_triggered():
            self.close_position()
            return

        # Time stop
        if self.holding_period_exceeded():
            self.close_position()
            return

        # Long
        if self.position.size > 0:
            if z <= -self.p.stop_z:
                self.close_position()
                return

            if z >= -self.p.exit_z:
                self.close_position()
                return

        # Short
        elif self.position.size < 0:
            if z >= self.p.stop_z:
                self.close_position()
                return

            if z <= self.p.exit_z:
                self.close_position()
                return
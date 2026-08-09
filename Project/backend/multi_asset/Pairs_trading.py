import backtrader as bt
import math
import numpy as np
from statsmodels.tsa.stattools import coint


class RollingZScore(bt.Indicator):
    lines = ("zscore",)
    params = (("period", 60), ("epsilon", 1e-12))

    def __init__(self):
        self.rolling_mean = bt.indicators.SMA(self.data, period=self.p.period)
        self.rolling_std = bt.indicators.StandardDeviation(self.data, period=self.p.period)

    def next(self):
        value = float(self.data[0])
        mean = float(self.rolling_mean[0])
        std = float(self.rolling_std[0])
        if not all(math.isfinite(x) for x in (value, mean, std)):
            self.lines.zscore[0] = float("nan")
            return
        if std <= self.p.epsilon:
            self.lines.zscore[0] = 0.0
            return
        self.lines.zscore[0] = (value - mean) / std


class LogValue(bt.Indicator):
    lines = ("value",)

    def next(self):
        current_value = float(self.data[0])
        if not math.isfinite(current_value) or current_value <= 0:
            self.lines.value[0] = float("nan")
            return
        self.lines.value[0] = math.log(current_value)


class DynamicSpread(bt.Indicator):
    lines = ("spread", "hedge_ratio", "intercept")
    params = (("period", 60),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        # Use only completed historical observations to estimate alpha/beta,
        # then apply them to the current bar. This avoids look-ahead bias.
        a = np.asarray([float(self.data0[-i]) for i in range(1, self.p.period + 1)], dtype=float)
        b = np.asarray([float(self.data1[-i]) for i in range(1, self.p.period + 1)], dtype=float)

        if not np.isfinite(a).all() or not np.isfinite(b).all():
            self.lines.spread[0] = float("nan")
            self.lines.hedge_ratio[0] = float("nan")
            self.lines.intercept[0] = float("nan")
            return

        x = np.column_stack((np.ones(len(b)), b))
        try:
            alpha, beta = np.linalg.lstsq(x, a, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.lines.spread[0] = float("nan")
            self.lines.hedge_ratio[0] = float("nan")
            self.lines.intercept[0] = float("nan")
            return

        current_a = float(self.data0[0])
        current_b = float(self.data1[0])
        if not all(math.isfinite(v) for v in (alpha, beta, current_a, current_b)):
            self.lines.spread[0] = float("nan")
            self.lines.hedge_ratio[0] = float("nan")
            self.lines.intercept[0] = float("nan")
            return

        self.lines.intercept[0] = alpha
        self.lines.hedge_ratio[0] = beta
        self.lines.spread[0] = current_a - alpha - beta * current_b


class RollingCointegrationPValue(bt.Indicator):
    lines = ("pvalue",)
    params = (("period", 120),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        # Also uses only completed historical observations.
        a = np.asarray([float(self.data0[-i]) for i in range(1, self.p.period + 1)], dtype=float)
        b = np.asarray([float(self.data1[-i]) for i in range(1, self.p.period + 1)], dtype=float)

        if not np.isfinite(a).all() or not np.isfinite(b).all():
            self.lines.pvalue[0] = float("nan")
            return

        try:
            _, pvalue, _ = coint(a, b, trend="c", autolag="aic")
        except Exception:
            self.lines.pvalue[0] = float("nan")
            return

        self.lines.pvalue[0] = float(pvalue)


class PairsTradingStrategy(bt.Strategy):
    params = (
        ("lookback", 60),
        ("entry_z", 2.0),
        ("exit_z", 0.5),
        ("gross_exposure", 0.20),
        # Maximum loss of the pair as a fraction of equity captured at entry.
        ("risk_per_trade", 0.01),
        ("stop_z", 4.0),
        ("max_holding_bars", 100),
        ("min_hedge_ratio", 0.10),
        ("max_hedge_ratio", 5.00),
        ("cointegration_period", 120),
        ("max_coint_pvalue", 0.05),
        ("require_cointegration", True),
        ("max_strategy_drawdown", 0.15),
    )

    def __init__(self):
        if len(self.datas) != 2:
            raise ValueError("PairsTradingStrategy wymaga dokładnie dwóch data feedów.")

        self.asset_a = self.datas[0]
        self.asset_b = self.datas[1]

        self.log_a = LogValue(self.asset_a.close)
        self.log_b = LogValue(self.asset_b.close)

        self.dynamic_pair = DynamicSpread(self.log_a, self.log_b, period=self.p.lookback)
        self.spread = self.dynamic_pair.spread
        self.hedge_ratio = self.dynamic_pair.hedge_ratio
        self.intercept = self.dynamic_pair.intercept
        self.zscore = RollingZScore(self.spread, period=self.p.lookback)

        self.coint_pvalue = RollingCointegrationPValue(
            self.log_a,
            self.log_b,
            period=max(self.p.cointegration_period, self.p.lookback),
        )

        self.pending_orders = []
        self.pair_state = "flat"
        self.requested_state = None
        self.entry_bar = None
        self.entry_equity = None
        self.entry_prices = {}
        self.entry_sizes = {}
        self.peak_equity = None
        self.trading_disabled = False
        self.entry_failed = False

    def has_pending_orders(self):
        return any(order.status in (order.Submitted, order.Accepted) for order in self.pending_orders)

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            self.entry_prices[order.data] = float(order.executed.price)
            self.entry_sizes[order.data] = int(self.getposition(order.data).size)

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            if self.requested_state in ("long_spread", "short_spread"):
                self.entry_failed = True

        if order in self.pending_orders:
            self.pending_orders.remove(order)

        if not self.pending_orders:
            if self.entry_failed:
                self.entry_failed = False
                self.requested_state = None
                self._flatten_broken_pair()
                return

            if self.requested_state in ("long_spread", "short_spread"):
                pos_a = self.getposition(self.asset_a).size
                pos_b = self.getposition(self.asset_b).size
                if pos_a != 0 and pos_b != 0:
                    self.pair_state = self.requested_state
                    self.entry_bar = len(self)
                    self.entry_equity = float(self.broker.getvalue())
                else:
                    self._flatten_broken_pair()
                self.requested_state = None

            elif self.requested_state == "flat":
                if self.getposition(self.asset_a).size == 0 and self.getposition(self.asset_b).size == 0:
                    self.pair_state = "flat"
                    self.entry_bar = None
                    self.entry_equity = None
                    self.entry_prices.clear()
                    self.entry_sizes.clear()
                self.requested_state = None

    def _flatten_broken_pair(self):
        self.pair_state = "flat"
        self.entry_bar = None
        self.entry_equity = None
        self.entry_prices.clear()
        self.entry_sizes.clear()
        self.requested_state = "flat"

        for data in (self.asset_a, self.asset_b):
            if self.getposition(data).size != 0:
                order = self.close(data=data)
                if order:
                    self.pending_orders.append(order)

        if not self.pending_orders:
            self.requested_state = None

    def update_drawdown(self):
        equity = float(self.broker.getvalue())
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def calculate_position_sizes(self):
        equity = float(self.broker.getvalue())
        price_a = float(self.asset_a.close[0])
        price_b = float(self.asset_b.close[0])
        beta = float(self.hedge_ratio[0])

        if not all(math.isfinite(x) for x in (equity, price_a, price_b, beta)):
            return 0, 0
        if equity <= 0 or price_a <= 0 or price_b <= 0:
            return 0, 0
        if beta < self.p.min_hedge_ratio or beta > self.p.max_hedge_ratio:
            return 0, 0

        gross_capital = equity * self.p.gross_exposure
        capital_a = gross_capital / (1.0 + beta)
        capital_b = gross_capital - capital_a

        size_a = int(capital_a / price_a)
        size_b = int(capital_b / price_b)
        return max(size_a, 0), max(size_b, 0)

    def open_long_spread(self):
        size_a, size_b = self.calculate_position_sizes()
        if size_a <= 0 or size_b <= 0:
            return

        order_a = self.buy(data=self.asset_a, size=size_a)
        order_b = self.sell(data=self.asset_b, size=size_b)
        self.pending_orders.extend([order_a, order_b])
        self.requested_state = "long_spread"
        self.entry_failed = False

    def open_short_spread(self):
        size_a, size_b = self.calculate_position_sizes()
        if size_a <= 0 or size_b <= 0:
            return

        order_a = self.sell(data=self.asset_a, size=size_a)
        order_b = self.buy(data=self.asset_b, size=size_b)
        self.pending_orders.extend([order_a, order_b])
        self.requested_state = "short_spread"
        self.entry_failed = False

    def close_pair(self):
        if self.has_pending_orders():
            return

        orders = []
        for data in (self.asset_a, self.asset_b):
            if self.getposition(data).size != 0:
                order = self.close(data=data)
                if order:
                    orders.append(order)

        if orders:
            self.pending_orders.extend(orders)
            self.requested_state = "flat"
        else:
            self.pair_state = "flat"
            self.entry_bar = None
            self.entry_equity = None
            self.entry_prices.clear()
            self.entry_sizes.clear()

    def pair_unrealized_pnl(self):
        if self.pair_state == "flat":
            return 0.0

        pnl = 0.0
        for data in (self.asset_a, self.asset_b):
            position = self.getposition(data)
            if position.size == 0:
                continue
            current_price = float(data.close[0])
            entry_price = float(position.price)
            if math.isfinite(current_price) and math.isfinite(entry_price):
                pnl += position.size * (current_price - entry_price)
        return pnl

    def trade_loss_exceeded(self):
        if self.entry_equity is None:
            return False
        max_loss = self.entry_equity * self.p.risk_per_trade
        return self.pair_unrealized_pnl() <= -max_loss

    def holding_period_exceeded(self):
        return self.entry_bar is not None and len(self) - self.entry_bar >= self.p.max_holding_bars

    def cointegration_ok(self):
        if not self.p.require_cointegration:
            return True
        pvalue = float(self.coint_pvalue[0])
        return math.isfinite(pvalue) and pvalue <= self.p.max_coint_pvalue

    def next(self):
        if self.has_pending_orders():
            return

        z = float(self.zscore[0])
        beta = float(self.hedge_ratio[0])
        if not all(math.isfinite(x) for x in (z, beta)):
            return

        drawdown = self.update_drawdown()
        if drawdown >= self.p.max_strategy_drawdown:
            if self.pair_state != "flat":
                self.close_pair()
            self.trading_disabled = True
            return

        if self.trading_disabled:
            return

        beta_ok = self.p.min_hedge_ratio <= beta <= self.p.max_hedge_ratio
        coint_ok = self.cointegration_ok()

        if not beta_ok or not coint_ok:
            if self.pair_state != "flat":
                self.close_pair()
            return

        if self.pair_state == "flat":
            if z <= -self.p.entry_z:
                self.open_long_spread()
            elif z >= self.p.entry_z:
                self.open_short_spread()
            return

        if self.trade_loss_exceeded():
            self.close_pair()
            return

        if self.holding_period_exceeded():
            self.close_pair()
            return

        if self.pair_state == "long_spread":
            if z <= -self.p.stop_z or z >= -self.p.exit_z:
                self.close_pair()
                return

        elif self.pair_state == "short_spread":
            if z >= self.p.stop_z or z <= self.p.exit_z:
                self.close_pair()
                return
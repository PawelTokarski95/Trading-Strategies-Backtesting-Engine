import backtrader as bt
import math


class RollingZScore(bt.Indicator):
    lines = ("zscore",)
    params = (("period", 60), ("epsilon", 1e-12))

    def __init__(self):
        self.mean = bt.indicators.SMA(self.data, period=self.p.period)
        self.std = bt.indicators.StandardDeviation(self.data, period=self.p.period)

    def next(self):
        value = float(self.data[0])
        mean = float(self.mean[0])
        std = float(self.std[0])
        if not all(math.isfinite(x) for x in (value, mean, std)):
            self.lines.zscore[0] = float("nan")
            return
        if std <= self.p.epsilon:
            self.lines.zscore[0] = 0.0
            return
        self.lines.zscore[0] = (value - mean) / std


class MultiAssetMeanReversionStrategy(bt.Strategy):
    params = (
        ("lookback", 60),
        ("entry_z", 2.0),
        ("exit_z", 0.5),
        ("stop_z", 3.5),
        ("atr_period", 14),
        ("max_atr_pct", 0.08),
        ("risk_per_trade", 0.005),
        ("atr_stop_mult", 2.5),
        ("max_asset_exposure", 0.15),
        ("max_gross_exposure", 0.60),
        ("max_positions", 5),
        ("max_holding_bars", 40),
        ("max_drawdown", 0.15),
        ("allow_short", True),
    )

    def __init__(self):
        self.zscores = {}
        self.atrs = {}
        self.entry_bars = {}
        self.entry_prices = {}
        self.stop_prices = {}
        self.orders = {}

        self.peak_equity = None
        self.trading_disabled = False

        for data in self.datas:
            self.zscores[data] = RollingZScore(data.close, period=self.p.lookback)
            self.atrs[data] = bt.indicators.ATR(data, period=self.p.atr_period)
            self.entry_bars[data] = None
            self.entry_prices[data] = None
            self.stop_prices[data] = None
            self.orders[data] = None

    def reset_asset_state(self, data):
        self.entry_bars[data] = None
        self.entry_prices[data] = None
        self.stop_prices[data] = None

    def notify_order(self, order):
        data = order.data
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            position = self.getposition(data)
            atr = float(self.atrs[data][0])

            if position.size != 0:
                if self.entry_prices[data] is None:
                    self.entry_prices[data] = float(position.price)
                    self.entry_bars[data] = len(self)

                    if math.isfinite(atr) and atr > 0:
                        if position.size > 0:
                            self.stop_prices[data] = self.entry_prices[data] - self.p.atr_stop_mult * atr
                        else:
                            self.stop_prices[data] = self.entry_prices[data] + self.p.atr_stop_mult * atr
            else:
                self.reset_asset_state(data)

            self.orders[data] = None

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.orders[data] = None

    def calculate_drawdown(self):
        equity = float(self.broker.getvalue())
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def gross_exposure(self):
        equity = float(self.broker.getvalue())
        if equity <= 0:
            return 0.0

        exposure = 0.0
        for data in self.datas:
            position = self.getposition(data)
            if position.size == 0:
                continue
            price = float(data.close[0])
            if math.isfinite(price) and price > 0:
                exposure += abs(position.size * price)
        return exposure / equity

    def count_positions(self):
        return sum(1 for data in self.datas if self.getposition(data).size != 0)

    def volatility_ok(self, data):
        price = float(data.close[0])
        atr = float(self.atrs[data][0])
        if not all(math.isfinite(x) for x in (price, atr)) or price <= 0 or atr <= 0:
            return False
        return atr / price <= self.p.max_atr_pct

    def calculate_position_size(self, data):
        equity = float(self.broker.getvalue())
        price = float(data.close[0])
        atr = float(self.atrs[data][0])
        if not all(math.isfinite(x) for x in (equity, price, atr)):
            return 0
        if equity <= 0 or price <= 0 or atr <= 0:
            return 0

        risk_budget = equity * self.p.risk_per_trade
        stop_distance = atr * self.p.atr_stop_mult
        risk_size = risk_budget / stop_distance

        max_asset_capital = equity * self.p.max_asset_exposure
        exposure_size = max_asset_capital / price
        return max(int(min(risk_size, exposure_size)), 0)

    def proposed_exposure(self, data, size):
        equity = float(self.broker.getvalue())
        price = float(data.close[0])
        if equity <= 0 or price <= 0 or size <= 0:
            return 0.0
        return abs(size * price) / equity

    def open_long(self, data, size=None):
        if self.orders[data]:
            return False
        size = self.calculate_position_size(data) if size is None else size
        if size <= 0:
            return False
        self.orders[data] = self.buy(data=data, size=size)
        return True

    def open_short(self, data, size=None):
        if self.orders[data] or not self.p.allow_short:
            return False
        size = self.calculate_position_size(data) if size is None else size
        if size <= 0:
            return False
        self.orders[data] = self.sell(data=data, size=size)
        return True

    def close_asset(self, data):
        if self.orders[data] or self.getposition(data).size == 0:
            return
        self.orders[data] = self.close(data=data)

    def close_all_positions(self):
        for data in self.datas:
            if self.getposition(data).size != 0 and not self.orders[data]:
                self.orders[data] = self.close(data=data)

    def holding_period_exceeded(self, data):
        entry_bar = self.entry_bars[data]
        return entry_bar is not None and len(self) - entry_bar >= self.p.max_holding_bars

    def manage_position(self, data):
        position = self.getposition(data)
        if position.size == 0 or self.orders[data]:
            return

        z = float(self.zscores[data][0])
        if not math.isfinite(z):
            return

        if self.holding_period_exceeded(data):
            self.close_asset(data)
            return

        stop = self.stop_prices[data]
        if position.size > 0:
            low = float(data.low[0])
            if stop is not None and math.isfinite(low) and low <= stop:
                self.close_asset(data)
                return
            if z <= -self.p.stop_z:
                self.close_asset(data)
                return
            if z >= -self.p.exit_z:
                self.close_asset(data)
                return

        elif position.size < 0:
            high = float(data.high[0])
            if stop is not None and math.isfinite(high) and high >= stop:
                self.close_asset(data)
                return
            if z >= self.p.stop_z:
                self.close_asset(data)
                return
            if z <= self.p.exit_z:
                self.close_asset(data)
                return

    def get_candidates(self):
        candidates = []
        for data in self.datas:
            if self.getposition(data).size != 0 or self.orders[data]:
                continue

            z = float(self.zscores[data][0])
            if not math.isfinite(z) or not self.volatility_ok(data):
                continue

            if z <= -self.p.entry_z:
                candidates.append((abs(z), data, "long"))
            elif z >= self.p.entry_z and self.p.allow_short:
                candidates.append((abs(z), data, "short"))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def next(self):
        drawdown = self.calculate_drawdown()
        if drawdown >= self.p.max_drawdown:
            self.close_all_positions()
            self.trading_disabled = True
            return
        if self.trading_disabled:
            return

        for data in self.datas:
            self.manage_position(data)

        current_positions = self.count_positions()
        current_gross = self.gross_exposure()
        reserved_positions = 0
        reserved_gross = 0.0

        if current_positions >= self.p.max_positions or current_gross >= self.p.max_gross_exposure:
            return

        for _, data, direction in self.get_candidates():
            if current_positions + reserved_positions >= self.p.max_positions:
                break

            size = self.calculate_position_size(data)
            if size <= 0:
                continue

            new_exposure = self.proposed_exposure(data, size)
            if current_gross + reserved_gross + new_exposure > self.p.max_gross_exposure:
                continue

            submitted = False
            if direction == "long":
                submitted = self.open_long(data, size=size)
            elif direction == "short":
                submitted = self.open_short(data, size=size)

            if submitted:
                reserved_positions += 1
                reserved_gross += new_exposure
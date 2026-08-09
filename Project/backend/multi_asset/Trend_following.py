import backtrader as bt
import math


class MultiAssetTrendFollowingStrategy(bt.Strategy):
    params = (
        ("fast_period", 50),
        ("slow_period", 200),
        ("adx_period", 14),
        ("min_adx", 20.0),
        ("atr_period", 14),
        ("atr_stop_mult", 3.0),
        ("trailing_atr_mult", 3.0),
        ("risk_per_trade", 0.005),
        ("max_asset_exposure", 0.15),
        ("max_gross_exposure", 0.80),
        ("max_positions", 6),
        ("max_drawdown", 0.20),
        ("allow_short", True),
    )

    def __init__(self):
        self.fast_ma = {}
        self.slow_ma = {}
        self.cross = {}
        self.adx = {}
        self.atr = {}

        self.orders = {}
        self.entry_prices = {}
        self.stop_prices = {}
        self.highest_prices = {}
        self.lowest_prices = {}

        self.peak_equity = None
        self.trading_disabled = False

        for data in self.datas:
            self.fast_ma[data] = bt.indicators.EMA(data.close, period=self.p.fast_period)
            self.slow_ma[data] = bt.indicators.EMA(data.close, period=self.p.slow_period)
            self.cross[data] = bt.indicators.CrossOver(self.fast_ma[data], self.slow_ma[data])
            self.adx[data] = bt.indicators.ADX(data, period=self.p.adx_period)
            self.atr[data] = bt.indicators.ATR(data, period=self.p.atr_period)

            self.orders[data] = None
            self.entry_prices[data] = None
            self.stop_prices[data] = None
            self.highest_prices[data] = None
            self.lowest_prices[data] = None

    def reset_asset_state(self, data):
        self.entry_prices[data] = None
        self.stop_prices[data] = None
        self.highest_prices[data] = None
        self.lowest_prices[data] = None

    def notify_order(self, order):
        data = order.data
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            position = self.getposition(data)
            atr = float(self.atr[data][0])

            if position.size > 0:
                self.entry_prices[data] = float(position.price)
                self.highest_prices[data] = self.entry_prices[data]
                self.lowest_prices[data] = None
                if math.isfinite(atr) and atr > 0:
                    self.stop_prices[data] = self.entry_prices[data] - self.p.atr_stop_mult * atr

            elif position.size < 0:
                self.entry_prices[data] = float(position.price)
                self.lowest_prices[data] = self.entry_prices[data]
                self.highest_prices[data] = None
                if math.isfinite(atr) and atr > 0:
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

        gross = 0.0
        for data in self.datas:
            position = self.getposition(data)
            if position.size == 0:
                continue
            price = float(data.close[0])
            if math.isfinite(price) and price > 0:
                gross += abs(position.size * price)
        return gross / equity

    def count_positions(self):
        return sum(1 for data in self.datas if self.getposition(data).size != 0)

    def calculate_position_size(self, data):
        equity = float(self.broker.getvalue())
        price = float(data.close[0])
        atr = float(self.atr[data][0])
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

    def update_long_stop(self, data):
        price = float(data.close[0])
        atr = float(self.atr[data][0])
        if not all(math.isfinite(x) for x in (price, atr)) or atr <= 0:
            return
        current_high = float(data.high[0])
        reference = current_high if math.isfinite(current_high) else price
        self.highest_prices[data] = reference if self.highest_prices[data] is None else max(self.highest_prices[data], reference)
        new_stop = self.highest_prices[data] - self.p.trailing_atr_mult * atr
        self.stop_prices[data] = new_stop if self.stop_prices[data] is None else max(self.stop_prices[data], new_stop)

    def update_short_stop(self, data):
        price = float(data.close[0])
        atr = float(self.atr[data][0])
        if not all(math.isfinite(x) for x in (price, atr)) or atr <= 0:
            return
        current_low = float(data.low[0])
        reference = current_low if math.isfinite(current_low) else price
        self.lowest_prices[data] = reference if self.lowest_prices[data] is None else min(self.lowest_prices[data], reference)
        new_stop = self.lowest_prices[data] + self.p.trailing_atr_mult * atr
        self.stop_prices[data] = new_stop if self.stop_prices[data] is None else min(self.stop_prices[data], new_stop)

    def manage_position(self, data):
        position = self.getposition(data)
        if position.size == 0 or self.orders[data]:
            return

        fast = float(self.fast_ma[data][0])
        slow = float(self.slow_ma[data][0])
        if not all(math.isfinite(x) for x in (fast, slow)):
            return

        if position.size > 0:
            self.update_long_stop(data)
            stop = self.stop_prices[data]
            low = float(data.low[0])
            if stop is not None and math.isfinite(low) and low <= stop:
                self.close_asset(data)
                return
            if fast < slow:
                self.close_asset(data)

        elif position.size < 0:
            self.update_short_stop(data)
            stop = self.stop_prices[data]
            high = float(data.high[0])
            if stop is not None and math.isfinite(high) and high >= stop:
                self.close_asset(data)
                return
            if fast > slow:
                self.close_asset(data)

    def get_candidates(self):
        candidates = []
        for data in self.datas:
            if self.getposition(data).size != 0 or self.orders[data]:
                continue

            values = (
                float(self.fast_ma[data][0]),
                float(self.slow_ma[data][0]),
                float(self.adx[data][0]),
                float(self.atr[data][0]),
                float(data.close[0]),
            )
            if not all(math.isfinite(x) for x in values):
                continue

            fast, slow, adx, atr, price = values
            if atr <= 0 or price <= 0 or adx < self.p.min_adx:
                continue

            trend_distance = abs(fast - slow) / price
            if fast > slow and self.cross[data][0] > 0:
                candidates.append((trend_distance, adx, data, "long"))
            elif fast < slow and self.cross[data][0] < 0 and self.p.allow_short:
                candidates.append((trend_distance, adx, data, "short"))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
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

        for _, _, data, direction in self.get_candidates():
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
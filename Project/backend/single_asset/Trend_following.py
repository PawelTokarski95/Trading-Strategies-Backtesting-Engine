import backtrader as bt
import math


class TrendFollowingStrategy(bt.Strategy):
    params = (
        ("fast_period", 50),
        ("slow_period", 200),
        ("adx_period", 14),
        ("min_adx", 20.0),
        ("atr_period", 14),
        ("atr_stop_mult", 3.0),
        ("risk_per_trade", 0.01),
        ("max_exposure", 0.30),
        ("use_trailing_stop", True),
        ("trailing_atr_mult", 3.0),
        ("max_drawdown", 0.20),
        ("allow_short", True),
    )

    def __init__(self):
        self.price = self.data.close
        self.fast_ma = bt.indicators.EMA(self.price, period=self.p.fast_period)
        self.slow_ma = bt.indicators.EMA(self.price, period=self.p.slow_period)
        self.ma_cross = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.highest_price = None
        self.lowest_price = None

        self.peak_equity = None
        self.trading_disabled = False

    def reset_trade_state(self):
        self.entry_price = None
        self.stop_price = None
        self.highest_price = None
        self.lowest_price = None

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            position = self.position
            atr = float(self.atr[0])

            if position.size > 0:
                self.entry_price = float(position.price)
                self.highest_price = self.entry_price
                self.lowest_price = None
                if math.isfinite(atr) and atr > 0:
                    self.stop_price = self.entry_price - self.p.atr_stop_mult * atr

            elif position.size < 0:
                self.entry_price = float(position.price)
                self.lowest_price = self.entry_price
                self.highest_price = None
                if math.isfinite(atr) and atr > 0:
                    self.stop_price = self.entry_price + self.p.atr_stop_mult * atr

            else:
                self.reset_trade_state()

            self.order = None

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.order = None

    def calculate_drawdown(self):
        equity = float(self.broker.getvalue())
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def calculate_position_size(self):
        equity = float(self.broker.getvalue())
        price = float(self.price[0])
        atr = float(self.atr[0])

        if not all(math.isfinite(x) for x in (equity, price, atr)):
            return 0
        if equity <= 0 or price <= 0 or atr <= 0:
            return 0

        risk_budget = equity * self.p.risk_per_trade
        stop_distance = atr * self.p.atr_stop_mult
        risk_based_size = risk_budget / stop_distance

        max_capital = equity * self.p.max_exposure
        exposure_based_size = max_capital / price

        return max(int(min(risk_based_size, exposure_based_size)), 0)

    def open_long(self):
        if self.order:
            return
        size = self.calculate_position_size()
        if size > 0:
            self.order = self.buy(size=size)

    def open_short(self):
        if self.order or not self.p.allow_short:
            return
        size = self.calculate_position_size()
        if size > 0:
            self.order = self.sell(size=size)

    def close_position(self):
        if self.order or not self.position:
            return
        self.order = self.close()

    def update_long_stop(self):
        price = float(self.price[0])
        atr = float(self.atr[0])
        if not all(math.isfinite(x) for x in (price, atr)) or atr <= 0:
            return

        self.highest_price = price if self.highest_price is None else max(self.highest_price, price)
        if self.p.use_trailing_stop:
            new_stop = self.highest_price - self.p.trailing_atr_mult * atr
            self.stop_price = new_stop if self.stop_price is None else max(self.stop_price, new_stop)

    def update_short_stop(self):
        price = float(self.price[0])
        atr = float(self.atr[0])
        if not all(math.isfinite(x) for x in (price, atr)) or atr <= 0:
            return

        self.lowest_price = price if self.lowest_price is None else min(self.lowest_price, price)
        if self.p.use_trailing_stop:
            new_stop = self.lowest_price + self.p.trailing_atr_mult * atr
            self.stop_price = new_stop if self.stop_price is None else min(self.stop_price, new_stop)

    def next(self):
        if self.order:
            return

        values = (
            float(self.price[0]),
            float(self.fast_ma[0]),
            float(self.slow_ma[0]),
            float(self.adx[0]),
            float(self.atr[0]),
        )
        if not all(math.isfinite(x) for x in values):
            return

        price, fast, slow, adx, atr = values
        if price <= 0 or atr <= 0:
            return

        drawdown = self.calculate_drawdown()
        if drawdown >= self.p.max_drawdown:
            if self.position:
                self.close_position()
            self.trading_disabled = True
            return

        if self.trading_disabled:
            return

        if not self.position:
            if self.ma_cross[0] > 0 and adx >= self.p.min_adx:
                self.open_long()
                return
            if self.ma_cross[0] < 0 and adx >= self.p.min_adx and self.p.allow_short:
                self.open_short()
                return
            return

        if self.position.size > 0:
            self.update_long_stop()
            # Using the bar low catches an intrabar touch; the close order is still
            # executed according to the broker/backtest execution model.
            bar_low = float(self.data.low[0])
            if self.stop_price is not None and math.isfinite(bar_low) and bar_low <= self.stop_price:
                self.close_position()
                return
            if fast < slow:
                self.close_position()
                return

        elif self.position.size < 0:
            self.update_short_stop()
            bar_high = float(self.data.high[0])
            if self.stop_price is not None and math.isfinite(bar_high) and bar_high >= self.stop_price:
                self.close_position()
                return
            if fast > slow:
                self.close_position()
                return
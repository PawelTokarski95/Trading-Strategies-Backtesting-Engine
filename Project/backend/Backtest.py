import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type

import backtrader as bt
import numpy as np
import optuna
import pandas as pd
import yfinance as yf

from Project.backend.single_asset.Mean_reversion import MeanReversionStrategy
from Project.backend.multi_asset.Mean_reversion import MultiAssetMeanReversionStrategy
from Project.backend.single_asset.Trend_following import TrendFollowingStrategy
from Project.backend.multi_asset.Trend_following import MultiAssetTrendFollowingStrategy
from Project.backend.multi_asset.Pairs_trading import PairsTradingStrategy


TRADING_DAYS_PER_YEAR = 252


# ============================================================
# OPTIMIZATION CONFIGURATION
# ============================================================

@dataclass
class OptimizationConfig:


    n_trials: int = 150
    timeout: Optional[int] = None
    n_jobs: int = 1

    n_splits: int = 4
    holdout_fraction: float = 0.20
    initial_train_fraction: float = 0.40
    warmup_bars: int = 400
    min_validation_bars: int = 126
    embargo_bars: int = 5

    min_total_trades: int = 8
    min_trades_per_fold: int = 1
    max_fold_drawdown: float = 0.25

    stability_penalty: float = 0.35
    worst_fold_weight: float = 0.15
    negative_fold_penalty: float = 0.35
    inactive_fold_penalty: float = 0.30

    random_seed: int = 42

    study_name: Optional[str] = None
    storage: Optional[str] = None
    load_if_exists: bool = True
    show_progress_bar: bool = False

    calculate_param_importance: bool = True
    save_directory: Optional[str] = None

    def validate(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("n_trials must be greater than zero.")

        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        if not 0.05 <= self.holdout_fraction <= 0.40:
            raise ValueError(
                "holdout_fraction should be between 0.05 and 0.40."
            )

        if not 0.10 <= self.initial_train_fraction <= 0.80:
            raise ValueError(
                "initial_train_fraction should be between 0.10 and 0.80."
            )

        if self.warmup_bars < 30:
            raise ValueError("warmup_bars must be at least 30.")

        if self.min_validation_bars < 30:
            raise ValueError("min_validation_bars must be at least 30.")

        if self.embargo_bars < 0:
            raise ValueError("embargo_bars cannot be negative.")

        if self.min_total_trades < 0:
            raise ValueError("min_total_trades cannot be negative.")

        if self.min_trades_per_fold < 0:
            raise ValueError("min_trades_per_fold cannot be negative.")

        if not 0 < self.max_fold_drawdown <= 1:
            raise ValueError(
                "max_fold_drawdown must be expressed as a decimal in (0, 1]."
            )

        if self.stability_penalty < 0:
            raise ValueError("stability_penalty cannot be negative.")

        if self.worst_fold_weight < 0:
            raise ValueError("worst_fold_weight cannot be negative.")


@dataclass(frozen=True)
class WalkForwardFold:
    number: int
    data_start: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "data_start": self.data_start.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "validation_bars": self.validation_bars,
        }


@dataclass(frozen=True)
class HoldoutWindow:
    data_start: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    holdout_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_start": self.data_start.isoformat(),
            "holdout_start": self.holdout_start.isoformat(),
            "holdout_end": self.holdout_end.isoformat(),
            "holdout_bars": self.holdout_bars,
        }


# ============================================================
# DATA
# ============================================================

def download_data(
    tickers,
    start,
    end,
):


    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    data_dict = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                print(f"{ticker}: no data")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            if not all(
                column in df.columns
                for column in required_columns
            ):
                print(f"{ticker}: missing required columns")
                continue

            df = df[required_columns].copy()

            df = df.replace(
                [float("inf"), float("-inf")],
                float("nan"),
            )

            df = df.dropna()
            df = df[~df.index.duplicated(keep="first")]
            df = df.sort_index()

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            if len(df) < 100:
                print(
                    f"{ticker}: insufficient history "
                    f"({len(df)} observations)"
                )
                continue

            data_dict[ticker] = df
            print(f"{ticker}: {len(df)} observations")

        except Exception as exc:
            print(f"{ticker}: ERROR -> {exc}")

    return data_dict


def align_datasets(
    datasets: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Align all assets to a common calendar.

    This is used by the optimizer so every fold has the same dates for every
    asset. The normal run_backtest function keeps the original behavior and
    does not force alignment.
    """

    if not datasets:
        raise ValueError("No datasets were supplied.")

    common_index: Optional[pd.DatetimeIndex] = None

    for df in datasets.values():
        index = pd.DatetimeIndex(df.index)

        if index.tz is not None:
            index = index.tz_localize(None)

        common_index = (
            index
            if common_index is None
            else common_index.intersection(index)
        )

    if common_index is None or common_index.empty:
        raise ValueError(
            "The selected assets have no common observation dates."
        )

    common_index = common_index.sort_values()

    aligned = {}

    for ticker, df in datasets.items():
        frame = df.copy()

        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)

        frame = frame.loc[common_index].copy()
        frame = frame.dropna()

        aligned[ticker] = frame

    final_index: Optional[pd.DatetimeIndex] = None

    for frame in aligned.values():
        final_index = (
            frame.index
            if final_index is None
            else final_index.intersection(frame.index)
        )

    if final_index is None or final_index.empty:
        raise ValueError(
            "No common clean observations remain after alignment."
        )

    final_index = final_index.sort_values()

    return {
        ticker: frame.loc[final_index].copy()
        for ticker, frame in aligned.items()
    }


def slice_datasets(
    datasets: Mapping[str, pd.DataFrame],
    start,
    end,
) -> Dict[str, pd.DataFrame]:
    start_timestamp = _naive_timestamp(start)
    end_timestamp = _naive_timestamp(end)

    sliced = {}

    for ticker, df in datasets.items():
        mask = (
            (df.index >= start_timestamp)
            & (df.index <= end_timestamp)
        )

        frame = df.loc[mask].copy()

        if frame.empty:
            raise ValueError(
                f"No observations for {ticker} between "
                f"{start_timestamp.date()} and {end_timestamp.date()}."
            )

        sliced[ticker] = frame

    return sliced


# ============================================================
# EQUITY ANALYZER
# ============================================================

class EquityAnalyzer(bt.Analyzer):

    def start(self):
        self.equity_curve = []

    def next(self):
        date = self.strategy.datetime.datetime()
        equity = self.strategy.broker.getvalue()

        self.equity_curve.append((date, equity))

    def get_analysis(self):
        return self.equity_curve


# ============================================================
# STRATEGY REGISTRY AND VALIDATION
# ============================================================

STRATEGY_REGISTRY: Dict[Tuple[str, str], Type[bt.Strategy]] = {
    (
        "Mean Reversion",
        "single_asset_strategy",
    ): MeanReversionStrategy,
    (
        "Mean Reversion",
        "multi_asset_strategy",
    ): MultiAssetMeanReversionStrategy,
    (
        "Trend Following",
        "single_asset_strategy",
    ): TrendFollowingStrategy,
    (
        "Trend Following",
        "multi_asset_strategy",
    ): MultiAssetTrendFollowingStrategy,
    (
        "Pairs Trading",
        "multi_asset_strategy",
    ): PairsTradingStrategy,
}


def validate_strategy_configuration(
    tickers,
    strategy_name,
    strategy_type,
):
    """Validate strategy/ticker configuration."""

    valid_strategy_types = [
        "single_asset_strategy",
        "multi_asset_strategy",
    ]

    if strategy_type not in valid_strategy_types:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}"
        )

    if not tickers:
        raise ValueError("Select at least one ticker.")

    if strategy_type == "single_asset_strategy":
        if len(tickers) != 1:
            raise ValueError(
                "Single-asset strategy requires exactly one ticker."
            )

        if strategy_name not in [
            "Mean Reversion",
            "Trend Following",
        ]:
            raise ValueError(
                f"{strategy_name} is not available "
                f"as a single-asset strategy."
            )

    if strategy_name == "Pairs Trading":
        if strategy_type != "multi_asset_strategy":
            raise ValueError(
                "Pairs Trading is available only in multi-asset mode."
            )

        if len(tickers) != 2:
            raise ValueError(
                "Pairs Trading requires exactly two tickers."
            )

    if (strategy_name, strategy_type) not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy configuration: "
            f"{strategy_name} / {strategy_type}"
        )


def get_strategy_class(
    strategy_name: str,
    strategy_type: str,
) -> Type[bt.Strategy]:
    try:
        return STRATEGY_REGISTRY[
            (strategy_name, strategy_type)
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unknown strategy configuration: "
            f"{strategy_name} / {strategy_type}"
        ) from exc


def get_supported_strategy_parameters(
    strategy_class: Type[bt.Strategy],
) -> Set[str]:
    """
    Read parameter names from Backtrader's MetaParams object.

    The function makes the optimizer compatible with both the original and
    corrected versions of the strategies. Parameters absent from the actual
    class are not passed to it.
    """

    params = getattr(strategy_class, "params", None)

    if params is None:
        return set()

    try:
        return set(params._getkeys())
    except Exception:
        pass

    try:
        return {
            str(name)
            for name, _ in params._getitems()
        }
    except Exception:
        return set()


def filter_strategy_parameters(
    strategy_class: Type[bt.Strategy],
    parameters: Optional[Mapping[str, Any]],
    strict: bool = False,
) -> Dict[str, Any]:
    if not parameters:
        return {}

    supported = get_supported_strategy_parameters(
        strategy_class
    )

    if not supported:
        return dict(parameters)

    unknown = set(parameters) - supported

    if strict and unknown:
        raise ValueError(
            f"Unsupported parameters for "
            f"{strategy_class.__name__}: {sorted(unknown)}"
        )

    return {
        key: value
        for key, value in parameters.items()
        if key in supported
    }


# ============================================================
# OPTIMIZATION TRADING WINDOW
# ============================================================

_WINDOWED_STRATEGY_CACHE: Dict[
    Type[bt.Strategy],
    Type[bt.Strategy],
] = {}


def _naive_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp


def get_windowed_strategy_class(
    base_strategy: Type[bt.Strategy],
) -> Type[bt.Strategy]:
    """
    Create a strategy subclass that blocks orders during indicator warm-up.

    Indicators still receive historical bars, but the original strategy's
    next method is not called until optimization_trade_start. Consequently,
    every validation window starts with no position and no trades inherited
    from the warm-up period.
    """

    if base_strategy in _WINDOWED_STRATEGY_CACHE:
        return _WINDOWED_STRATEGY_CACHE[base_strategy]

    class WindowedStrategy(base_strategy):
        params = (
            ("optimization_trade_start", None),
        )

        def next(self):
            trade_start = self.p.optimization_trade_start

            if trade_start is not None:
                current_date = _naive_timestamp(
                    self.datas[0].datetime.datetime(0)
                )

                if current_date < _naive_timestamp(
                    trade_start
                ):
                    return

            return super().next()

    WindowedStrategy.__name__ = (
        f"Windowed{base_strategy.__name__}"
    )

    WindowedStrategy.__qualname__ = (
        WindowedStrategy.__name__
    )

    _WINDOWED_STRATEGY_CACHE[
        base_strategy
    ] = WindowedStrategy

    return WindowedStrategy


def add_strategy(
    cerebro,
    strategy_name,
    strategy_type,
    strategy_params: Optional[Mapping[str, Any]] = None,
    trade_start=None,
):
    """Add a selected strategy, optionally with optimized parameters."""

    base_strategy = get_strategy_class(
        strategy_name,
        strategy_type,
    )

    params = filter_strategy_parameters(
        base_strategy,
        strategy_params,
        strict=False,
    )

    if trade_start is None:
        cerebro.addstrategy(
            base_strategy,
            **params,
        )
        return

    windowed_strategy = get_windowed_strategy_class(
        base_strategy
    )

    params["optimization_trade_start"] = (
        _naive_timestamp(trade_start).to_pydatetime()
    )

    cerebro.addstrategy(
        windowed_strategy,
        **params,
    )


# ============================================================
# SAFE VALUES AND METRICS
# ============================================================

def safe_float(
    value,
    default=float("nan"),
):
    """Safely convert a value to a finite float."""

    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except (TypeError, ValueError):
        return default


def _nested_trade_value(
    analysis,
    path: Sequence[str],
    default: float = 0.0,
) -> float:
    current = analysis

    try:
        for key in path:
            try:
                current = current[key]
            except Exception:
                current = getattr(current, key)

        return safe_float(current, default=default)

    except Exception:
        return default


def _build_equity_dataframe(
    strategy,
    start,
    end,
) -> pd.DataFrame:
    equity = strategy.analyzers.equity.get_analysis()

    equity_df = pd.DataFrame(
        equity,
        columns=["date", "equity"],
    )

    if equity_df.empty:
        return pd.DataFrame(
            columns=[
                "equity",
                "returns",
                "peak",
                "drawdown",
            ]
        )

    equity_df["date"] = pd.to_datetime(
        equity_df["date"]
    )

    if getattr(equity_df["date"].dt, "tz", None) is not None:
        equity_df["date"] = (
            equity_df["date"].dt.tz_localize(None)
        )

    equity_df = (
        equity_df
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .set_index("date")
    )

    start_timestamp = _naive_timestamp(start)
    end_timestamp = _naive_timestamp(end)

    equity_df = equity_df.loc[
        (equity_df.index >= start_timestamp)
        & (equity_df.index <= end_timestamp)
    ].copy()

    if equity_df.empty:
        return pd.DataFrame(
            columns=[
                "equity",
                "returns",
                "peak",
                "drawdown",
            ]
        )

    equity_df["returns"] = (
        equity_df["equity"].pct_change()
    )

    equity_df["peak"] = (
        equity_df["equity"].cummax()
    )

    equity_df["drawdown"] = (
        equity_df["equity"]
        / equity_df["peak"]
        - 1.0
    )

    return equity_df


def analyze_results(
    strategy,
    initial_cash,
    start,
    end,
    risk_free_rate: float = 0.0,
):

    if initial_cash <= 0:
        raise ValueError(
            "Initial cash must be greater than zero."
        )

    equity_df = _build_equity_dataframe(
        strategy=strategy,
        start=start,
        end=end,
    )

    if not equity_df.empty:
        final_value = safe_float(
            equity_df["equity"].iloc[-1],
            default=initial_cash,
        )
    else:
        final_value = safe_float(
            strategy.broker.getvalue(),
            default=initial_cash,
        )

    total_return = final_value / initial_cash - 1.0
    net_pnl = final_value - initial_cash

    if len(equity_df) >= 2:
        first_date = equity_df.index[0]
        last_date = equity_df.index[-1]
        years = max(
            (last_date - first_date).days / 365.25,
            1.0 / TRADING_DAYS_PER_YEAR,
        )
    else:
        start_date = _naive_timestamp(start)
        end_date = _naive_timestamp(end)
        years = max(
            (end_date - start_date).days / 365.25,
            1.0 / TRADING_DAYS_PER_YEAR,
        )

    if final_value > 0 and years > 0:
        cagr = (
            final_value / initial_cash
        ) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    returns = (
        equity_df["returns"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        if not equity_df.empty
        else pd.Series(dtype=float)
    )

    volatility = float("nan")
    sharpe = float("nan")
    sortino = float("nan")

    if len(returns) > 1:
        return_std = safe_float(
            returns.std(ddof=1)
        )

        if math.isfinite(return_std) and return_std > 0:
            volatility = (
                return_std
                * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

            daily_risk_free = (
                (1.0 + risk_free_rate)
                ** (1.0 / TRADING_DAYS_PER_YEAR)
                - 1.0
                if risk_free_rate > -1
                else 0.0
            )

            excess_returns = (
                returns - daily_risk_free
            )

            sharpe = (
                excess_returns.mean()
                / return_std
                * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

            downside = np.minimum(
                excess_returns.to_numpy(dtype=float),
                0.0,
            )

            downside_deviation = (
                math.sqrt(float(np.mean(downside ** 2)))
                * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

            annual_excess_return = (
                safe_float(excess_returns.mean(), 0.0)
                * TRADING_DAYS_PER_YEAR
            )

            if downside_deviation > 0:
                sortino = (
                    annual_excess_return
                    / downside_deviation
                )

    if not equity_df.empty:
        max_drawdown_decimal = abs(
            safe_float(
                equity_df["drawdown"].min(),
                default=0.0,
            )
        )
    else:
        max_drawdown_decimal = 0.0

    max_drawdown = max_drawdown_decimal * 100.0

    if (
        math.isfinite(cagr)
        and max_drawdown_decimal > 0
    ):
        calmar = cagr / max_drawdown_decimal
    else:
        calmar = float("nan")

    trade_analysis = (
        strategy.analyzers.trades.get_analysis()
    )

    total_trades = int(
        _nested_trade_value(
            trade_analysis,
            ["total", "closed"],
            default=0.0,
        )
    )

    open_trades = int(
        _nested_trade_value(
            trade_analysis,
            ["total", "open"],
            default=0.0,
        )
    )

    winning_trades = int(
        _nested_trade_value(
            trade_analysis,
            ["won", "total"],
            default=0.0,
        )
    )

    losing_trades = int(
        _nested_trade_value(
            trade_analysis,
            ["lost", "total"],
            default=0.0,
        )
    )

    win_rate = (
        winning_trades / total_trades
        if total_trades > 0
        else float("nan")
    )

    gross_profit = _nested_trade_value(
        trade_analysis,
        ["won", "pnl", "total"],
        default=0.0,
    )

    gross_loss = abs(
        _nested_trade_value(
            trade_analysis,
            ["lost", "pnl", "total"],
            default=0.0,
        )
    )

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    average_trade = _nested_trade_value(
        trade_analysis,
        ["pnl", "net", "average"],
        default=float("nan"),
    )

    average_win = _nested_trade_value(
        trade_analysis,
        ["won", "pnl", "average"],
        default=float("nan"),
    )

    average_loss = _nested_trade_value(
        trade_analysis,
        ["lost", "pnl", "average"],
        default=float("nan"),
    )

    average_holding_bars = _nested_trade_value(
        trade_analysis,
        ["len", "average"],
        default=float("nan"),
    )

    if (
        total_trades > 0
        and math.isfinite(win_rate)
        and math.isfinite(average_win)
        and math.isfinite(average_loss)
    ):
        expectancy = (
            win_rate * average_win
            + (1.0 - win_rate) * average_loss
        )
    else:
        expectancy = float("nan")

    yearly_returns: Dict[Any, float] = {}

    if not equity_df.empty:
        year_end_equity = (
            equity_df["equity"]
            .groupby(equity_df.index.year)
            .last()
        )

        previous_value = initial_cash

        for year, value in year_end_equity.items():
            value = safe_float(value)

            if (
                math.isfinite(value)
                and previous_value > 0
            ):
                yearly_returns[int(year)] = (
                    value / previous_value - 1.0
                )

            previous_value = value

    return {
        "initial_cash": initial_cash,
        "final_value": final_value,
        "net_pnl": net_pnl,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "max_drawdown_decimal": max_drawdown_decimal,
        "calmar": calmar,
        "total_trades": total_trades,
        "open_trades": open_trades,
        "trade_observations": total_trades + open_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "average_win": average_win,
        "average_loss": average_loss,
        "average_holding_bars": average_holding_bars,
        "expectancy": expectancy,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "yearly_returns": yearly_returns,
        "equity_curve": equity_df,
    }


# ============================================================
# BACKTEST EXECUTION
# ============================================================

def _add_analyzers(
    cerebro: bt.Cerebro,
    lightweight: bool = False,
) -> None:
    cerebro.addanalyzer(
        bt.analyzers.TradeAnalyzer,
        _name="trades",
    )

    cerebro.addanalyzer(
        EquityAnalyzer,
        _name="equity",
    )

    if lightweight:
        return

    # Retained for compatibility with the previous project structure.
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        riskfreerate=0.0,
    )

    cerebro.addanalyzer(
        bt.analyzers.DrawDown,
        _name="drawdown",
    )

    cerebro.addanalyzer(
        bt.analyzers.Returns,
        _name="returns",
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name="timereturn",
        timeframe=bt.TimeFrame.Years,
    )


def _run_backtest_on_datasets(
    datasets: Mapping[str, pd.DataFrame],
    ticker_order: Sequence[str],
    strategy_name: str,
    strategy_type: str,
    start,
    end,
    initial_cash: float,
    commission: float,
    slippage: float,
    strategy_params: Optional[Mapping[str, Any]] = None,
    trade_start=None,
    risk_free_rate: float = 0.0,
    lightweight: bool = False,
) -> Dict[str, Any]:
    if initial_cash <= 0:
        raise ValueError(
            "Initial cash must be greater than zero."
        )

    if commission < 0:
        raise ValueError("Commission cannot be negative.")

    if slippage < 0:
        raise ValueError("Slippage cannot be negative.")

    start_timestamp = _naive_timestamp(start)
    end_timestamp = _naive_timestamp(end)

    if start_timestamp >= end_timestamp:
        raise ValueError(
            "Start date must be earlier than end date."
        )

    cerebro = bt.Cerebro(stdstats=False)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=commission
    )

    if slippage > 0:
        cerebro.broker.set_slippage_perc(
            perc=slippage
        )

    added_tickers = []

    for ticker in ticker_order:
        if ticker not in datasets:
            continue

        df = datasets[ticker]

        if df.empty:
            continue

        feed = bt.feeds.PandasData(
            dataname=df.copy()
        )

        cerebro.adddata(
            feed,
            name=ticker,
        )

        added_tickers.append(ticker)

    if not added_tickers:
        raise RuntimeError(
            "No valid datasets were added to Cerebro."
        )

    add_strategy(
        cerebro=cerebro,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        strategy_params=strategy_params,
        trade_start=trade_start,
    )

    _add_analyzers(
        cerebro,
        lightweight=lightweight,
    )

    backtest_results = cerebro.run(
        runonce=True,
        preload=True,
    )

    if not backtest_results:
        raise RuntimeError(
            "Backtrader returned no strategy results."
        )

    strategy_result = backtest_results[0]

    results = analyze_results(
        strategy=strategy_result,
        initial_cash=initial_cash,
        start=start_timestamp,
        end=end_timestamp,
        risk_free_rate=risk_free_rate,
    )

    base_strategy = get_strategy_class(
        strategy_name,
        strategy_type,
    )

    filtered_params = filter_strategy_parameters(
        base_strategy,
        strategy_params,
        strict=False,
    )

    results.update(
        {
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "strategy_params": filtered_params,
            "tickers": added_tickers,
            "start": start_timestamp,
            "end": end_timestamp,
            "commission": commission,
            "slippage": slippage,
        }
    )

    return results


def run_backtest(
    tickers,
    strategy_name,
    strategy_type,
    start,
    end,
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
    strategy_params: Optional[Mapping[str, Any]] = None,
    risk_free_rate: float = 0.0,
):


    start_timestamp = _naive_timestamp(start)
    end_timestamp = _naive_timestamp(end)

    if start_timestamp >= end_timestamp:
        raise ValueError(
            "Start date must be earlier than end date."
        )

    validate_strategy_configuration(
        tickers=tickers,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
    )

    datasets = download_data(
        tickers=tickers,
        start=start_timestamp,
        end=end_timestamp,
    )

    if not datasets:
        raise RuntimeError(
            "No valid datasets were downloaded."
        )

    if (
        strategy_type == "single_asset_strategy"
        and len(datasets) != 1
    ):
        raise RuntimeError(
            "Could not download data for the selected ticker."
        )

    if (
        strategy_name == "Pairs Trading"
        and len(datasets) != 2
    ):
        raise RuntimeError(
            "Pairs Trading requires valid data for both tickers."
        )

    return _run_backtest_on_datasets(
        datasets=datasets,
        ticker_order=tickers,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        start=start_timestamp,
        end=end_timestamp,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        strategy_params=strategy_params,
        trade_start=None,
        risk_free_rate=risk_free_rate,
        lightweight=False,
    )


def build_walk_forward_plan(
    common_index: pd.DatetimeIndex,
    config: OptimizationConfig,
) -> Tuple[List[WalkForwardFold], HoldoutWindow]:

    config.validate()

    index = pd.DatetimeIndex(common_index).sort_values().unique()

    if index.tz is not None:
        index = index.tz_localize(None)

    total_bars = len(index)

    holdout_bars = max(
        config.min_validation_bars,
        int(round(total_bars * config.holdout_fraction)),
    )

    holdout_start_position = total_bars - holdout_bars
    development_end_exclusive = (
        holdout_start_position - config.embargo_bars
    )

    if development_end_exclusive <= 0:
        raise ValueError(
            "Not enough observations after reserving holdout and embargo."
        )

    development_bars = development_end_exclusive

    initial_train_bars = max(
        config.warmup_bars,
        int(round(
            development_bars
            * config.initial_train_fraction
        )),
    )

    validation_pool = (
        development_bars - initial_train_bars
    )

    possible_splits = (
        validation_pool
        // config.min_validation_bars
    )

    actual_splits = min(
        config.n_splits,
        possible_splits,
    )

    if actual_splits < 2:
        required = (
            config.warmup_bars
            + 2 * config.min_validation_bars
            + holdout_bars
            + config.embargo_bars
        )

        raise ValueError(
            "Insufficient common history for robust optimization. "
            f"Available bars: {total_bars}; approximately required: "
            f"{required}. Reduce warmup_bars/min_validation_bars, "
            "use fewer folds, or extend the date range."
        )

    base_size = validation_pool // actual_splits
    remainder = validation_pool % actual_splits

    folds: List[WalkForwardFold] = []
    validation_start_position = initial_train_bars

    for fold_number in range(actual_splits):
        fold_size = base_size + (
            1 if fold_number < remainder else 0
        )

        validation_end_position = (
            validation_start_position
            + fold_size
            - 1
        )

        data_start_position = max(
            0,
            validation_start_position
            - config.warmup_bars,
        )

        folds.append(
            WalkForwardFold(
                number=fold_number + 1,
                data_start=_naive_timestamp(
                    index[data_start_position]
                ),
                validation_start=_naive_timestamp(
                    index[validation_start_position]
                ),
                validation_end=_naive_timestamp(
                    index[validation_end_position]
                ),
                validation_bars=fold_size,
            )
        )

        validation_start_position = (
            validation_end_position + 1
        )

    holdout_data_start_position = max(
        0,
        holdout_start_position - config.warmup_bars,
    )

    holdout = HoldoutWindow(
        data_start=_naive_timestamp(
            index[holdout_data_start_position]
        ),
        holdout_start=_naive_timestamp(
            index[holdout_start_position]
        ),
        holdout_end=_naive_timestamp(index[-1]),
        holdout_bars=holdout_bars,
    )

    return folds, holdout



def _can_optimize(
    parameter: str,
    supported: Set[str],
    fixed_keys: Set[str],
) -> bool:
    return (
        parameter not in fixed_keys
        and (
            not supported
            or parameter in supported
        )
    )


def suggest_strategy_parameters(
    trial: optuna.Trial,
    strategy_name: str,
    strategy_type: str,
    strategy_class: Type[bt.Strategy],
    number_of_assets: int,
    fixed_strategy_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:

    supported = get_supported_strategy_parameters(
        strategy_class
    )

    fixed = dict(fixed_strategy_params or {})
    fixed_keys = set(fixed)
    params: Dict[str, Any] = {}

    def can(name: str) -> bool:
        return _can_optimize(
            name,
            supported,
            fixed_keys,
        )



    if (
        strategy_name == "Trend Following"
        and strategy_type == "single_asset_strategy"
    ):
        if can("fast_period"):
            params["fast_period"] = trial.suggest_int(
                "fast_period",
                10,
                100,
                step=5,
            )

        if can("slow_period"):
            params["slow_period"] = trial.suggest_int(
                "slow_period",
                80,
                300,
                step=10,
            )

        fast = params.get(
            "fast_period",
            fixed.get("fast_period", 50),
        )
        slow = params.get(
            "slow_period",
            fixed.get("slow_period", 200),
        )

        if slow <= fast + 20:
            raise optuna.TrialPruned(
                "slow_period must exceed fast_period by at least 20 bars."
            )

        if can("adx_period"):
            params["adx_period"] = trial.suggest_int(
                "adx_period",
                7,
                30,
            )

        if can("min_adx"):
            params["min_adx"] = trial.suggest_float(
                "min_adx",
                10.0,
                35.0,
                step=2.5,
            )

        if can("atr_period"):
            params["atr_period"] = trial.suggest_int(
                "atr_period",
                7,
                30,
            )

        if can("atr_stop_mult"):
            params["atr_stop_mult"] = trial.suggest_float(
                "atr_stop_mult",
                1.5,
                5.0,
                step=0.25,
            )

        if can("trailing_atr_mult"):
            params["trailing_atr_mult"] = trial.suggest_float(
                "trailing_atr_mult",
                1.5,
                5.0,
                step=0.25,
            )

        if can("risk_per_trade"):
            params["risk_per_trade"] = trial.suggest_float(
                "risk_per_trade",
                0.0025,
                0.0200,
                log=True,
            )

        if can("max_exposure"):
            params["max_exposure"] = trial.suggest_float(
                "max_exposure",
                0.10,
                0.60,
                step=0.05,
            )



    elif (
        strategy_name == "Trend Following"
        and strategy_type == "multi_asset_strategy"
    ):
        if can("fast_period"):
            params["fast_period"] = trial.suggest_int(
                "fast_period",
                10,
                100,
                step=5,
            )

        if can("slow_period"):
            params["slow_period"] = trial.suggest_int(
                "slow_period",
                80,
                300,
                step=10,
            )

        fast = params.get(
            "fast_period",
            fixed.get("fast_period", 50),
        )
        slow = params.get(
            "slow_period",
            fixed.get("slow_period", 200),
        )

        if slow <= fast + 20:
            raise optuna.TrialPruned(
                "slow_period must exceed fast_period by at least 20 bars."
            )

        if can("adx_period"):
            params["adx_period"] = trial.suggest_int(
                "adx_period",
                7,
                30,
            )

        if can("min_adx"):
            params["min_adx"] = trial.suggest_float(
                "min_adx",
                10.0,
                35.0,
                step=2.5,
            )

        if can("atr_period"):
            params["atr_period"] = trial.suggest_int(
                "atr_period",
                7,
                30,
            )

        if can("atr_stop_mult"):
            params["atr_stop_mult"] = trial.suggest_float(
                "atr_stop_mult",
                1.5,
                5.0,
                step=0.25,
            )

        if can("trailing_atr_mult"):
            params["trailing_atr_mult"] = trial.suggest_float(
                "trailing_atr_mult",
                1.5,
                5.0,
                step=0.25,
            )

        if can("risk_per_trade"):
            params["risk_per_trade"] = trial.suggest_float(
                "risk_per_trade",
                0.0025,
                0.0150,
                log=True,
            )

        if can("max_asset_exposure"):
            params["max_asset_exposure"] = trial.suggest_float(
                "max_asset_exposure",
                0.05,
                0.30,
                step=0.025,
            )

        if can("max_gross_exposure"):
            params["max_gross_exposure"] = trial.suggest_float(
                "max_gross_exposure",
                0.30,
                1.00,
                step=0.05,
            )

        asset_exposure = params.get(
            "max_asset_exposure",
            fixed.get("max_asset_exposure", 0.15),
        )
        gross_exposure = params.get(
            "max_gross_exposure",
            fixed.get("max_gross_exposure", 0.80),
        )

        if gross_exposure < asset_exposure:
            raise optuna.TrialPruned(
                "max_gross_exposure cannot be lower than "
                "max_asset_exposure."
            )

        if can("max_positions"):
            params["max_positions"] = trial.suggest_int(
                "max_positions",
                1,
                max(1, min(12, number_of_assets)),
            )


    elif (
        strategy_name == "Mean Reversion"
        and strategy_type == "single_asset_strategy"
    ):
        if can("lookback"):
            params["lookback"] = trial.suggest_int(
                "lookback",
                20,
                150,
                step=5,
            )

        if can("entry_z"):
            params["entry_z"] = trial.suggest_float(
                "entry_z",
                1.25,
                3.50,
                step=0.25,
            )

        if can("exit_z"):
            params["exit_z"] = trial.suggest_float(
                "exit_z",
                0.00,
                1.25,
                step=0.25,
            )

        if can("stop_z"):
            params["stop_z"] = trial.suggest_float(
                "stop_z",
                2.50,
                6.00,
                step=0.25,
            )

        entry_z = params.get(
            "entry_z",
            fixed.get("entry_z", 2.0),
        )
        exit_z = params.get(
            "exit_z",
            fixed.get("exit_z", 0.5),
        )
        stop_z = params.get(
            "stop_z",
            fixed.get("stop_z", 3.5),
        )

        if exit_z >= entry_z:
            raise optuna.TrialPruned(
                "exit_z must be lower than entry_z."
            )

        if stop_z <= entry_z + 0.50:
            raise optuna.TrialPruned(
                "stop_z must be at least 0.50 above entry_z."
            )

        if can("atr_period"):
            params["atr_period"] = trial.suggest_int(
                "atr_period",
                7,
                30,
            )

        if can("max_atr_pct"):
            params["max_atr_pct"] = trial.suggest_float(
                "max_atr_pct",
                0.03,
                0.15,
                step=0.01,
            )

        if can("atr_stop_mult"):
            params["atr_stop_mult"] = trial.suggest_float(
                "atr_stop_mult",
                1.0,
                4.0,
                step=0.25,
            )

        if can("risk_per_trade"):
            params["risk_per_trade"] = trial.suggest_float(
                "risk_per_trade",
                0.0025,
                0.0200,
                log=True,
            )

        if can("max_exposure"):
            params["max_exposure"] = trial.suggest_float(
                "max_exposure",
                0.10,
                0.60,
                step=0.05,
            )

        if can("max_holding_bars"):
            params["max_holding_bars"] = trial.suggest_int(
                "max_holding_bars",
                10,
                120,
                step=5,
            )


    elif (
        strategy_name == "Mean Reversion"
        and strategy_type == "multi_asset_strategy"
    ):
        if can("lookback"):
            params["lookback"] = trial.suggest_int(
                "lookback",
                20,
                150,
                step=5,
            )

        if can("entry_z"):
            params["entry_z"] = trial.suggest_float(
                "entry_z",
                1.25,
                3.50,
                step=0.25,
            )

        if can("exit_z"):
            params["exit_z"] = trial.suggest_float(
                "exit_z",
                0.00,
                1.25,
                step=0.25,
            )

        if can("stop_z"):
            params["stop_z"] = trial.suggest_float(
                "stop_z",
                2.50,
                6.00,
                step=0.25,
            )

        entry_z = params.get(
            "entry_z",
            fixed.get("entry_z", 2.0),
        )
        exit_z = params.get(
            "exit_z",
            fixed.get("exit_z", 0.5),
        )
        stop_z = params.get(
            "stop_z",
            fixed.get("stop_z", 3.5),
        )

        if exit_z >= entry_z:
            raise optuna.TrialPruned(
                "exit_z must be lower than entry_z."
            )

        if stop_z <= entry_z + 0.50:
            raise optuna.TrialPruned(
                "stop_z must be at least 0.50 above entry_z."
            )

        if can("atr_period"):
            params["atr_period"] = trial.suggest_int(
                "atr_period",
                7,
                30,
            )

        if can("max_atr_pct"):
            params["max_atr_pct"] = trial.suggest_float(
                "max_atr_pct",
                0.03,
                0.15,
                step=0.01,
            )

        if can("atr_stop_mult"):
            params["atr_stop_mult"] = trial.suggest_float(
                "atr_stop_mult",
                1.0,
                4.0,
                step=0.25,
            )

        if can("risk_per_trade"):
            params["risk_per_trade"] = trial.suggest_float(
                "risk_per_trade",
                0.0025,
                0.0150,
                log=True,
            )

        if can("max_asset_exposure"):
            params["max_asset_exposure"] = trial.suggest_float(
                "max_asset_exposure",
                0.05,
                0.30,
                step=0.025,
            )

        if can("max_gross_exposure"):
            params["max_gross_exposure"] = trial.suggest_float(
                "max_gross_exposure",
                0.25,
                1.00,
                step=0.05,
            )

        asset_exposure = params.get(
            "max_asset_exposure",
            fixed.get("max_asset_exposure", 0.15),
        )
        gross_exposure = params.get(
            "max_gross_exposure",
            fixed.get("max_gross_exposure", 0.60),
        )

        if gross_exposure < asset_exposure:
            raise optuna.TrialPruned(
                "max_gross_exposure cannot be lower than "
                "max_asset_exposure."
            )

        if can("max_positions"):
            params["max_positions"] = trial.suggest_int(
                "max_positions",
                1,
                max(1, min(12, number_of_assets)),
            )

        if can("max_holding_bars"):
            params["max_holding_bars"] = trial.suggest_int(
                "max_holding_bars",
                10,
                120,
                step=5,
            )


    elif (
        strategy_name == "Pairs Trading"
        and strategy_type == "multi_asset_strategy"
    ):
        if can("lookback"):
            params["lookback"] = trial.suggest_int(
                "lookback",
                30,
                180,
                step=10,
            )

        if can("entry_z"):
            params["entry_z"] = trial.suggest_float(
                "entry_z",
                1.25,
                3.50,
                step=0.25,
            )

        if can("exit_z"):
            params["exit_z"] = trial.suggest_float(
                "exit_z",
                0.00,
                1.25,
                step=0.25,
            )

        if can("stop_z"):
            params["stop_z"] = trial.suggest_float(
                "stop_z",
                2.50,
                6.00,
                step=0.25,
            )

        entry_z = params.get(
            "entry_z",
            fixed.get("entry_z", 2.0),
        )
        exit_z = params.get(
            "exit_z",
            fixed.get("exit_z", 0.5),
        )
        stop_z = params.get(
            "stop_z",
            fixed.get("stop_z", 4.0),
        )

        if exit_z >= entry_z:
            raise optuna.TrialPruned(
                "exit_z must be lower than entry_z."
            )

        if stop_z <= entry_z + 0.50:
            raise optuna.TrialPruned(
                "stop_z must be at least 0.50 above entry_z."
            )

        if can("gross_exposure"):
            params["gross_exposure"] = trial.suggest_float(
                "gross_exposure",
                0.05,
                0.60,
                step=0.05,
            )

        if can("risk_per_trade"):
            params["risk_per_trade"] = trial.suggest_float(
                "risk_per_trade",
                0.0025,
                0.0300,
                log=True,
            )

        if can("max_holding_bars"):
            params["max_holding_bars"] = trial.suggest_int(
                "max_holding_bars",
                20,
                180,
                step=10,
            )

        if can("min_hedge_ratio"):
            params["min_hedge_ratio"] = trial.suggest_float(
                "min_hedge_ratio",
                0.05,
                0.50,
                step=0.05,
            )

        if can("max_hedge_ratio"):
            params["max_hedge_ratio"] = trial.suggest_float(
                "max_hedge_ratio",
                2.0,
                8.0,
                step=0.5,
            )

        min_hedge = params.get(
            "min_hedge_ratio",
            fixed.get("min_hedge_ratio", 0.10),
        )
        max_hedge = params.get(
            "max_hedge_ratio",
            fixed.get("max_hedge_ratio", 5.00),
        )

        if min_hedge >= max_hedge:
            raise optuna.TrialPruned(
                "min_hedge_ratio must be below max_hedge_ratio."
            )

        if can("cointegration_period"):
            params["cointegration_period"] = trial.suggest_int(
                "cointegration_period",
                60,
                252,
                step=12,
            )

        lookback = params.get(
            "lookback",
            fixed.get("lookback", 60),
        )
        coint_period = params.get(
            "cointegration_period",
            fixed.get("cointegration_period", 120),
        )

        if (
            "cointegration_period" in params
            or "cointegration_period" in fixed
        ) and coint_period < lookback:
            raise optuna.TrialPruned(
                "cointegration_period cannot be below lookback."
            )

        if can("max_coint_pvalue"):
            params["max_coint_pvalue"] = trial.suggest_float(
                "max_coint_pvalue",
                0.01,
                0.10,
                step=0.01,
            )

        if (
            (not supported or "require_cointegration" in supported)
            and "require_cointegration" not in fixed
        ):
            params["require_cointegration"] = True

    else:
        raise ValueError(
            f"No search space for "
            f"{strategy_name} / {strategy_type}."
        )

    params.update(fixed)

    return filter_strategy_parameters(
        strategy_class,
        params,
        strict=False,
    )


def _finite_or_zero(value) -> float:
    result = safe_float(value, default=0.0)
    return result if math.isfinite(result) else 0.0


def calculate_fold_score(
    results: Mapping[str, Any],
) -> float:


    sharpe = float(np.clip(
        _finite_or_zero(results.get("sharpe")),
        -3.0,
        3.0,
    ))

    sortino = float(np.clip(
        _finite_or_zero(results.get("sortino")),
        -5.0,
        5.0,
    ))

    calmar = float(np.clip(
        _finite_or_zero(results.get("calmar")),
        -5.0,
        5.0,
    ))

    cagr = _finite_or_zero(
        results.get("cagr")
    )

    total_return = _finite_or_zero(
        results.get("total_return")
    )

    max_drawdown = max(
        0.0,
        _finite_or_zero(
            results.get("max_drawdown_decimal")
        ),
    )

    profit_factor = safe_float(
        results.get("profit_factor"),
        default=1.0,
    )

    if not math.isfinite(profit_factor):
        profit_factor = 4.0

    profit_factor = float(np.clip(
        profit_factor,
        0.0,
        4.0,
    ))

    return_component = math.tanh(
        cagr / 0.15
    )

    period_return_component = math.tanh(
        total_return / 0.10
    )

    profit_factor_component = math.tanh(
        (profit_factor - 1.0) / 1.0
    )

    score = (
        0.30 * sharpe
        + 0.15 * sortino
        + 0.15 * calmar
        + 0.18 * return_component
        + 0.10 * period_return_component
        + 0.12 * profit_factor_component
        - 0.50 * max_drawdown
    )

    return float(score)


def _activity_trade_count(
    result: Mapping[str, Any],
) -> int:
    return int(
        result.get(
            "trade_observations",
            int(result.get("total_trades", 0))
            + int(result.get("open_trades", 0)),
        )
    )


def aggregate_fold_scores(
    fold_scores: Sequence[float],
    fold_results: Sequence[Mapping[str, Any]],
    config: OptimizationConfig,
) -> float:
    if not fold_scores:
        return -1_000_000.0

    scores = np.asarray(
        fold_scores,
        dtype=float,
    )

    mean_score = float(np.mean(scores))
    score_std = float(np.std(scores, ddof=0))
    worst_score = float(np.min(scores))

    total_trades = sum(
        _activity_trade_count(result)
        for result in fold_results
    )

    negative_fold_fraction = float(np.mean([
        _finite_or_zero(
            result.get("total_return")
        ) <= 0
        for result in fold_results
    ]))

    inactive_fold_fraction = float(np.mean([
        _activity_trade_count(result)
        < config.min_trades_per_fold
        for result in fold_results
    ]))

    robust_score = (
        mean_score
        - config.stability_penalty * score_std
        + config.worst_fold_weight * worst_score
        - config.negative_fold_penalty
        * negative_fold_fraction
        - config.inactive_fold_penalty
        * inactive_fold_fraction
    )

    if (
        config.min_total_trades > 0
        and total_trades < config.min_total_trades
    ):
        trade_shortfall = (
            config.min_total_trades - total_trades
        ) / config.min_total_trades

        robust_score -= 1.50 * trade_shortfall

    drawdown_excesses = []

    for result in fold_results:
        drawdown = max(
            0.0,
            _finite_or_zero(
                result.get("max_drawdown_decimal")
            ),
        )

        drawdown_excesses.append(
            max(
                0.0,
                drawdown - config.max_fold_drawdown,
            )
        )

    robust_score -= 4.0 * float(
        np.mean(drawdown_excesses)
    )

    return float(robust_score)


def _serializable_metrics(
    results: Mapping[str, Any],
) -> Dict[str, Any]:
    excluded = {
        "equity_curve",
        "yearly_returns",
    }

    serializable = {}

    for key, value in results.items():
        if key in excluded:
            continue

        if isinstance(value, pd.Timestamp):
            serializable[key] = value.isoformat()

        elif isinstance(value, (np.integer, int)):
            serializable[key] = int(value)

        elif isinstance(value, (np.floating, float)):
            number = float(value)
            serializable[key] = (
                number if math.isfinite(number) else None
            )

        elif isinstance(value, (str, bool)) or value is None:
            serializable[key] = value

        elif isinstance(value, Mapping):
            serializable[key] = _json_safe(value)

        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            serializable[key] = _json_safe(value)

    return serializable


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    return value


def optimization_trials_dataframe(
    study: optuna.Study,
) -> pd.DataFrame:
    rows = []

    for trial in study.trials:
        row: Dict[str, Any] = {
            "trial": trial.number,
            "state": trial.state.name,
            "objective": trial.value,
        }

        for name, value in trial.params.items():
            row[f"param_{name}"] = value

        for name in [
            "mean_fold_score",
            "std_fold_score",
            "worst_fold_score",
            "total_cv_trades",
            "negative_fold_fraction",
            "inactive_fold_fraction",
            "error",
        ]:
            if name in trial.user_attrs:
                row[f"cv_{name}"] = (
                    trial.user_attrs[name]
                )

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        dataframe["_complete_rank"] = (
            dataframe["state"] != "COMPLETE"
        ).astype(int)

        dataframe = (
            dataframe
            .sort_values(
                ["_complete_rank", "objective"],
                ascending=[True, False],
                na_position="last",
            )
            .drop(columns="_complete_rank")
            .reset_index(drop=True)
        )

    return dataframe


def _default_study_name(
    strategy_name: str,
    strategy_type: str,
    tickers: Sequence[str],
    start,
    end,
    config: OptimizationConfig,
    initial_cash: float,
    commission: float,
    slippage: float,
    risk_free_rate: float,
    fixed_strategy_params: Mapping[str, Any],
    strategy_code_signature: str,
) -> Tuple[str, str]:
    signature_payload = {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "tickers": list(tickers),
        "start": _naive_timestamp(start).isoformat(),
        "end": _naive_timestamp(end).isoformat(),
        "n_splits": config.n_splits,
        "holdout_fraction": config.holdout_fraction,
        "initial_train_fraction": (
            config.initial_train_fraction
        ),
        "warmup_bars": config.warmup_bars,
        "min_validation_bars": (
            config.min_validation_bars
        ),
        "embargo_bars": config.embargo_bars,
        "min_total_trades": config.min_total_trades,
        "min_trades_per_fold": config.min_trades_per_fold,
        "max_fold_drawdown": config.max_fold_drawdown,
        "stability_penalty": config.stability_penalty,
        "worst_fold_weight": config.worst_fold_weight,
        "negative_fold_penalty": config.negative_fold_penalty,
        "inactive_fold_penalty": config.inactive_fold_penalty,
        "initial_cash": initial_cash,
        "commission": commission,
        "slippage": slippage,
        "risk_free_rate": risk_free_rate,
        "fixed_strategy_params": _json_safe(
            fixed_strategy_params
        ),
        "strategy_code_signature": strategy_code_signature,
        "optimizer_schema_version": 2,
    }

    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    slug = (
        f"{strategy_name}_{strategy_type}"
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"{slug}_{signature[:12]}", signature


def save_optimization_outputs(
    optimization_result: Mapping[str, Any],
    directory,
) -> Dict[str, str]:
    output_directory = Path(directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    study_name = str(
        optimization_result["study_name"]
    )

    trials_path = (
        output_directory
        / f"{study_name}_trials.csv"
    )

    best_path = (
        output_directory
        / f"{study_name}_best.json"
    )

    optimization_result["trials"].to_csv(
        trials_path,
        index=False,
    )

    holdout_results = dict(
        optimization_result["holdout_results"]
    )

    holdout_results.pop("equity_curve", None)

    payload = {
        "study_name": study_name,
        "best_trial_number": (
            optimization_result[
                "best_trial_number"
            ]
        ),
        "best_objective": (
            optimization_result[
                "best_objective"
            ]
        ),
        "best_params": (
            optimization_result["best_params"]
        ),
        "cv_summary": (
            optimization_result["cv_summary"]
        ),
        "holdout_window": (
            optimization_result["holdout_window"]
        ),
        "holdout_results": _json_safe(
            holdout_results
        ),
        "parameter_importance": (
            optimization_result[
                "parameter_importance"
            ]
        ),
        "optimization_config": (
            optimization_result[
                "optimization_config"
            ]
        ),
    }

    best_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "trials_csv": str(trials_path),
        "best_json": str(best_path),
    }



def optimize_hyperparameters(
    tickers,
    strategy_name,
    strategy_type,
    start,
    end,
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
    risk_free_rate: float = 0.0,
    optimization_config: Optional[OptimizationConfig] = None,
    fixed_strategy_params: Optional[Mapping[str, Any]] = None,
):

    config = (
        optimization_config
        if optimization_config is not None
        else OptimizationConfig()
    )

    config.validate()

    start_timestamp = _naive_timestamp(start)
    end_timestamp = _naive_timestamp(end)

    if start_timestamp >= end_timestamp:
        raise ValueError(
            "Start date must be earlier than end date."
        )

    if initial_cash <= 0:
        raise ValueError(
            "Initial cash must be greater than zero."
        )

    if commission < 0:
        raise ValueError("Commission cannot be negative.")

    if slippage < 0:
        raise ValueError("Slippage cannot be negative.")

    validate_strategy_configuration(
        tickers=tickers,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
    )

    downloaded = download_data(
        tickers=tickers,
        start=start_timestamp,
        end=end_timestamp,
    )

    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in downloaded
    ]

    if missing_tickers:
        raise RuntimeError(
            "Optimization requires valid data for every selected asset. "
            f"Missing: {missing_tickers}"
        )

    datasets = align_datasets(
        {
            ticker: downloaded[ticker]
            for ticker in tickers
        }
    )

    common_index = next(
        iter(datasets.values())
    ).index

    folds, holdout = build_walk_forward_plan(
        common_index=common_index,
        config=config,
    )

    strategy_class = get_strategy_class(
        strategy_name,
        strategy_type,
    )

    fixed_params = filter_strategy_parameters(
        strategy_class,
        fixed_strategy_params,
        strict=True,
    )

    try:
        strategy_source = inspect.getsource(
            strategy_class
        )
    except Exception:
        strategy_source = (
            f"{strategy_class.__module__}."
            f"{strategy_class.__qualname__}"
        )

    strategy_code_signature = hashlib.sha256(
        strategy_source.encode("utf-8")
    ).hexdigest()

    default_study_name, signature = (
        _default_study_name(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            tickers=tickers,
            start=start_timestamp,
            end=end_timestamp,
            config=config,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            risk_free_rate=risk_free_rate,
            fixed_strategy_params=fixed_params,
            strategy_code_signature=strategy_code_signature,
        )
    )

    study_name = (
        config.study_name
        or default_study_name
    )

    startup_trials = max(
        10,
        min(30, config.n_trials // 5),
    )

    sampler = optuna.samplers.TPESampler(
        seed=config.random_seed,
        n_startup_trials=startup_trials,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=startup_trials,
        n_warmup_steps=1,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=config.storage,
        load_if_exists=config.load_if_exists,
    )

    existing_signature = study.user_attrs.get(
        "optimization_signature"
    )

    if existing_signature is None and study.trials:
        raise ValueError(
            "The existing Optuna study has no compatibility signature. "
            "Use another study_name to avoid mixing trials from a different "
            "optimizer or strategy version."
        )

    if (
        existing_signature is not None
        and existing_signature != signature
    ):
        raise ValueError(
            "The existing Optuna study has a different data/split "
            "signature. Use another study_name or delete the old study."
        )

    study.set_user_attr(
        "optimization_signature",
        signature,
    )

    study.set_user_attr(
        "strategy_name",
        strategy_name,
    )

    study.set_user_attr(
        "strategy_type",
        strategy_type,
    )

    study.set_user_attr(
        "tickers",
        list(tickers),
    )

    def objective(trial: optuna.Trial) -> float:
        strategy_params = suggest_strategy_parameters(
            trial=trial,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            strategy_class=strategy_class,
            number_of_assets=len(tickers),
            fixed_strategy_params=fixed_params,
        )

        trial.set_user_attr(
            "strategy_params",
            _json_safe(strategy_params),
        )

        fold_scores: List[float] = []
        fold_results: List[Dict[str, Any]] = []

        for step, fold in enumerate(folds):
            fold_datasets = slice_datasets(
                datasets=datasets,
                start=fold.data_start,
                end=fold.validation_end,
            )

            try:
                result = _run_backtest_on_datasets(
                    datasets=fold_datasets,
                    ticker_order=tickers,
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    start=fold.validation_start,
                    end=fold.validation_end,
                    initial_cash=initial_cash,
                    commission=commission,
                    slippage=slippage,
                    strategy_params=strategy_params,
                    trade_start=fold.validation_start,
                    risk_free_rate=risk_free_rate,
                    lightweight=True,
                )

            except optuna.TrialPruned:
                raise

            except Exception as exc:
                trial.set_user_attr(
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
                raise

            fold_score = calculate_fold_score(
                result
            )

            fold_scores.append(fold_score)
            fold_results.append(result)

            running_objective = aggregate_fold_scores(
                fold_scores=fold_scores,
                fold_results=fold_results,
                config=config,
            )

            trial.report(
                running_objective,
                step=step,
            )

            if step >= 1 and trial.should_prune():
                raise optuna.TrialPruned(
                    "Trial pruned after walk-forward fold."
                )

        final_objective = aggregate_fold_scores(
            fold_scores=fold_scores,
            fold_results=fold_results,
            config=config,
        )

        scores_array = np.asarray(
            fold_scores,
            dtype=float,
        )

        total_cv_trades = sum(
            _activity_trade_count(result)
            for result in fold_results
        )

        negative_fold_fraction = float(np.mean([
            _finite_or_zero(
                result.get("total_return")
            ) <= 0
            for result in fold_results
        ]))

        inactive_fold_fraction = float(np.mean([
            _activity_trade_count(result)
            < config.min_trades_per_fold
            for result in fold_results
        ]))

        trial.set_user_attr(
            "fold_scores",
            _json_safe(fold_scores),
        )

        trial.set_user_attr(
            "fold_metrics",
            [
                _serializable_metrics(result)
                for result in fold_results
            ],
        )

        trial.set_user_attr(
            "mean_fold_score",
            float(np.mean(scores_array)),
        )

        trial.set_user_attr(
            "std_fold_score",
            float(np.std(scores_array, ddof=0)),
        )

        trial.set_user_attr(
            "worst_fold_score",
            float(np.min(scores_array)),
        )

        trial.set_user_attr(
            "total_cv_trades",
            int(total_cv_trades),
        )

        trial.set_user_attr(
            "negative_fold_fraction",
            negative_fold_fraction,
        )

        trial.set_user_attr(
            "inactive_fold_fraction",
            inactive_fold_fraction,
        )

        return final_objective

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        n_jobs=config.n_jobs,
        gc_after_trial=True,
        show_progress_bar=config.show_progress_bar,
        catch=(Exception,),
    )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]

    if not completed_trials:
        failed_examples = [
            trial.user_attrs.get("error")
            for trial in study.trials
            if trial.user_attrs.get("error")
        ][:3]

        raise RuntimeError(
            "Optuna produced no completed trials. "
            f"Example errors: {failed_examples}"
        )

    best_trial = study.best_trial

    best_params = dict(
        best_trial.user_attrs.get(
            "strategy_params",
            {},
        )
    )

    holdout_datasets = slice_datasets(
        datasets=datasets,
        start=holdout.data_start,
        end=holdout.holdout_end,
    )

    holdout_results = _run_backtest_on_datasets(
        datasets=holdout_datasets,
        ticker_order=tickers,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        start=holdout.holdout_start,
        end=holdout.holdout_end,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        strategy_params=best_params,
        trade_start=holdout.holdout_start,
        risk_free_rate=risk_free_rate,
        lightweight=False,
    )

    holdout_results["evaluation_type"] = (
        "final_holdout_out_of_sample"
    )

    parameter_importance: Dict[str, float] = {}

    if config.calculate_param_importance:
        try:
            parameter_importance = {
                key: float(value)
                for key, value in (
                    optuna.importance
                    .get_param_importances(study)
                    .items()
                )
            }
        except Exception:
            parameter_importance = {}

    cv_summary = {
        "folds": [fold.to_dict() for fold in folds],
        "fold_scores": best_trial.user_attrs.get(
            "fold_scores",
            [],
        ),
        "fold_metrics": best_trial.user_attrs.get(
            "fold_metrics",
            [],
        ),
        "mean_fold_score": best_trial.user_attrs.get(
            "mean_fold_score"
        ),
        "std_fold_score": best_trial.user_attrs.get(
            "std_fold_score"
        ),
        "worst_fold_score": best_trial.user_attrs.get(
            "worst_fold_score"
        ),
        "total_cv_trades": best_trial.user_attrs.get(
            "total_cv_trades"
        ),
        "negative_fold_fraction": (
            best_trial.user_attrs.get(
                "negative_fold_fraction"
            )
        ),
        "inactive_fold_fraction": (
            best_trial.user_attrs.get(
                "inactive_fold_fraction"
            )
        ),
    }

    result = {
        "study_name": study_name,
        "study": study,
        "best_trial_number": best_trial.number,
        "best_objective": float(best_trial.value),
        "best_params": best_params,
        "cv_summary": cv_summary,
        "holdout_window": holdout.to_dict(),
        "holdout_results": holdout_results,
        "trials": optimization_trials_dataframe(study),
        "parameter_importance": parameter_importance,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "tickers": list(tickers),
        "optimization_config": asdict(config),
        "commission": commission,
        "slippage": slippage,
        "risk_free_rate": risk_free_rate,
    }

    if config.save_directory:
        result["saved_files"] = (
            save_optimization_outputs(
                result,
                config.save_directory,
            )
        )

    return result

run_optuna_optimization = optimize_hyperparameters

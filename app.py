import datetime
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from Project.backend import Backtest as Backtest
from Project.backend.data import Stocks



st.set_page_config(
    page_title="Trading Strategies Backtesting Engine",
    layout="wide",
)

st.title("TRADING STRATEGIES BACKTESTING ENGINE")



SINGLE_ASSET_STRATEGIES = [
    "Trend Following",
    "Mean Reversion",
]

MULTI_ASSET_STRATEGIES = [
    "Trend Following",
    "Mean Reversion",
    "Pairs Trading",
]



def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def format_number(
    value: Any,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    if not is_finite_number(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}{suffix}"


def format_percent_decimal(
    value: Any,
    decimals: int = 2,
) -> str:

    if not is_finite_number(value):
        return "N/A"

    return f"{float(value):.{decimals}%}"


def format_percent_number(
    value: Any,
    decimals: int = 2,
) -> str:

    if not is_finite_number(value):
        return "N/A"

    return f"{float(value):.{decimals}f}%"


def format_currency(
    value: Any,
    decimals: int = 2,
) -> str:
    if not is_finite_number(value):
        return "N/A"

    return f"${float(value):,.{decimals}f}"


def format_integer(value: Any) -> str:
    if not is_finite_number(value):
        return "N/A"

    return f"{int(float(value)):,}"


def render_metric_grid(
    metrics: Sequence[Tuple[str, str]],
    columns_per_row: int = 5,
) -> None:
    """Render metrics on a fixed grid so every row uses the same columns."""
    if columns_per_row < 1:
        raise ValueError("columns_per_row must be at least 1")

    for start in range(0, len(metrics), columns_per_row):
        row_metrics = metrics[start:start + columns_per_row]
        columns = st.columns(columns_per_row, gap="small")

        for index, column in enumerate(columns):
            with column:
                if index < len(row_metrics):
                    label, value = row_metrics[index]
                    st.metric(label, value)
                else:
                    st.empty()


def render_backtest_results(
    results: Mapping[str, Any],
    evaluation_label: Optional[str] = None,
) -> None:
    if evaluation_label:
        st.info(evaluation_label)


    holding_value = results.get("average_holding_bars")
    average_holding = (
        f"{float(holding_value):.1f} bars"
        if is_finite_number(holding_value)
        else "N/A"
    )

    # All result rows use the same five-column grid.
    st.subheader("Performance")

    render_metric_grid(
        [
            (
                "Initial Capital",
                format_currency(results.get("initial_cash")),
            ),
            (
                "Final Capital",
                format_currency(results.get("final_value")),
            ),
            (
                "Net P&L",
                format_currency(results.get("net_pnl")),
            ),
            (
                "Total Return",
                format_percent_decimal(results.get("total_return")),
            ),
            (
                "CAGR",
                format_percent_decimal(results.get("cagr")),
            ),
        ],
        columns_per_row=5,
    )

    st.subheader("Risk")

    render_metric_grid(
        [
            (
                "Sharpe Ratio",
                format_number(results.get("sharpe")),
            ),
            (
                "Sortino Ratio",
                format_number(results.get("sortino")),
            ),
            (
                "Max Drawdown",
                format_percent_number(results.get("max_drawdown")),
            ),
            (
                "Annual Volatility",
                format_percent_decimal(results.get("volatility")),
            ),
            (
                "Calmar Ratio",
                format_number(results.get("calmar")),
            ),
        ],
        columns_per_row=5,
    )

    st.subheader("Trading Statistics")

    render_metric_grid(
        [
            (
                "Closed Trades",
                format_integer(results.get("total_trades")),
            ),
            (
                "Winning Trades",
                format_integer(results.get("winning_trades")),
            ),
            (
                "Losing Trades",
                format_integer(results.get("losing_trades")),
            ),
            (
                "Win Rate",
                format_percent_decimal(results.get("win_rate")),
            ),
            (
                "Profit Factor",
                format_number(results.get("profit_factor")),
            ),
            (
                "Average Trade",
                format_currency(results.get("average_trade")),
            ),
            (
                "Average Win",
                format_currency(results.get("average_win")),
            ),
            (
                "Average Loss",
                format_currency(results.get("average_loss")),
            ),
            (
                "Expectancy",
                format_currency(results.get("expectancy")),
            ),
            (
                "Average Holding",
                average_holding,
            ),
        ],
        columns_per_row=5,
    )

    open_trades_value = results.get("open_trades")
    open_trades_text = format_integer(open_trades_value)

    if (
        is_finite_number(open_trades_value)
        and int(float(open_trades_value)) > 0
    ):
        st.warning(
            f"Open trades at the end of the backtest: {open_trades_text}"
        )
    else:
        st.caption(
            f"Open trades at the end of the backtest: {open_trades_text}"
        )


    equity_df = results.get("equity_curve")

    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
        st.subheader("Equity Curve")

        st.line_chart(
            equity_df[["equity"]],
            use_container_width=True,
        )

        if "drawdown" in equity_df.columns:
            st.subheader("Drawdown")

            drawdown_df = (
                equity_df[["drawdown"]]
                .mul(100.0)
                .rename(
                    columns={
                        "drawdown": "Drawdown (%)"
                    }
                )
            )

            st.line_chart(
                drawdown_df,
                use_container_width=True,
            )

    else:
        st.warning(
            "The backtest returned no equity-curve observations."
        )



    yearly_returns_dict = results.get(
        "yearly_returns",
        {},
    )

    st.subheader("Yearly Returns")

    if yearly_returns_dict:
        yearly_returns = pd.DataFrame(
            {
                "Year": list(
                    yearly_returns_dict.keys()
                ),
                "Return": list(
                    yearly_returns_dict.values()
                ),
            }
        )

        yearly_returns["Return (%)"] = (
            yearly_returns["Return"] * 100.0
        )

        yearly_chart = (
            yearly_returns
            .set_index("Year")[["Return (%)"]]
        )

        st.bar_chart(
            yearly_chart,
            use_container_width=True,
        )

        st.dataframe(
            yearly_returns[
                [
                    "Year",
                    "Return (%)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.caption(
            "No yearly-return observations are available."
        )



    if isinstance(equity_df, pd.DataFrame):
        with st.expander(
            "Show equity curve data"
        ):
            st.dataframe(
                equity_df,
                use_container_width=True,
            )



def build_fold_table(
    optimization: Mapping[str, Any],
) -> pd.DataFrame:
    cv_summary = optimization.get(
        "cv_summary",
        {},
    )

    folds = cv_summary.get(
        "folds",
        [],
    )

    fold_metrics = cv_summary.get(
        "fold_metrics",
        [],
    )

    fold_scores = cv_summary.get(
        "fold_scores",
        [],
    )

    rows = []

    count = max(
        len(folds),
        len(fold_metrics),
        len(fold_scores),
    )

    for index in range(count):
        fold = (
            folds[index]
            if index < len(folds)
            else {}
        )

        metrics = (
            fold_metrics[index]
            if index < len(fold_metrics)
            else {}
        )

        score = (
            fold_scores[index]
            if index < len(fold_scores)
            else None
        )

        rows.append(
            {
                "Fold": fold.get(
                    "number",
                    index + 1,
                ),
                "Validation start": fold.get(
                    "validation_start"
                ),
                "Validation end": fold.get(
                    "validation_end"
                ),
                "Bars": fold.get(
                    "validation_bars"
                ),
                "Score": score,
                "Return (%)": (
                    float(metrics["total_return"]) * 100.0
                    if is_finite_number(
                        metrics.get("total_return")
                    )
                    else None
                ),
                "CAGR (%)": (
                    float(metrics["cagr"]) * 100.0
                    if is_finite_number(
                        metrics.get("cagr")
                    )
                    else None
                ),
                "Sharpe": metrics.get(
                    "sharpe"
                ),
                "Sortino": metrics.get(
                    "sortino"
                ),
                "Max DD (%)": metrics.get(
                    "max_drawdown"
                ),
                "Trades": metrics.get(
                    "total_trades"
                ),
                "Profit Factor": metrics.get(
                    "profit_factor"
                ),
            }
        )

    return pd.DataFrame(rows)


def render_optimization_results(
    optimization: Mapping[str, Any],
) -> None:
    st.success(
        "Optuna optimization completed successfully."
    )

    st.warning(
        "The performance section below uses only the final holdout "
        "period. Optuna did not use this segment to select parameters."
    )


    st.subheader("Optimization Summary")

    cv_summary = optimization.get(
        "cv_summary",
        {},
    )

    holdout_window = optimization.get(
        "holdout_window",
        {},
    )

    render_metric_grid(
        [
            (
                "Best Trial",
                format_integer(optimization.get("best_trial_number")),
            ),
            (
                "Best CV Objective",
                format_number(
                    optimization.get("best_objective"),
                    decimals=4,
                ),
            ),
            (
                "Mean Fold Score",
                format_number(
                    cv_summary.get("mean_fold_score"),
                    decimals=4,
                ),
            ),
            (
                "Worst Fold Score",
                format_number(
                    cv_summary.get("worst_fold_score"),
                    decimals=4,
                ),
            ),
            (
                "Holdout Bars",
                format_integer(holdout_window.get("holdout_bars")),
            ),
        ],
        columns_per_row=5,
    )

    if holdout_window:
        st.caption(
            "Final holdout: "
            f"{holdout_window.get('holdout_start', 'N/A')} "
            f"to {holdout_window.get('holdout_end', 'N/A')} "
            f"({holdout_window.get('holdout_bars', 'N/A')} bars)"
        )

    # --------------------------------------------------------
    # BEST PARAMETERS
    # --------------------------------------------------------

    best_params = optimization.get(
        "best_params",
        {},
    )

    st.subheader("Best Hyperparameters")

    if best_params:
        params_df = pd.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": value,
                }
                for key, value in best_params.items()
            ]
        )

        st.dataframe(
            params_df,
            use_container_width=True,
            hide_index=True,
        )

        best_params_json = json.dumps(
            best_params,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        st.download_button(
            label="Download best parameters (JSON)",
            data=best_params_json,
            file_name="best_params.json",
            mime="application/json",
        )


    st.subheader("Walk-Forward Validation")

    fold_df = build_fold_table(
        optimization
    )

    if not fold_df.empty:
        st.dataframe(
            fold_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.caption(
            "No fold-level metrics are available."
        )

    # --------------------------------------------------------
    # PARAMETER IMPORTANCE
    # --------------------------------------------------------

    importance = optimization.get(
        "parameter_importance",
        {},
    )

    st.subheader("Parameter Importance")

    if importance:
        importance_df = (
            pd.DataFrame(
                {
                    "Parameter": list(
                        importance.keys()
                    ),
                    "Importance": list(
                        importance.values()
                    ),
                }
            )
            .sort_values(
                "Importance",
                ascending=False,
            )
            .set_index(
                "Parameter"
            )
        )

        st.bar_chart(
            importance_df,
            use_container_width=True,
        )

        st.dataframe(
            importance_df.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.caption(
            "Parameter importance could not be calculated."
        )

    # --------------------------------------------------------
    # ALL TRIALS
    # --------------------------------------------------------

    trials_df = optimization.get(
        "trials"
    )

    st.subheader("Optuna Trials")

    if isinstance(trials_df, pd.DataFrame) and not trials_df.empty:
        display_trials = trials_df.copy()

        if "objective" in display_trials.columns:
            display_trials = display_trials.sort_values(
                "objective",
                ascending=False,
                na_position="last",
            )

        st.dataframe(
            display_trials.head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download all trials (CSV)",
            data=trials_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="optuna_trials.csv",
            mime="text/csv",
        )

    else:
        st.caption(
            "No trial table is available."
        )

    # --------------------------------------------------------
    # FINAL HOLDOUT
    # --------------------------------------------------------

    holdout_results = optimization.get(
        "holdout_results"
    )

    if not isinstance(
        holdout_results,
        Mapping,
    ):
        raise RuntimeError(
            "Optimization output does not contain holdout_results."
        )

    st.divider()

    st.header(
        "Final Out-of-Sample Holdout Performance"
    )

    render_backtest_results(
        holdout_results,
        evaluation_label=(
            "These metrics are calculated on the final holdout only. "
            "They should be treated as the primary result of the optimization."
        ),
    )



if "last_run" not in st.session_state:
    st.session_state["last_run"] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Settings")


# ------------------------------------------------------------
# EXECUTION MODE
# ------------------------------------------------------------

execution_mode = st.sidebar.radio(
    "Execution mode",
    [
        "Standard Backtest",
        "Optuna Optimization",
    ],
)


# ------------------------------------------------------------
# STRATEGY TYPE
# ------------------------------------------------------------

asset_mode = st.sidebar.selectbox(
    "Strategy type",
    [
        "Single",
        "Multi",
    ],
)


# ------------------------------------------------------------
# STRATEGY
# ------------------------------------------------------------

if asset_mode == "Single":
    strategy = st.sidebar.selectbox(
        "Select strategy",
        SINGLE_ASSET_STRATEGIES,
    )

    strategy_type = (
        "single_asset_strategy"
    )

else:
    strategy = st.sidebar.selectbox(
        "Select strategy",
        MULTI_ASSET_STRATEGIES,
    )

    strategy_type = (
        "multi_asset_strategy"
    )


# ------------------------------------------------------------
# TICKERS
# ------------------------------------------------------------

if asset_mode == "Single":
    selected_ticker = st.sidebar.selectbox(
        "Select ticker",
        Stocks.tickers,
    )

    stocks = [
        selected_ticker
    ]

else:
    stocks = st.sidebar.multiselect(
        "Select tickers",
        Stocks.tickers,
    )


# ------------------------------------------------------------
# DATES
# ------------------------------------------------------------

default_end = datetime.date.today()

default_start = (
    default_end
    - datetime.timedelta(
        days=365 * 8
    )
)

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
)

end_date = st.sidebar.date_input(
    "End date",
    value=default_end,
)



cash = st.sidebar.slider(
    "Initial cash",
    min_value=10_000,
    max_value=1_000_000,
    value=100_000,
    step=10_000,
)


# ------------------------------------------------------------
# TRANSACTION COSTS
# ------------------------------------------------------------

commission = st.sidebar.slider(
    "Commission",
    min_value=0.0,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f",
)

slippage = st.sidebar.slider(
    "Slippage",
    min_value=0.0,
    max_value=0.005,
    value=0.0005,
    step=0.0001,
    format="%.4f",
)


optuna_settings: Dict[str, Any] = {}

if execution_mode == "Optuna Optimization":
    st.sidebar.divider()

    st.sidebar.subheader(
        "Optuna settings"
    )

    optuna_settings[
        "n_trials"
    ] = st.sidebar.number_input(
        "Trials",
        min_value=10,
        max_value=2_000,
        value=100,
        step=10,
    )

    optuna_settings[
        "n_splits"
    ] = st.sidebar.slider(
        "Walk-forward folds",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
    )

    optuna_settings[
        "holdout_fraction"
    ] = st.sidebar.slider(
        "Final holdout",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        format="%.2f",
    )

    optuna_settings[
        "initial_train_fraction"
    ] = st.sidebar.slider(
        "Initial development history",
        min_value=0.20,
        max_value=0.70,
        value=0.40,
        step=0.05,
        format="%.2f",
    )

    optuna_settings[
        "warmup_bars"
    ] = st.sidebar.number_input(
        "Indicator warm-up bars",
        min_value=100,
        max_value=1_000,
        value=400,
        step=50,
    )

    optuna_settings[
        "min_validation_bars"
    ] = st.sidebar.number_input(
        "Minimum validation bars",
        min_value=30,
        max_value=504,
        value=126,
        step=21,
    )

    optuna_settings[
        "embargo_bars"
    ] = st.sidebar.number_input(
        "Embargo bars",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
    )

    optuna_settings[
        "min_total_trades"
    ] = st.sidebar.number_input(
        "Minimum CV trades",
        min_value=0,
        max_value=200,
        value=8,
        step=1,
    )

    optuna_settings[
        "max_fold_drawdown"
    ] = st.sidebar.slider(
        "Maximum preferred fold drawdown",
        min_value=0.05,
        max_value=0.60,
        value=0.25,
        step=0.05,
        format="%.2f",
    )

    optuna_settings[
        "random_seed"
    ] = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=100_000,
        value=42,
        step=1,
    )

    optuna_settings[
        "persistent_storage"
    ] = st.sidebar.checkbox(
        "Continue studies from SQLite",
        value=False,
        help=(
            "When enabled, compatible trials are stored in "
            "optuna_studies.db and can be continued later."
        ),
    )

    st.sidebar.caption(
        "For the first test, use 20-50 trials. "
        "For a more serious study, use 100-300 trials."
    )



st.subheader("Backtest configuration")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Strategy",
        strategy,
    )

with col2:
    st.metric(
        "Mode",
        asset_mode,
    )

with col3:
    st.metric(
        "Assets",
        len(stocks),
    )

with col4:
    st.metric(
        "Execution",
        (
            "Optuna"
            if execution_mode
            == "Optuna Optimization"
            else "Backtest"
        ),
    )


# ============================================================
# VALIDATION
# ============================================================

validation_error = None

if start_date >= end_date:
    validation_error = (
        "Start date must be earlier than end date."
    )

elif not stocks:
    validation_error = (
        "Select at least one ticker."
    )

elif (
    asset_mode == "Single"
    and len(stocks) != 1
):
    validation_error = (
        "Single-asset strategy requires exactly one ticker."
    )

elif (
    strategy == "Pairs Trading"
    and len(stocks) != 2
):
    validation_error = (
        "Pairs Trading requires exactly two tickers."
    )

elif (
    execution_mode == "Optuna Optimization"
    and (
        end_date - start_date
    ).days < 365 * 3
):
    validation_error = (
        "Optuna optimization should use at least about three years "
        "of daily history. A longer sample is preferable."
    )

if validation_error:
    st.warning(
        validation_error
    )


# ============================================================
# RUN SIGNATURE
# ============================================================

current_signature = {
    "execution_mode": execution_mode,
    "asset_mode": asset_mode,
    "strategy": strategy,
    "strategy_type": strategy_type,
    "stocks": tuple(stocks),
    "start_date": str(start_date),
    "end_date": str(end_date),
    "cash": int(cash),
    "commission": float(commission),
    "slippage": float(slippage),
    "optuna_settings": optuna_settings,
}



button_label = (
    "Optimize Hyperparameters"
    if execution_mode
    == "Optuna Optimization"
    else "Start Backtest"
)

run_button = st.button(
    button_label,
    type="primary",
    disabled=validation_error is not None,
)



if run_button:
    try:
        if (
            execution_mode
            == "Standard Backtest"
        ):
            with st.spinner(
                "Running backtest..."
            ):
                results = (
                    Backtest.run_backtest(
                        tickers=stocks,
                        strategy_name=strategy,
                        strategy_type=strategy_type,
                        start=start_date,
                        end=end_date,
                        initial_cash=cash,
                        commission=commission,
                        slippage=slippage,
                    )
                )

            st.session_state[
                "last_run"
            ] = {
                "kind": "backtest",
                "signature": current_signature,
                "results": results,
            }

        else:
            storage = (
                "sqlite:///optuna_studies.db"
                if optuna_settings[
                    "persistent_storage"
                ]
                else None
            )

            config = (
                Backtest.OptimizationConfig(
                    n_trials=int(
                        optuna_settings[
                            "n_trials"
                        ]
                    ),
                    n_jobs=1,
                    n_splits=int(
                        optuna_settings[
                            "n_splits"
                        ]
                    ),
                    holdout_fraction=float(
                        optuna_settings[
                            "holdout_fraction"
                        ]
                    ),
                    initial_train_fraction=float(
                        optuna_settings[
                            "initial_train_fraction"
                        ]
                    ),
                    warmup_bars=int(
                        optuna_settings[
                            "warmup_bars"
                        ]
                    ),
                    min_validation_bars=int(
                        optuna_settings[
                            "min_validation_bars"
                        ]
                    ),
                    embargo_bars=int(
                        optuna_settings[
                            "embargo_bars"
                        ]
                    ),
                    min_total_trades=int(
                        optuna_settings[
                            "min_total_trades"
                        ]
                    ),
                    min_trades_per_fold=1,
                    max_fold_drawdown=float(
                        optuna_settings[
                            "max_fold_drawdown"
                        ]
                    ),
                    stability_penalty=0.35,
                    worst_fold_weight=0.15,
                    negative_fold_penalty=0.35,
                    inactive_fold_penalty=0.30,
                    random_seed=int(
                        optuna_settings[
                            "random_seed"
                        ]
                    ),
                    storage=storage,
                    load_if_exists=True,
                    show_progress_bar=False,
                    calculate_param_importance=True,
                    save_directory=None,
                )
            )

            with st.spinner(
                "Running walk-forward Optuna optimization. "
                "This may take several minutes..."
            ):
                optimization = (
                    Backtest.optimize_hyperparameters(
                        tickers=stocks,
                        strategy_name=strategy,
                        strategy_type=strategy_type,
                        start=start_date,
                        end=end_date,
                        initial_cash=cash,
                        commission=commission,
                        slippage=slippage,
                        risk_free_rate=0.0,
                        optimization_config=config,
                    )
                )

            st.session_state[
                "last_run"
            ] = {
                "kind": "optimization",
                "signature": current_signature,
                "optimization": optimization,
            }

    except Exception as exc:
        st.session_state[
            "last_run"
        ] = None

        st.exception(
            exc
        )


last_run = st.session_state.get(
    "last_run"
)

if last_run:
    if (
        last_run.get("signature")
        != current_signature
    ):
        st.warning(
            "The displayed result belongs to the previous configuration. "
            "Run the engine again to update it."
        )

    if last_run.get("kind") == "backtest":
        st.success(
            "Backtest completed successfully."
        )

        render_backtest_results(
            last_run["results"]
        )

    elif (
        last_run.get("kind")
        == "optimization"
    ):
        render_optimization_results(
            last_run[
                "optimization"
            ]
        )

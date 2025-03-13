import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pnl(predictions, true_percentage_changes, initial_capital=1000):

    pnl = np.zeros_like(true_percentage_changes, dtype=float)

    # Initialize position **
    position = 0

    for i in range(len(predictions)):
        if i == 0:
            # No position at the first time step
            pnl[i] = 0
            continue

        if predictions[i] == 1:  # Predict Up (go long)
            position = 1
        elif predictions[i] == 0:  # Predict Down (sell)
            position = 0

        pnl[i] = position * true_percentage_changes[i]

    # Cumulative P&L
    cumulative_pnl = np.cumprod(1 + pnl) - 1

    # Final capital
    final_capital = initial_capital * np.prod(1 + pnl)

    return pnl, cumulative_pnl, final_capital


def backtest_strategy(
    predictions, true_percentage_changes, timestamps, initial_capital=1000
):

    pnl, cumulative_pnl, final_capital = compute_pnl(
        predictions, true_percentage_changes, initial_capital
    )

    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.to_datetime(timestamps)

    df = pd.DataFrame(
        {"timestamp": timestamps, "pnl": pnl, "cumulative_pnl": cumulative_pnl}
    )

    df["date"] = df["timestamp"].dt.date
    num_days = len(df["date"].unique())

    def compute_daily_pnl(group):
        if len(group) == 0:
            return 0
        return np.cumprod(1 + group["pnl"].values)[-1] - 1

    daily_pnl = df.groupby("date").apply(compute_daily_pnl)

    cumulative_daily_pnl = np.cumprod(1 + daily_pnl) - 1

    # Annualized return (using daily P&L)
    annualized_return = np.prod(1 + daily_pnl) ** (252 / len(daily_pnl)) - 1

    # Volatility (annualized, using daily P&L)
    annualized_volatility = np.std(daily_pnl) * np.sqrt(252)

    results = {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "time": num_days,
        "pnl": pnl,
        "cumulative_pnl": cumulative_pnl,
        "daily_pnl": daily_pnl,
        "cumulative_daily_pnl": cumulative_daily_pnl,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
    }

    return results


def plot_backtest_results(results, label="Strategy"):

    print(f"\nBacktest Metrics for {label}:")
    print(f"Duration: {results["time"]} days")
    print(
        f"Initial Capital: {results['initial_capital']:.2f}, Final Capital: {results['final_capital']:.2f}"
    )
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Annualized Volatility: {results['annualized_volatility']:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(results["cumulative_pnl"], label=f"Cumulative P&L ({label})")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative P&L")
    plt.title(f"Cumulative Profit and Loss ({label})")
    plt.legend()
    plt.grid()
    plt.show()

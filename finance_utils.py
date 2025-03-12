import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pnl(predictions, true_percentage_changes, initial_capital=1000):
    """
    Compute the Profit and Loss (P&L) based on predictions and true percentage changes.

    Parameters:
        predictions (np.array): Predicted classes (0 for Down, 1 for Up).
        true_percentage_changes (np.array): True percentage changes of the price.
        initial_capital (float): Initial capital for trading.

    Returns:
        pnl (np.array): Array of P&L for each time step.
        cumulative_pnl (np.array): Cumulative P&L over time.
        final_capital (float): Final capital at the end of the period.
    """
    # Initialize P&L array
    pnl = np.zeros_like(true_percentage_changes, dtype=float)

    # Initialize position (1 for long, -1 for short, 0 for no position)
    position = 0

    # Iterate through predictions and true percentage changes
    for i in range(len(predictions)):
        if i == 0:
            # No position at the first time step
            pnl[i] = 0
            continue

        # Determine the trading decision based on the prediction
        if predictions[i] == 1:  # Predict Up (go long)
            position = 1
        elif predictions[i] == 0:  # Predict Down (go short)
            position = 0

        # Compute P&L for the current time step
        pnl[i] = position * true_percentage_changes[i]

    # Compute cumulative P&L
    cumulative_pnl = np.cumprod(1 + pnl) - 1

    # Compute final capital
    final_capital = initial_capital * np.prod(1 + pnl)

    return pnl, cumulative_pnl, final_capital


def backtest_strategy(
    predictions, true_percentage_changes, timestamps, initial_capital=1000
):
    """
    Backtest a trading strategy and compute relevant metrics.

    Parameters:
        predictions (np.array): Predicted classes (0 for Down, 1 for Up).
        true_percentage_changes (np.array): True percentage changes of the price.
        timestamps (pd.Series or np.array): Timestamps for each time step.
        initial_capital (float): Initial capital for trading.

    Returns:
        results (dict): Dictionary containing backtest results and metrics.
    """
    # Compute P&L
    pnl, cumulative_pnl, final_capital = compute_pnl(
        predictions, true_percentage_changes, initial_capital
    )

    # Convert timestamps to datetime if not already
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.to_datetime(timestamps)

    # Create a DataFrame for easier analysis
    df = pd.DataFrame(
        {"timestamp": timestamps, "pnl": pnl, "cumulative_pnl": cumulative_pnl}
    )

    # Group by day to compute daily P&L (for annualized metrics)
    df["date"] = df["timestamp"].dt.date
    num_days = len(df["date"].unique())

    # Compute daily P&L using a safer approach
    def compute_daily_pnl(group):
        if len(group) == 0:
            return 0  # Return 0 if the group is empty
        return (
            np.cumprod(1 + group["pnl"].values)[-1] - 1
        )  # Compute cumulative product and return the last value

    daily_pnl = df.groupby("date").apply(compute_daily_pnl)

    # Compute cumulative daily P&L
    cumulative_daily_pnl = np.cumprod(1 + daily_pnl) - 1

    # Compute annualized return (using daily P&L)
    annualized_return = np.prod(1 + daily_pnl) ** (252 / len(daily_pnl)) - 1

    # Compute volatility (annualized, using daily P&L)
    annualized_volatility = np.std(daily_pnl) * np.sqrt(252)

    # Compile results
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
    """
    Plot the backtest results, including cumulative P&L for every period and drawdown,
    and display all computed metrics.

    Parameters:
        results (dict): Dictionary containing backtest results and metrics.
        label (str): Label to identify the predictive method used (e.g., "LSTM", "ARIMA").
    """
    # Display all computed metrics with the label
    print(f"\nBacktest Metrics for {label}:")
    print(f"Duration: {results["time"]} days")
    print(
        f"Initial Capital: {results['initial_capital']:.2f}, Final Capital: {results['final_capital']:.2f}"
    )
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Annualized Volatility: {results['annualized_volatility']:.2%}")

    # Plot cumulative P&L for every period
    plt.figure(figsize=(10, 6))
    plt.plot(results["cumulative_pnl"], label=f"Cumulative P&L ({label})")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative P&L")
    plt.title(f"Cumulative Profit and Loss ({label})")
    plt.legend()
    plt.grid()
    plt.show()

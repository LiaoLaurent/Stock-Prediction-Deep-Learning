import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def plot_close_price_split(df, train_size, val_size):
    """Plots stock close price with a discontinuous time axis (removing overnight gaps)."""

    plt.figure(figsize=(14, 5))

    # Define market open and close times
    market_open_time = pd.to_datetime("09:30:00").time()
    market_close_time = pd.to_datetime("16:00:00").time()

    # Create a new transformed index for a continuous time axis
    df = df.copy()
    df["market_time"] = np.nan  # Placeholder for transformed time index
    continuous_time = 0  # Track cumulative trading time without gaps

    for date, group in df.groupby(df.index.date):  # Loop through each trading day
        market_open = group.between_time(market_open_time, market_close_time)

        if not market_open.empty:  # Ensure there's data for this day
            time_since_open = (market_open.index - market_open.index[0]).total_seconds()
            df.loc[market_open.index, "market_time"] = continuous_time + time_since_open
            continuous_time += time_since_open[-1]  # Update cumulative market time

    # Drop NaNs (removes non-market hours)
    df = df.dropna(subset=["market_time"])

    # Plot the close price with transformed time axis
    plt.plot(
        df["market_time"],
        df["mid_price_close"],
        color="blue",
        linewidth=1.5,
        label="Stock Close Price",
    )

    # Get transformed time indices for splits
    train_end = df["market_time"].iloc[train_size - 1]
    val_end = df["market_time"].iloc[train_size + val_size - 1]

    # Highlight Training, Validation, and Testing Regions
    plt.axvspan(
        df["market_time"].iloc[0],
        train_end,
        color="blue",
        alpha=0.2,
        label="Training Data",
    )
    plt.axvspan(train_end, val_end, color="orange", alpha=0.2, label="Validation Data")
    plt.axvspan(
        val_end, df["market_time"].iloc[-1], color="red", alpha=0.2, label="Test Data"
    )

    plt.xlabel("Trading Time (Discontinuous)")
    plt.ylabel("Close Price")
    plt.title("Close Price Distribution")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def plot_samples(datasets, titles, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Define colors for the three classes
    target_colors = {-1: "red", 0: "blue", 1: "green"}
    target_labels = {-1: "Down", 0: "Neutral", 1: "Up"}

    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        sample_idx = np.random.randint(len(dataset))
        sample, target = dataset[sample_idx]
        sample = sample.numpy().T  # Transpose for feature-wise plotting

        for j, feature in enumerate(feature_names):
            axes[i].plot(sample[j], marker="o", linestyle="-", alpha=0.7, label=feature)

        # Get target class as integer
        target_value = int(target.item())

        # Determine color and text for target
        target_color = target_colors.get(target_value, "black")
        target_text = f"Target: {target_labels.get(target_value, 'Unknown')}"

        # Set title and target annotation
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].text(
            0.5,
            -0.15,
            target_text,
            fontsize=12,
            fontweight="bold",
            color=target_color,
            ha="center",
            transform=axes[i].transAxes,
        )

        axes[i].grid(True)

    # Shared legend with flexible layout
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(feature_names), 4),
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy"
    )
    plt.plot(
        range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.show()


def plot_evaluation_metrics(y_true, y_pred, log_probabilities):
    probabilities = np.exp(log_probabilities)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=[-1, 0, 1],
        yticklabels=[-1, 0, 1],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Histogram of Predicted Probabilities
    for i, class_label in enumerate(["Class -1", "Class 0", "Class 1"]):
        label = i - 1
        sns.histplot(
            probabilities[y_true == label][:, i], bins=30, label=class_label, ax=axes[1]
        )
    axes[1].set_title("Probability Distribution")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].legend()

    # Scatter Plot of Predictions
    scatter = sns.scatterplot(
        x=np.arange(len(probabilities)),
        y=probabilities.max(axis=1),
        hue=y_true,
        palette={-1: "red", 0: "blue", 1: "green"},
        alpha=0.7,
        ax=axes[2],
    )
    axes[2].set_title("Scatter Plot of Predictions")
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("Max Predicted Probability")

    handles, labels = scatter.get_legend_handles_labels()
    new_labels = ["Class -1", "Class 0", "Class 1"]
    axes[2].legend(handles, new_labels, title="True Class")

    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Print Evaluation Metrics
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_baseline_evaluation_metrics(baseline_targets, baseline_preds):
    conf_matrix = confusion_matrix(baseline_targets, baseline_preds, labels=[-1, 0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Down", "Neutral", "Up"],
        yticklabels=["Down", "Neutral", "Up"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline Strategy Confusion Matrix")
    plt.show()

    # Print Evaluation Metrics
    print(f"Accuracy: {accuracy_score(baseline_targets, baseline_preds):.4f}")
    print(classification_report(baseline_targets, baseline_preds, zero_division=0))

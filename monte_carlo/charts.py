# Visualizations here, be careful with sending big arrays here, can crash the UI

import matplotlib.pyplot as plt
import numpy as np


def _figure(title, x_label, y_label):
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(alpha=0.25)
    return figure, axes


def path_chart(display_paths, time_grid, strike):
    figure, axes = _figure(
        "Sample simulated asset paths", "Time (years)", "Asset price"
    )
    line_count = min(100, len(display_paths))
    indices = np.linspace(0, len(display_paths) - 1, line_count, dtype=int)
    axes.plot(time_grid, display_paths[indices].T, alpha=0.35, linewidth=0.8)
    axes.axhline(
        strike, color="black", linestyle="--", label=f"Strike = {strike:.2f}"
    )
    axes.legend()
    figure.tight_layout()
    return figure


def terminal_price_chart(terminal_prices):
    figure, axes = _figure(
        "Final asset-price distribution", "Asset price at maturity", "Frequency"
    )
    mean_price = np.mean(terminal_prices)
    axes.hist(terminal_prices, bins=50, color="#4169E1", alpha=0.8)
    axes.axvline(
        mean_price,
        color="darkred",
        linestyle="--",
        label=f"Mean = {mean_price:.2f}",
    )
    axes.legend()
    figure.tight_layout()
    return figure


def payoff_chart(discounted_payoffs, option_price):
    figure, axes = _figure(
        "Discounted payoff distribution", "Present value of payoff", "Frequency"
    )
    axes.hist(discounted_payoffs, bins=50, color="#2E8B57", alpha=0.8)
    axes.axvline(
        option_price,
        color="darkred",
        linestyle="--",
        label=f"Estimated price = {option_price:.4f}",
    )
    axes.legend()
    figure.tight_layout()
    return figure


def price_convergence_chart(discounted_payoffs, option_price):
    figure, axes = _figure(
        "Monte Carlo price convergence", "Number of paths", "Option-price estimate"
    )
    counts = np.arange(1, len(discounted_payoffs) + 1)
    running_mean = np.cumsum(discounted_payoffs) / counts

    running_std = np.zeros_like(running_mean)
    if len(discounted_payoffs) > 1:
        sum_of_squares = np.cumsum(discounted_payoffs**2)
        centered_sum = sum_of_squares[1:] - counts[1:] * running_mean[1:] ** 2
        running_std[1:] = np.sqrt(
            np.maximum(centered_sum, 0.0) / (counts[1:] - 1)
        )
    half_width = 1.96 * running_std / np.sqrt(counts)

    # Downsample only the plotted points. The calculations above still use all
    # payoffs, including all one million when that path count is selected.
    shown = np.unique(
        np.linspace(0, len(discounted_payoffs) - 1, min(1500, len(counts)), dtype=int)
    )
    axes.plot(counts[shown], running_mean[shown], color="#4169E1")
    axes.fill_between(
        counts[shown],
        (running_mean - half_width)[shown],
        (running_mean + half_width)[shown],
        color="#4169E1",
        alpha=0.18,
        label="Approx. 95% confidence interval",
    )
    axes.axhline(option_price, color="darkred", linestyle="--", label="Final estimate")
    axes.legend()
    figure.tight_layout()
    return figure

def variance_mean_convergence_chart(discounted_payoffs):
    n = len(discounted_payoffs)
    counts = np.arange(1, n + 1)
    running_mean = np.cumsum(discounted_payoffs) / counts

    running_std = np.zeros(n)
    if n > 1:
        sum_sq = np.cumsum(discounted_payoffs ** 2)
        # Online variance via E[X^2] - E[X]^2, corrected for sample variance.
        centered = sum_sq[1:] - counts[1:] * running_mean[1:] ** 2
        running_std[1:] = np.sqrt(np.maximum(centered, 0.0) / (counts[1:] - 1))

    shown = np.unique(
        np.linspace(0, n - 1, min(1500, n), dtype=int)
    )

    figure, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title("Running mean and standard deviation")
    ax1.set_xlabel("Number of paths")
    ax1.set_ylabel("Running mean", color="#4169E1")
    ax1.tick_params(axis="y", labelcolor="#4169E1")
    ax1.grid(alpha=0.25)
    ax1.plot(counts[shown], running_mean[shown], color="#4169E1", label="Running mean")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Running std deviation", color="#2E8B57")
    ax2.tick_params(axis="y", labelcolor="#2E8B57")
    # Skip index 0 where std is undefined (only one observation).
    shown_std = shown[shown > 0]
    ax2.plot(
        counts[shown_std],
        running_std[shown_std],
        color="#2E8B57",
        alpha=0.85,
        label="Running std dev",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    figure.tight_layout()
    return figure

def volatility_path_chart(display_volatility_paths, time_grid):
    """Plot a representative subset of Heston volatility paths."""
    figure, axes = _figure(
        "Sample simulated volatility paths", "Time (years)", "Volatility"
    )
    line_count = min(100, len(display_volatility_paths))
    indices = np.linspace(
        0, len(display_volatility_paths) - 1, line_count, dtype=int
    )
    axes.plot(
        time_grid,
        display_volatility_paths[indices].T,
        alpha=0.35,
        linewidth=0.8,
    )
    figure.tight_layout()
    return figure


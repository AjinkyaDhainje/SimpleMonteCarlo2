"""The short, high-level flow that turns inputs into a price and UI-safe data."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .engine import MODELS, MonteCarloEngine
from .payoffs import PAYOFFS


@dataclass
class SimulationResult:
    """Only the data required for the price and four UI charts."""

    display_paths: np.ndarray
    terminal_prices: np.ndarray
    discounted_payoffs: np.ndarray
    time_grid: np.ndarray
    option_price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    elapsed_seconds: float


@dataclass
class MultiRunSimulationResult:
    """Prices from every run and full chart data from only the final run."""

    option_prices: list[float]
    average_option_price: float
    final_run: SimulationResult
    elapsed_seconds: float


class SimulationManager:
    """Validate, simulate, calculate payoffs, and summarize the result."""

    def __init__(self, engine_class=MonteCarloEngine):
        self.engine_class = engine_class

    def run_multiple(self, inputs):
        """Run the same simulation repeatedly and retain only the last paths.

        Every option price is kept for the UI. The large path, terminal-price,
        and payoff arrays from earlier runs are released as the loop advances,
        so increasing ``runs`` does not multiply the retained chart data.
        """
        inputs.validate_common_inputs()
        start_time = perf_counter()
        option_prices = []
        final_run = None

        for run_index in range(inputs.runs):
            current_run = self.run(inputs)
            option_prices.append(current_run.option_price)

            if run_index == inputs.runs - 1:
                final_run = current_run
            else:
                # Earlier runs contribute only their final option price.
                del current_run

        return MultiRunSimulationResult(
            option_prices=option_prices,
            average_option_price=float(np.mean(option_prices)),
            final_run=final_run,
            elapsed_seconds=perf_counter() - start_time,
        )

    def run(self, inputs):
        inputs.validate_common_inputs()
        model = self._selected(MODELS, inputs.model, "model")
        payoff = self._selected(PAYOFFS, inputs.payoff, "payoff")
        model.validate(inputs)
        payoff.validate(inputs)
        engine = self.engine_class(inputs)

        start_time = perf_counter()
        display_count = min(inputs.num_paths, 10_000)
        display_paths = np.empty((display_count, inputs.num_steps + 1))
        terminal_prices = np.empty(inputs.num_paths)
        discounted_payoffs = np.empty(inputs.num_paths)

        displayed = 0
        discount_factor = np.exp(-inputs.risk_free_rate * inputs.maturity)

        batch_size = engine.batch_size()
        for offset in range(0, inputs.num_paths, batch_size):
            count = min(batch_size, inputs.num_paths - offset)
            paths = engine.generate_paths(count)
            count = len(paths)
            batch_payoffs = payoff.calculate(paths, inputs) * discount_factor

            terminal_prices[offset : offset + count] = paths[:, -1]
            discounted_payoffs[offset : offset + count] = batch_payoffs

            keep = min(count, display_count - displayed)
            if keep > 0:
                display_paths[displayed : displayed + keep] = paths[:keep]
                displayed += keep

        option_price = float(np.mean(discounted_payoffs))
        # Adjacent paths are antithetic pairs. Their average is one independent
        # observation for the standard-error calculation.
        pair_count = inputs.num_paths // 2
        paired_payoffs = (
            discounted_payoffs[0 : 2 * pair_count : 2]
            + discounted_payoffs[1 : 2 * pair_count : 2]
        ) / 2.0
        if inputs.num_paths % 2 == 1:
            paired_payoffs = np.append(paired_payoffs, discounted_payoffs[-1])
        if len(paired_payoffs) > 1:
            standard_error = float(
                np.std(paired_payoffs, ddof=1) / np.sqrt(len(paired_payoffs))
            )
        else:
            standard_error = 0.0
        half_width = 1.96 * standard_error

        return SimulationResult(
            display_paths=display_paths,
            terminal_prices=terminal_prices,
            discounted_payoffs=discounted_payoffs,
            time_grid=np.linspace(0.0, inputs.maturity, inputs.num_steps + 1),
            option_price=option_price,
            standard_error=standard_error,
            confidence_interval=(
                option_price - half_width,
                option_price + half_width,
            ),
            elapsed_seconds=perf_counter() - start_time,
        )

    @staticmethod
    def _selected(registry, name, component_name):
        try:
            return registry[name]
        except KeyError as error:
            raise ValueError(f"Unknown {component_name}: {name}.") from error

"""Dedicated Monte Carlo engine for single-underlying autocallables."""

import math

import numpy as np

from .engine import MonteCarloEngine


class AutocallableEngine(MonteCarloEngine):
    """Generate paths and value a standard memory-coupon autocallable.

    Path generation is inherited from ``MonteCarloEngine`` so the same GBM LV
    and Heston SV models are reused. Product validation, observation handling,
    early redemption, coupons, downside protection, and cash-flow discounting
    live in this dedicated engine.
    """

    name = "Autocallable"
    extra_inputs = [
        {
            "name": "autocall_notional",
            "label": "Notional",
            "min_value": 0.01,
            "step": 100.0,
        },
        {
            "name": "autocall_coupon_rate",
            "label": "Coupon per observation",
            "min_value": 0.0,
            "step": 0.005,
            "format": "%.4f",
            "help": "Decimal rate; for example, 0.02 means 2% of notional.",
        },
        {
            "name": "autocall_observation_frequency_months",
            "label": "Observation frequency (months)",
            "min_value": 0.01,
            "step": 1.0,
        },
        {
            "name": "autocall_barrier",
            "label": "Autocall barrier (initial spot ratio)",
            "min_value": 0.0,
            "step": 0.05,
            "format": "%.4f",
            "help": "1.00 means 100% of the initial asset price.",
        },
        {
            "name": "autocall_coupon_barrier",
            "label": "Coupon barrier (initial spot ratio)",
            "min_value": 0.0,
            "step": 0.05,
            "format": "%.4f",
        },
        {
            "name": "autocall_protection_barrier",
            "label": "Protection barrier (initial spot ratio)",
            "min_value": 0.0,
            "step": 0.05,
            "format": "%.4f",
        },
        {
            "name": "autocall_memory_coupon",
            "label": "Memory coupon",
            "widget": "checkbox",
            "help": "Pay missed coupons when a later coupon barrier is met.",
        },
    ]

    def validate_product(self):
        """Validate the autocallable fields attached to this engine's inputs."""
        inputs = self.inputs
        numeric_values = {
            "notional": inputs.autocall_notional,
            "coupon rate": inputs.autocall_coupon_rate,
            "observation frequency": inputs.autocall_observation_frequency_months,
            "autocall barrier": inputs.autocall_barrier,
            "coupon barrier": inputs.autocall_coupon_barrier,
            "protection barrier": inputs.autocall_protection_barrier,
        }
        if not all(math.isfinite(value) for value in numeric_values.values()):
            raise ValueError("All autocallable numeric inputs must be finite.")
        if inputs.autocall_notional <= 0:
            raise ValueError("Autocallable notional must be greater than zero.")
        if inputs.autocall_coupon_rate < 0:
            raise ValueError("Autocallable coupon rate cannot be negative.")
        if inputs.autocall_observation_frequency_months <= 0:
            raise ValueError("Observation frequency must be greater than zero.")
        if inputs.autocall_protection_barrier < 0:
            raise ValueError("Protection barrier cannot be negative.")
        if not (
            inputs.autocall_protection_barrier
            <= inputs.autocall_coupon_barrier
            <= inputs.autocall_barrier
        ):
            raise ValueError(
                "Barriers must satisfy protection <= coupon <= autocall."
            )
        if not isinstance(inputs.autocall_memory_coupon, bool):
            raise ValueError("Memory coupon must be true or false.")

    def observation_schedule(self):
        """Return unique grid indices and times, always including maturity."""
        inputs = self.inputs
        interval = inputs.autocall_observation_frequency_months / 12.0
        observation_times = np.arange(interval, inputs.maturity, interval)
        observation_times = np.append(observation_times, inputs.maturity)
        indices = np.rint(
            observation_times / inputs.maturity * inputs.num_steps
        ).astype(int)
        indices = np.unique(np.clip(indices, 1, inputs.num_steps))
        return indices, indices * inputs.maturity / inputs.num_steps

    def calculate_discounted_payoffs(self, paths):
        """Return the present value of all product cash flows for each path."""
        inputs = self.inputs
        observation_indices, observation_times = self.observation_schedule()
        path_count = len(paths)
        present_values = np.zeros(path_count)
        alive = np.ones(path_count, dtype=bool)
        missed_coupons = np.zeros(path_count, dtype=int)
        coupon_amount = inputs.autocall_notional * inputs.autocall_coupon_rate

        for index, payment_time in zip(observation_indices, observation_times):
            active = np.flatnonzero(alive)
            if len(active) == 0:
                break

            relative_spot = paths[active, index] / inputs.start_price
            coupon_due = relative_spot >= inputs.autocall_coupon_barrier
            coupon_multiples = np.ones(len(active))
            if inputs.autocall_memory_coupon:
                coupon_multiples += missed_coupons[active]
            cash_flow = coupon_due * coupon_amount * coupon_multiples

            called = relative_spot >= inputs.autocall_barrier
            cash_flow += called * inputs.autocall_notional
            present_values[active] += cash_flow * math.exp(
                -inputs.risk_free_rate * payment_time
            )

            if inputs.autocall_memory_coupon:
                missed_coupons[active[~coupon_due]] += 1
                missed_coupons[active[coupon_due]] = 0

            alive[active[called]] = False

        survivors = np.flatnonzero(alive)
        if len(survivors):
            terminal_ratio = paths[survivors, -1] / inputs.start_price
            redemption = np.where(
                terminal_ratio >= inputs.autocall_protection_barrier,
                inputs.autocall_notional,
                inputs.autocall_notional * terminal_ratio,
            )
            present_values[survivors] += redemption * math.exp(
                -inputs.risk_free_rate * inputs.maturity
            )

        return present_values

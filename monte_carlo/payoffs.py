# All different payoffs live here, add new ones as a separate class

import math

import numpy as np


def option_direction(option_type):
    """Calls use +1 and puts use -1 in the same payoff equation."""
    return {"Call": 1.0, "Put": -1.0}[option_type]


class VanillaPayoff:
    name = "Vanilla"
    extra_inputs = []

    def validate(self, inputs):
        pass

    def calculate(self, paths, inputs):
        terminal_prices = paths[:, -1]
        return np.maximum(
            option_direction(inputs.option_type)
            * (terminal_prices - inputs.strike),
            0.0,
        )


class AsianPayoff:
    name = "Asian"
    extra_inputs = [
        {
            "name": "averaging_days",
            "label": "Average over final days",
            "min_value": 0.01,
            "step": 1.0,
            "help": "Prices from this many days before maturity are averaged.",
            "limited_by_maturity": True,
        }
    ]

    def validate(self, inputs):
        if not math.isfinite(inputs.averaging_days):
            raise ValueError("Averaging days must be a finite number.")
        if inputs.averaging_days <= 0:
            raise ValueError("Averaging days must be greater than zero.")
        if inputs.averaging_days > inputs.maturity * 365.0:
            raise ValueError("Averaging days cannot exceed the maturity in days.")

    def calculate(self, paths, inputs):
        step_count = paths.shape[1] - 1
        dt = inputs.maturity / step_count
        averaging_years = inputs.averaging_days / 365.0
        observations = math.ceil(averaging_years / dt)
        observations = min(step_count, max(1, observations))

        average_price = np.mean(paths[:, -observations:], axis=1)
        return np.maximum(
            option_direction(inputs.option_type) * (average_price - inputs.strike),
            0.0,
        )

class LookbackPayoff:
    name = "Lookback"
    extra_inputs = []

    def validate(self, inputs):
        pass

    def calculate(self, paths, inputs):
        direction = option_direction(inputs.option_type)

        # For Calls (direction is 1), find max price
        # For Puts (direction is -1), find min price
        path_extremes = np.where(
            direction == 1,
            np.max(paths, axis=1),
            np.min(paths, axis=1)
        )

        return np.maximum(
            direction * (path_extremes - inputs.strike),
            0.0,
        )


# Add a payoff class here to make it available to the manager and UI.
PAYOFFS = {
    VanillaPayoff.name: VanillaPayoff(),
    AsianPayoff.name: AsianPayoff(),
    LookbackPayoff.name: LookbackPayoff()
}

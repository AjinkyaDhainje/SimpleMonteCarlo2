"""All inputs for one simulation live in this single, simple object."""

from dataclasses import dataclass
import math


@dataclass
class SimulationInputs:
    """Inputs shared by the UI, models, payoffs, and simulation engine.

    Model-specific and payoff-specific values are deliberately kept here too.
    A selected component reads only the fields it needs. For example, the GBM
    model ignores all jump fields and the vanilla payoff ignores
    ``averaging_days``.
    """

    # Component choices
    model: str = "Geometric Brownian Motion"
    discretization: str = "Euler"
    payoff: str = "Vanilla"
    option_type: str = "Call"
    sampling: str = "Standard"

    # Common market and numerical inputs
    start_price: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    risk_free_rate: float = 0.05
    volatility: float = 0.20
    num_paths: int = 10_000
    num_steps: int = 252
    runs: int = 1

    # Merton jump-model inputs. Other models ignore these values.
    jump_intensity: float = 0.75
    jump_mean: float = -0.10
    jump_volatility: float = 0.20

    # Hull-White inputs. The first factor uses the common volatility above.
    mean_reversion: float = 0.10
    long_term_mean: float = 0.05

    # Extra inputs used only by the two-factor Hull-White model.
    second_mean_reversion: float = 0.30
    second_volatility: float = 0.01
    factor_correlation: float = -0.70

    # Variance-Gamma parameters used by the Levy Process model.
    levy_skew: float = -0.10
    levy_variance: float = 0.20

    # Asian-payoff input. Other payoffs ignore this value.
    averaging_days: float = 365.0

    def validate_common_inputs(self) -> None:
        """Validate only the values required by every simulation."""
        numbers = {
            "start_price": self.start_price,
            "strike": self.strike,
            "maturity": self.maturity,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
        }
        for name, value in numbers.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if self.start_price <= 0:
            raise ValueError("Start price must be greater than zero.")
        if self.strike <= 0:
            raise ValueError("Strike must be greater than zero.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be greater than zero.")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")
        if not 2 <= self.num_paths <= 1_000_000:
            raise ValueError("Number of paths must be between 2 and 1,000,000.")
        if not isinstance(self.num_paths, int):
            raise ValueError("Number of paths must be a whole number.")
        if self.num_steps < 1:
            raise ValueError("Number of time steps must be at least one.")
        if not isinstance(self.num_steps, int):
            raise ValueError("Number of time steps must be a whole number.")
        if not isinstance(self.runs, int):
            raise ValueError("Number of runs must be a whole number.")
        if self.runs < 1:
            raise ValueError("Number of runs must be at least one.")
        if self.option_type not in ("Call", "Put"):
            raise ValueError("Option type must be Call or Put.")

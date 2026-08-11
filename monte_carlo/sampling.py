"""Sampling methods that generate only the standard-normal random part, Z."""

import warnings

import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri


class StandardSampler:
    """Independent pseudo-random normal values for ordinary Monte Carlo."""

    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.random_generator = np.random.default_rng()

    def draw(self, num_paths):
        return self.random_generator.standard_normal((num_paths, self.num_steps))


class SobolSampler:
    """Sobol points transformed from uniform values to normal values."""

    def __init__(self, num_steps, scrambled):
        self.sequence = qmc.Sobol(d=num_steps, scramble=scrambled)
        if not scrambled:
            # The first non-scrambled point is zero and norm.ppf(0) is -inf.
            self.sequence.fast_forward(1)

    def draw(self, num_paths):
        # Arbitrary path counts work, although powers of two have the strongest
        # Sobol balance properties.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The balance properties.*")
            uniform_values = self.sequence.random(num_paths)

        epsilon = np.finfo(float).eps
        uniform_values = np.clip(uniform_values, epsilon, 1.0 - epsilon)
        return ndtri(uniform_values)


# Each value creates one stateful stream that remains active across batches.
SAMPLERS = {
    "Standard": lambda num_steps: StandardSampler(num_steps),
    "Quasi": lambda num_steps: SobolSampler(num_steps, scrambled=False),
    "Quasi Random": lambda num_steps: SobolSampler(num_steps, scrambled=True),
}


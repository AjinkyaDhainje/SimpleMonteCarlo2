# All the different models and path Generation happens here

import math

import numpy as np

from .discretizations import DISCRETIZATIONS
from .sampling import SAMPLERS

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
    print("Running using numba")
except ImportError:
    print("Running without numba")
    _NUMBA_AVAILABLE = False

# Repeat each value twice : [1, 2, 3] => [1, 1, 2, 2, 3, 3]
def _paired_values(first_values, path_count):
    """Repeat each value for one antithetic pair, with one spare if needed."""
    values = np.empty(path_count)
    pair_count = path_count // 2
    values[0 : 2 * pair_count : 2] = first_values[:pair_count]
    values[1 : 2 * pair_count : 2] = first_values[:pair_count]
    if path_count % 2 == 1:
        values[-1] = first_values[-1]
    return values

# Only used in merton jump 
def _paired_normals(random_generator, path_count):
    """Return independent normal values in adjacent Z, -Z pairs."""
    first_values = random_generator.standard_normal((path_count + 1) // 2)
    values = np.empty(path_count)
    pair_count = path_count // 2
    values[0 : 2 * pair_count : 2] = first_values[:pair_count]
    values[1 : 2 * pair_count : 2] = -first_values[:pair_count]
    if path_count % 2 == 1:
        values[-1] = first_values[-1]
    return values


# ---------------------------------------------------------------------------
# JIT-compiled inner loops (only defined when numba is installed).
# Each function processes all paths in parallel (prange) and all steps
# sequentially within each path, giving cache-friendly row-major access and
# eliminating temporary numpy arrays for drift/diffusion.
# The first run after installation triggers JIT compilation; subsequent runs
# use the cached binary (cache=True).
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    @_numba.njit(cache=True, parallel=True)
    def _gbm_loop(paths, z, r, sigma, dt, sqrt_dt, use_milstein):
        r_dt = r * dt
        sigma_sqrt_dt = sigma * sqrt_dt
        sigma_sq_half_dt = 0.5 * sigma * sigma * dt
        n_steps = paths.shape[1] - 1
        for i in _numba.prange(paths.shape[0]):
            for j in range(n_steps):
                s = paths[i, j]
                zv = z[i, j]
                val = s + r_dt * s + sigma_sqrt_dt * s * zv
                if use_milstein:
                    val += sigma_sq_half_dt * s * (zv * zv - 1.0)
                paths[i, j + 1] = val if val > 0.0 else 0.0

    @_numba.njit(cache=True, parallel=True)
    def _abm_loop(paths, z, drift_step, diff_step):
        n_steps = paths.shape[1] - 1
        for i in _numba.prange(paths.shape[0]):
            for j in range(n_steps):
                paths[i, j + 1] = paths[i, j] + drift_step + diff_step * z[i, j]

    @_numba.njit(cache=True, parallel=True)
    def _hw1f_loop(paths, z, a_dt, b, sigma_sqrt_dt):
        n_steps = paths.shape[1] - 1
        for i in _numba.prange(paths.shape[0]):
            for j in range(n_steps):
                r = paths[i, j]
                paths[i, j + 1] = r + a_dt * (b - r) + sigma_sqrt_dt * z[i, j]

    @_numba.njit(cache=True, parallel=True)
    def _hw2f_loop(paths, z1, z2_corr, initial_x, a_dt, sigma1_sqrt_dt, b2_dt, sigma2_sqrt_dt, b):
        n_steps = paths.shape[1] - 1
        for i in _numba.prange(paths.shape[0]):
            x = initial_x
            y = 0.0
            for j in range(n_steps):
                x_new = x - a_dt * x + sigma1_sqrt_dt * z1[i, j]
                y_new = y - b2_dt * y + sigma2_sqrt_dt * z2_corr[i, j]
                x = x_new
                y = y_new
                paths[i, j + 1] = b + x + y

    @_numba.njit(cache=True, parallel=True)
    def _merton_loop(paths, z, jump_log_sum, r_comp_dt, sigma_sqrt_dt, sigma_sq_half_dt, use_milstein):
        n_steps = paths.shape[1] - 1
        for i in _numba.prange(paths.shape[0]):
            for j in range(n_steps):
                s = paths[i, j]
                zv = z[i, j]
                val = s + r_comp_dt * s + sigma_sqrt_dt * s * zv
                if use_milstein:
                    val += sigma_sq_half_dt * s * (zv * zv - 1.0)
                jumped = val * math.exp(jump_log_sum[i, j])
                paths[i, j + 1] = jumped if jumped > 0.0 else 0.0

    @_numba.njit(cache=True, parallel=True)
    def _heston_loop(
        paths, variances, z_price, z_variance, r, kappa, theta, xi, dt, sqrt_dt,
        use_milstein
    ):
        """Full-truncation Euler for variance, with the selected price scheme."""
        n_steps = paths.shape[1] - 1
        for i in _numba.prange(paths.shape[0]):
            for j in range(n_steps):
                s = paths[i, j]
                v = variances[i, j]
                v_positive = v if v > 0.0 else 0.0
                sqrt_v = math.sqrt(v_positive)

                # Asset equation: dS = r*S*dt + sqrt(v)*S*dW_s.
                zv = z_price[i, j]
                next_s = s + r * s * dt + sqrt_v * s * sqrt_dt * zv
                if use_milstein:
                    # This is the diagonal price Milstein term. The variance
                    # process itself remains full-truncation Euler.
                    next_s += 0.5 * v_positive * s * dt * (zv * zv - 1.0)
                paths[i, j + 1] = next_s if next_s > 0.0 else 0.0

                # Variance equation: dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW_v.
                next_v = (
                    v
                    + kappa * (theta - v_positive) * dt
                    + xi * sqrt_v * sqrt_dt * z_variance[i, j]
                )
                variances[i, j + 1] = next_v if next_v > 0.0 else 0.0


class DiffusionModel:
    """Shared loop for one-factor equations of the form dX = drift*dt + vol*dW."""

    name = "Base model"
    extra_inputs = []
    allow_negative = False

    def validate(self, inputs):
        """A model overrides this only when it has model-specific inputs."""

    def drift(self, values, inputs):
        raise NotImplementedError

    def diffusion(self, values, inputs):
        raise NotImplementedError

    def diffusion_derivative(self, values, inputs):
        raise NotImplementedError

    def after_step(self, values, dt, inputs, random_generator):
        """Apply optional effects, such as jumps, after the diffusion step."""
        del dt, inputs, random_generator
        if self.allow_negative:
            return values
        return np.maximum(values, 0.0)

    def simulate(self, inputs, normal_draws, discretization, random_generator):
        path_count, step_count = normal_draws.shape
        paths = np.empty((path_count, step_count + 1), dtype=float)
        paths[:, 0] = inputs.start_price
        dt = inputs.maturity / step_count
        sqrt_dt = math.sqrt(dt)

        for step in range(step_count):
            current = paths[:, step]
            next_values = discretization(
                current=current,
                drift=self.drift(current, inputs),
                diffusion=self.diffusion(current, inputs),
                diffusion_derivative=self.diffusion_derivative(current, inputs),
                dt=dt,
                z=normal_draws[:, step],
                sqrt_dt=sqrt_dt,
            )
            paths[:, step + 1] = self.after_step(
                next_values, dt, inputs, random_generator
            )

        return paths


class GeometricBrownianMotion(DiffusionModel):
    """Risk-neutral GBM: dS = r*S*dt + sigma*S*dW."""

    name = "Geometric Brownian Motion"

    def drift(self, prices, inputs):
        return inputs.risk_free_rate * prices

    def diffusion(self, prices, inputs):
        return inputs.volatility * prices

    def diffusion_derivative(self, prices, inputs):
        del prices
        return inputs.volatility

    def simulate(self, inputs, normal_draws, discretization, random_generator):
        if _NUMBA_AVAILABLE:
            path_count, step_count = normal_draws.shape
            paths = np.empty((path_count, step_count + 1), dtype=np.float64)
            paths[:, 0] = inputs.start_price
            dt = inputs.maturity / step_count
            sqrt_dt = math.sqrt(dt)
            use_milstein = discretization.__name__ == "milstein"
            _gbm_loop(paths, np.ascontiguousarray(normal_draws, dtype=np.float64),
                      inputs.risk_free_rate, inputs.volatility, dt, sqrt_dt, use_milstein)
            return paths
        return super().simulate(inputs, normal_draws, discretization, random_generator)


class GBMLocalVolatility(DiffusionModel):
    """Risk-neutral GBM with a power-law local-volatility surface.

    sigma_local(S) = sigma_0 * (S / S_0) ** beta

    ``beta=0`` recovers ordinary GBM. Positive beta increases volatility as
    spot rises; negative beta produces the equity-style inverse relationship.
    """

    name = "GBM LV"
    extra_inputs = [
        {
            "name": "local_volatility_exponent",
            "label": "Local-volatility exponent (beta)",
            "min_value": -2.0,
            "max_value": 2.0,
            "step": 0.05,
            "format": "%.4f",
            "help": "0 gives standard GBM; negative values add downside skew.",
        }
    ]

    def validate(self, inputs):
        if not math.isfinite(inputs.local_volatility_exponent):
            raise ValueError("Local-volatility exponent must be finite.")
        if not -2.0 <= inputs.local_volatility_exponent <= 2.0:
            raise ValueError("Local-volatility exponent must be between -2 and 2.")

    def _local_volatility(self, prices, inputs):
        relative_spot = np.maximum(prices / inputs.start_price, 1.0e-12)
        return inputs.volatility * relative_spot ** inputs.local_volatility_exponent

    def drift(self, prices, inputs):
        return inputs.risk_free_rate * prices

    def diffusion(self, prices, inputs):
        return self._local_volatility(prices, inputs) * prices

    def diffusion_derivative(self, prices, inputs):
        return (
            1.0 + inputs.local_volatility_exponent
        ) * self._local_volatility(prices, inputs)


class ArithmeticBrownianMotion(DiffusionModel):
    """Arithmetic model: dS = r*S0*dt + sigma*S0*dW.

    ``volatility`` is treated as a percentage of the initial price, matching
    the input convention used by GBM. Unlike GBM, the change is additive, so
    the simulated value can be negative.
    """

    name = "Arithmetic Brownian Motion"
    allow_negative = True

    def drift(self, prices, inputs):
        return np.full_like(prices, inputs.risk_free_rate * inputs.start_price)

    def diffusion(self, prices, inputs):
        return np.full_like(prices, inputs.volatility * inputs.start_price)

    def diffusion_derivative(self, prices, inputs):
        return np.zeros_like(prices)

    def simulate(self, inputs, normal_draws, discretization, random_generator):
        if _NUMBA_AVAILABLE:
            path_count, step_count = normal_draws.shape
            paths = np.empty((path_count, step_count + 1), dtype=np.float64)
            paths[:, 0] = inputs.start_price
            dt = inputs.maturity / step_count
            sqrt_dt = math.sqrt(dt)
            r_s0 = inputs.risk_free_rate * inputs.start_price
            sigma_s0 = inputs.volatility * inputs.start_price
            # ABM diffusion_derivative is 0, so Milstein correction vanishes.
            _abm_loop(paths, np.ascontiguousarray(normal_draws, dtype=np.float64),
                      r_s0 * dt, sigma_s0 * sqrt_dt)
            return paths
        return super().simulate(inputs, normal_draws, discretization, random_generator)


class HullWhiteModel(DiffusionModel):
    """One-factor Hull-White: dr = a*(b - r)*dt + sigma*dW.

    ``a`` is the mean-reversion speed and ``b`` is the long-term rate. This is
    the constant-parameter form of Hull-White, often called the Vasicek form.
    """

    name = "Hull-White"
    allow_negative = True
    extra_inputs = [
        {
            "name": "mean_reversion",
            "label": "Mean-reversion speed",
            "min_value": 0.0,
            "step": 0.01,
            "format": "%.4f",
        },
        {
            "name": "long_term_mean",
            "label": "Long-term rate",
            "step": 0.005,
            "format": "%.4f",
        },
    ]

    def validate(self, inputs):
        values = (inputs.mean_reversion, inputs.long_term_mean)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All Hull-White inputs must be finite numbers.")
        if inputs.mean_reversion < 0:
            raise ValueError("Mean-reversion speed cannot be negative.")

    def drift(self, rates, inputs):
        return inputs.mean_reversion * (inputs.long_term_mean - rates)

    def diffusion(self, rates, inputs):
        return np.full_like(rates, inputs.volatility)

    def diffusion_derivative(self, rates, inputs):
        return np.zeros_like(rates)

    def simulate(self, inputs, normal_draws, discretization, random_generator):
        if _NUMBA_AVAILABLE:
            path_count, step_count = normal_draws.shape
            paths = np.empty((path_count, step_count + 1), dtype=np.float64)
            paths[:, 0] = inputs.start_price
            dt = inputs.maturity / step_count
            sqrt_dt = math.sqrt(dt)
            # HW diffusion_derivative is 0, so Milstein correction vanishes.
            _hw1f_loop(paths, np.ascontiguousarray(normal_draws, dtype=np.float64),
                       inputs.mean_reversion * dt, inputs.long_term_mean,
                       inputs.volatility * sqrt_dt)
            return paths
        return super().simulate(inputs, normal_draws, discretization, random_generator)

# This equation does not have the form of diffusion model,
# hence, the separate validate and simulate
class TwoFactorHullWhiteModel:
    """Two-factor Hull-White (G2++ style) short-rate model.

    dx = -a*x*dt + sigma*dW1
    dy = -b*y*dt + eta*dW2
    corr(dW1, dW2) = rho and r = long_term_mean + x + y.

    The first factor starts at ``start_price - long_term_mean`` and the second
    starts at zero, so every returned path begins at ``start_price``.
    """

    name = "2f Hull White"
    extra_inputs = [
        {
            "name": "mean_reversion",
            "label": "First mean-reversion speed",
            "min_value": 0.0,
            "step": 0.01,
            "format": "%.4f",
        },
        {
            "name": "long_term_mean",
            "label": "Long-term rate",
            "step": 0.005,
            "format": "%.4f",
        },
        {
            "name": "second_mean_reversion",
            "label": "Second mean-reversion speed",
            "min_value": 0.0,
            "step": 0.01,
            "format": "%.4f",
        },
        {
            "name": "second_volatility",
            "label": "Second-factor volatility",
            "min_value": 0.0,
            "step": 0.005,
            "format": "%.4f",
        },
        {
            "name": "factor_correlation",
            "label": "Factor correlation",
            "min_value": -1.0,
            "max_value": 1.0,
            "step": 0.05,
            "format": "%.4f",
        },
    ]

    def validate(self, inputs):
        values = (
            inputs.mean_reversion,
            inputs.long_term_mean,
            inputs.second_mean_reversion,
            inputs.second_volatility,
            inputs.factor_correlation,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All two-factor Hull-White inputs must be finite.")
        if inputs.mean_reversion < 0 or inputs.second_mean_reversion < 0:
            raise ValueError("Mean-reversion speeds cannot be negative.")
        if inputs.second_volatility < 0:
            raise ValueError("Second-factor volatility cannot be negative.")
        if not -1.0 <= inputs.factor_correlation <= 1.0:
            raise ValueError("Factor correlation must be between -1 and 1.")

    def simulate(
        self,
        inputs,
        normal_draws,
        discretization,
        random_generator,
        second_normal_draws,
    ):
        del random_generator
        path_count, step_count = normal_draws.shape
        paths = np.empty((path_count, step_count + 1), dtype=float)
        paths[:, 0] = inputs.start_price
        first_factor = np.full(
            path_count, inputs.start_price - inputs.long_term_mean
        )
        second_factor = np.zeros(path_count)
        dt = inputs.maturity / step_count
        sqrt_dt = math.sqrt(dt)

        correlation = inputs.factor_correlation
        correlation_scale = math.sqrt(max(0.0, 1.0 - correlation**2))
        correlated_draws = (
            correlation * normal_draws
            + correlation_scale * second_normal_draws
        )

        if _NUMBA_AVAILABLE:
            _hw2f_loop(
                paths,
                np.ascontiguousarray(normal_draws, dtype=np.float64),
                np.ascontiguousarray(correlated_draws, dtype=np.float64),
                float(inputs.start_price - inputs.long_term_mean),
                inputs.mean_reversion * dt,
                inputs.volatility * sqrt_dt,
                inputs.second_mean_reversion * dt,
                inputs.second_volatility * sqrt_dt,
                inputs.long_term_mean,
            )
            return paths

        for step in range(step_count):
            first_factor = discretization(
                current=first_factor,
                drift=-inputs.mean_reversion * first_factor,
                diffusion=inputs.volatility,
                diffusion_derivative=0.0,
                dt=dt,
                z=normal_draws[:, step],
                sqrt_dt=sqrt_dt,
            )
            second_factor = discretization(
                current=second_factor,
                drift=-inputs.second_mean_reversion * second_factor,
                diffusion=inputs.second_volatility,
                diffusion_derivative=0.0,
                dt=dt,
                z=correlated_draws[:, step],
                sqrt_dt=sqrt_dt,
            )
            paths[:, step + 1] = (
                inputs.long_term_mean + first_factor + second_factor
            )

        return paths


class HestonModel:
    """Heston stochastic-volatility model.

    dS = r*S*dt + sqrt(v)*S*dW_s
    dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW_v
    corr(dW_s, dW_v) = rho

    The common ``volatility`` input is the initial volatility, so v(0) is
    volatility**2. Variance is evolved with full-truncation Euler to prevent
    negative values. If Milstein is selected, the diagonal Milstein correction
    is applied to the asset-price equation while variance still uses the robust
    full-truncation Euler step.
    """

    name = "Heston SV"
    extra_inputs = [
        {
            "name": "heston_mean_reversion",
            "label": "Variance mean-reversion speed (kappa)",
            "min_value": 0.0,
            "step": 0.10,
            "format": "%.4f",
        },
        {
            "name": "heston_long_run_variance",
            "label": "Long-run variance (theta)",
            "min_value": 0.0,
            "step": 0.01,
            "format": "%.4f",
            "help": "For example, 0.04 corresponds to a 20% long-run volatility.",
        },
        {
            "name": "heston_vol_of_vol",
            "label": "Volatility of variance (xi)",
            "min_value": 0.0,
            "step": 0.01,
            "format": "%.4f",
        },
        {
            "name": "heston_correlation",
            "label": "Price/variance correlation (rho)",
            "min_value": -1.0,
            "max_value": 1.0,
            "step": 0.05,
            "format": "%.4f",
        },
    ]

    def validate(self, inputs):
        values = (
            inputs.heston_mean_reversion,
            inputs.heston_long_run_variance,
            inputs.heston_vol_of_vol,
            inputs.heston_correlation,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All Heston inputs must be finite numbers.")
        if inputs.heston_mean_reversion < 0:
            raise ValueError("Heston mean-reversion speed cannot be negative.")
        if inputs.heston_long_run_variance < 0:
            raise ValueError("Heston long-run variance cannot be negative.")
        if inputs.heston_vol_of_vol < 0:
            raise ValueError("Heston volatility of variance cannot be negative.")
        if not -1.0 <= inputs.heston_correlation <= 1.0:
            raise ValueError("Heston correlation must be between -1 and 1.")

    def simulate(
        self, inputs, normal_draws, discretization, random_generator,
        second_normal_draws
    ):
        del random_generator
        path_count, step_count = normal_draws.shape
        paths = np.empty((path_count, step_count + 1), dtype=float)
        variances = np.empty((path_count, step_count + 1), dtype=float)
        paths[:, 0] = inputs.start_price
        variances[:, 0] = inputs.volatility ** 2

        dt = inputs.maturity / step_count
        sqrt_dt = math.sqrt(dt)
        rho = inputs.heston_correlation
        correlation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        variance_draws = rho * normal_draws + correlation_scale * second_normal_draws
        use_milstein = discretization.__name__ == "milstein"

        if _NUMBA_AVAILABLE:
            _heston_loop(
                paths,
                variances,
                np.ascontiguousarray(normal_draws, dtype=np.float64),
                np.ascontiguousarray(variance_draws, dtype=np.float64),
                inputs.risk_free_rate,
                inputs.heston_mean_reversion,
                inputs.heston_long_run_variance,
                inputs.heston_vol_of_vol,
                dt,
                sqrt_dt,
                use_milstein,
            )
            return paths, np.sqrt(variances)

        for step in range(step_count):
            current_price = paths[:, step]
            current_variance = np.maximum(variances[:, step], 0.0)
            sqrt_variance = np.sqrt(current_variance)
            z_price = normal_draws[:, step]

            next_price = discretization(
                current=current_price,
                drift=inputs.risk_free_rate * current_price,
                diffusion=sqrt_variance * current_price,
                diffusion_derivative=sqrt_variance,
                dt=dt,
                z=z_price,
                sqrt_dt=sqrt_dt,
            )
            paths[:, step + 1] = np.maximum(next_price, 0.0)

            next_variance = (
                variances[:, step]
                + inputs.heston_mean_reversion
                * (inputs.heston_long_run_variance - current_variance)
                * dt
                + inputs.heston_vol_of_vol
                * sqrt_variance
                * sqrt_dt
                * variance_draws[:, step]
            )
            variances[:, step + 1] = np.maximum(next_variance, 0.0)

        return paths, np.sqrt(variances)


class MertonJumpModel(DiffusionModel):
    """Merton jump diffusion.

    dS/S = (r - lambda*kappa)*dt + sigma*dW + (J - 1)*dN,
    where log(J) is Normal(jump_mean, jump_volatility^2) and
    kappa = E[J - 1]. The compensation term keeps the drift risk-neutral.
    """

    name = "Merton Jump Model"
    extra_inputs = [
        {
            "name": "jump_intensity",
            "label": "Jump intensity (expected jumps/year)",
            "min_value": 0.0,
            "step": 0.05,
        },
        {
            "name": "jump_mean",
            "label": "Mean log-jump size",
            "step": 0.01,
            "format": "%.4f",
        },
        {
            "name": "jump_volatility",
            "label": "Log-jump volatility",
            "min_value": 0.0,
            "step": 0.01,
            "format": "%.4f",
        },
    ]

    def validate(self, inputs):
        values = (
            inputs.jump_intensity,
            inputs.jump_mean,
            inputs.jump_volatility,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All jump inputs must be finite numbers.")
        if inputs.jump_intensity < 0:
            raise ValueError("Jump intensity cannot be negative.")
        if inputs.jump_volatility < 0:
            raise ValueError("Jump volatility cannot be negative.")

    def drift(self, prices, inputs):
        kappa = math.exp(
            inputs.jump_mean + 0.5 * inputs.jump_volatility**2
        ) - 1.0
        compensated_rate = (
            inputs.risk_free_rate - inputs.jump_intensity * kappa
        )
        return compensated_rate * prices

    def diffusion(self, prices, inputs):
        return inputs.volatility * prices

    def diffusion_derivative(self, prices, inputs):
        del prices
        return inputs.volatility

    def simulate(self, inputs, normal_draws, discretization, random_generator):
        """Override simulate to pre-generate all jump draws before the loop."""
        path_count, step_count = normal_draws.shape
        paths = np.empty((path_count, step_count + 1), dtype=float)
        paths[:, 0] = inputs.start_price
        dt = inputs.maturity / step_count
        sqrt_dt = math.sqrt(dt)

        # Pre-generate all Poisson jump counts: shape (half_count, step_count).
        half_count = (path_count + 1) // 2
        pair_count = path_count // 2
        first_counts = random_generator.poisson(
            inputs.jump_intensity * dt, (half_count, step_count)
        )
        # Pair counts: same count for each antithetic pair.
        jump_counts = np.empty((path_count, step_count))
        jump_counts[0 : 2 * pair_count : 2] = first_counts[:pair_count]
        jump_counts[1 : 2 * pair_count : 2] = first_counts[:pair_count]
        if path_count % 2 == 1:
            jump_counts[-1] = first_counts[-1]

        # Pre-generate all jump-size normals with antithetic pairing per pair.
        first_normals = random_generator.standard_normal((half_count, step_count))
        jump_normals = np.empty((path_count, step_count))
        jump_normals[0 : 2 * pair_count : 2] = first_normals[:pair_count]
        jump_normals[1 : 2 * pair_count : 2] = -first_normals[:pair_count]
        if path_count % 2 == 1:
            jump_normals[-1] = first_normals[-1]

        jump_log_sum = (
            jump_counts * inputs.jump_mean
            + np.sqrt(jump_counts) * inputs.jump_volatility * jump_normals
        )

        if _NUMBA_AVAILABLE:
            kappa = math.exp(inputs.jump_mean + 0.5 * inputs.jump_volatility ** 2) - 1.0
            r_comp = inputs.risk_free_rate - inputs.jump_intensity * kappa
            use_milstein = discretization.__name__ == "milstein"
            _merton_loop(
                paths,
                np.ascontiguousarray(normal_draws, dtype=np.float64),
                np.ascontiguousarray(jump_log_sum, dtype=np.float64),
                r_comp * dt,
                inputs.volatility * sqrt_dt,
                0.5 * inputs.volatility ** 2 * dt,
                use_milstein,
            )
            return paths

        for step in range(step_count):
            current = paths[:, step]
            next_values = discretization(
                current=current,
                drift=self.drift(current, inputs),
                diffusion=self.diffusion(current, inputs),
                diffusion_derivative=self.diffusion_derivative(current, inputs),
                dt=dt,
                z=normal_draws[:, step],
                sqrt_dt=sqrt_dt,
            )
            paths[:, step + 1] = np.maximum(
                next_values * np.exp(jump_log_sum[:, step]), 0.0
            )

        return paths

# Same as two factor: separate validate and simulate
class LevyProcess:
    """Exponential Variance-Gamma Levy process.

    X(dt) = theta*G + sigma*sqrt(G)*Z, where
    G ~ Gamma(shape=dt/nu, scale=nu).
    S(t+dt) = S(t)*exp((r + omega)*dt + X(dt)), where
    omega = log(1 - theta*nu - 0.5*sigma^2*nu)/nu.

    The transition is sampled directly, so Euler and Milstein give the same
    result for this model. The selection remains accepted so the common input
    flow does not need a special case.
    """

    name = "Levy Process"
    extra_inputs = [
        {
            "name": "levy_skew",
            "label": "Levy skew (theta)",
            "step": 0.01,
            "format": "%.4f",
        },
        {
            "name": "levy_variance",
            "label": "Levy variance rate (nu)",
            "min_value": 0.0001,
            "step": 0.01,
            "format": "%.4f",
        },
    ]

    def validate(self, inputs):
        if not math.isfinite(inputs.levy_skew):
            raise ValueError("Levy skew must be finite.")
        if not math.isfinite(inputs.levy_variance):
            raise ValueError("Levy variance rate must be finite.")
        if inputs.levy_variance <= 0:
            raise ValueError("Levy variance rate must be greater than zero.")
        martingale_term = (
            1.0
            - inputs.levy_skew * inputs.levy_variance
            - 0.5 * inputs.volatility**2 * inputs.levy_variance
        )
        if martingale_term <= 0:
            raise ValueError(
                "Levy inputs must satisfy "
                "1 - theta*nu - 0.5*volatility^2*nu > 0."
            )

    def simulate(self, inputs, normal_draws, discretization, random_generator):
        del discretization
        path_count, step_count = normal_draws.shape
        paths = np.empty((path_count, step_count + 1), dtype=float)
        paths[:, 0] = inputs.start_price
        dt = inputs.maturity / step_count
        nu = inputs.levy_variance
        martingale_term = (
            1.0 - inputs.levy_skew * nu - 0.5 * inputs.volatility**2 * nu
        )
        omega = math.log(martingale_term) / nu

        # Generate all gamma draws at once: shape (half_count, step_count).
        half_count = (path_count + 1) // 2
        first_gamma = random_generator.gamma(
            shape=dt / nu,
            scale=nu,
            size=(half_count, step_count),
        )
        # Pair each half-path row with a copy for antithetic pairing.
        pair_count = path_count // 2
        gamma_increments = np.empty((path_count, step_count))
        gamma_increments[0 : 2 * pair_count : 2] = first_gamma[:pair_count]
        gamma_increments[1 : 2 * pair_count : 2] = first_gamma[:pair_count]
        if path_count % 2 == 1:
            gamma_increments[-1] = first_gamma[-1]

        levy_increments = (
            inputs.levy_skew * gamma_increments
            + inputs.volatility * np.sqrt(gamma_increments) * normal_draws
        )
        log_steps = (inputs.risk_free_rate + omega) * dt + levy_increments
        paths[:, 1:] = inputs.start_price * np.exp(np.cumsum(log_steps, axis=1))

        return paths


MODELS = {
    GeometricBrownianMotion.name: GeometricBrownianMotion(),
    ArithmeticBrownianMotion.name: ArithmeticBrownianMotion(),
    HullWhiteModel.name: HullWhiteModel(),
    TwoFactorHullWhiteModel.name: TwoFactorHullWhiteModel(),
    GBMLocalVolatility.name: GBMLocalVolatility(),
    HestonModel.name: HestonModel(),
    MertonJumpModel.name: MertonJumpModel(),
    LevyProcess.name: LevyProcess(),
}

# Autocallables deliberately use a smaller, product-approved model universe.
# Normal options continue to use the complete model registry above.
AUTOCALLABLE_MODELS = {
    name: MODELS[name] for name in (GBMLocalVolatility.name, HestonModel.name)
}


class MonteCarloEngine:
    """Take the inputs and return path sets for the selected model and method."""

    max_points_per_batch = 5_000_000

    def __init__(self, inputs):
        self.inputs = inputs
        self.model = self._get(MODELS, inputs.model, "model")
        self.discretization = self._get(
            DISCRETIZATIONS, inputs.discretization, "discretization"
        )
        sampler_factory = self._get(
            SAMPLERS, inputs.sampling, "sampling method"
        )
        self.sampler = sampler_factory(inputs.num_steps)
        self.random_generator = np.random.default_rng()
        # Populated only for Heston batches. The manager copies the display
        # subset before the next batch is generated.
        self.last_volatility_paths = None

    def batch_size(self):
        """Choose an even batch size so adjacent antithetic paths stay paired."""
        size = min(
            10_000,
            max(2, self.max_points_per_batch // (self.inputs.num_steps + 1)),
        )
        if size % 2 == 1:
            size -= 1
        return size

    def generate_paths(self, path_count):
        """Return one pathSet using the model and discretization in the inputs."""
        normal_draws = self._antithetic_draw(path_count)

        if isinstance(self.model, (TwoFactorHullWhiteModel, HestonModel)):
            second_normal_draws = self._antithetic_draw(path_count)
            simulation_output = self.model.simulate(
                self.inputs,
                normal_draws,
                self.discretization,
                self.random_generator,
                second_normal_draws,
            )
            if isinstance(self.model, HestonModel):
                paths, self.last_volatility_paths = simulation_output
                return paths
            return simulation_output

        self.last_volatility_paths = None
        return self.model.simulate(
            self.inputs,
            normal_draws,
            self.discretization,
            self.random_generator,
        )

    def _antithetic_draw(self, path_count):
        """Generate Z for half the paths and use -Z for their paired paths."""
        first_count = (path_count + 1) // 2
        first_draws = self.sampler.draw(first_count)
        draws = np.empty((path_count, self.inputs.num_steps))
        pair_count = path_count // 2
        draws[0 : 2 * pair_count : 2] = first_draws[:pair_count]
        draws[1 : 2 * pair_count : 2] = -first_draws[:pair_count]
        if path_count % 2 == 1:
            draws[-1] = first_draws[-1]
        return draws

    @staticmethod
    def _get(registry, selected_name, component_name):
        try:
            return registry[selected_name]
        except KeyError as error:
            choices = ", ".join(registry)
            raise ValueError(
                f"Unknown {component_name} '{selected_name}'. "
                f"Choose from: {choices}."
            ) from error

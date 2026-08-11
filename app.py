"""Streamlit interface for the simple configurable Monte Carlo engine."""

import matplotlib.pyplot as plt
import streamlit as st

from monte_carlo import (
    DISCRETIZATIONS,
    MODELS,
    PAYOFFS,
    SAMPLERS,
    SimulationInputs,
    SimulationManager,
)
from monte_carlo.charts import (
    path_chart,
    payoff_chart,
    terminal_price_chart,
    price_convergence_chart,
    variance_mean_convergence_chart,
    volatility_path_chart,
)


def extra_number_inputs(component, defaults, maturity):
    """Display inputs declared by the selected model or payoff."""
    values = {}
    for field in component.extra_inputs:
        default_value = float(getattr(defaults, field["name"]))
        options = {
            "label": field["label"],
            "value": default_value,
            "step": field.get("step", 0.01),
            "key": f"extra_{field['name']}",
        }
        if field.get("limited_by_maturity"):
            maximum = float(maturity) * 365.0
            options["max_value"] = maximum
            options["value"] = min(default_value, maximum)
        for optional_setting in ("min_value", "max_value", "format", "help"):
            if optional_setting in field:
                options[optional_setting] = field[optional_setting]
        values[field["name"]] = float(st.number_input(**options))
    return values


st.set_page_config(page_title="Simple Monte Carlo Engine", layout="wide")
st.title("Configurable Monte Carlo Option-Pricing Engine")
st.caption("A small, readable implementation designed to be extended.")

defaults = SimulationInputs()

with st.sidebar:
    st.header("Simulation inputs")

    st.subheader("Market and option")
    start_price = st.number_input("Start price", min_value=0.01, value=100.0)
    strike = st.number_input("Strike", min_value=0.01, value=100.0)
    maturity = st.number_input(
        "Maturity (years)", min_value=0.01, value=1.0, step=0.25
    )
    risk_free_rate = st.number_input(
        "Risk-free rate", value=0.05, step=0.005, format="%.4f"
    )
    volatility = st.number_input(
        "Volatility", min_value=0.0, value=0.20, step=0.01, format="%.4f"
    )

    st.subheader("Components")
    model_name = st.selectbox("Model", list(MODELS))
    # Streamlit reruns on selection, so these fields immediately appear below
    # the model that requires them.
    model_values = extra_number_inputs(MODELS[model_name], defaults, maturity)

    discretization = st.selectbox("Discretization", list(DISCRETIZATIONS))

    payoff_name = st.selectbox("Payoff", list(PAYOFFS))
    payoff_values = extra_number_inputs(PAYOFFS[payoff_name], defaults, maturity)

    option_type = st.selectbox("Option type", ["Call", "Put"])
    sampling = st.selectbox("Sampling type", list(SAMPLERS))

    st.subheader("Numerical inputs")
    num_paths = st.number_input(
        "Number of paths",
        min_value=1,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
    )
    num_steps = st.number_input(
        "Time steps", min_value=1, max_value=300, value=100, step=50
    )
    runs = st.number_input(
        "Runs", min_value=1, max_value=100, value=1, step=1
    )
    run_button = st.button("Run simulation", type="primary")

if not run_button:
    st.info("Choose the inputs in the sidebar and click **Run simulation**.")
    st.stop()

try:
    inputs = SimulationInputs(
        model=model_name,
        discretization=discretization,
        payoff=payoff_name,
        option_type=option_type,
        sampling=sampling,
        start_price=float(start_price),
        strike=float(strike),
        maturity=float(maturity),
        risk_free_rate=float(risk_free_rate),
        volatility=float(volatility),
        num_paths=int(num_paths),
        num_steps=int(num_steps),
        runs=int(runs),
        **model_values,
        **payoff_values,
    )
    with st.spinner("Running the simulation..."):
        multi_run_result = SimulationManager().run_multiple(inputs)
except (ValueError, MemoryError) as error:
    st.error(f"Simulation could not be completed: {error}")
    st.stop()

result = multi_run_result.final_run
st.subheader("Option prices by run")
st.table(
    [
        {"Run": run_number, "Final option price": f"{price:.4f}"}
        for run_number, price in enumerate(multi_run_result.option_prices, start=1)
    ]
)

low, high = result.confidence_interval
columns = st.columns(4)
columns[0].metric(
    "Average final option price", f"{multi_run_result.average_option_price:.4f}"
)
columns[1].metric(
    "Final run 95% confidence interval low", f"{low:.4f}"
)
columns[2].metric("high", f"{high:.4f}")
columns[3].metric(
    "Total simulation time", f"{multi_run_result.elapsed_seconds:.3f} s"
)

# Each chart receives only the data it needs. The UI never receives the full
# path matrix: result.display_paths contains at most 10,000 complete paths.
st.subheader("Final run charts")
figures = [
    path_chart(result.display_paths, result.time_grid, inputs.strike),
    terminal_price_chart(result.terminal_prices),
    payoff_chart(result.discounted_payoffs, result.option_price),
    price_convergence_chart(result.discounted_payoffs, result.option_price),
    variance_mean_convergence_chart(result.discounted_payoffs),
]
tab_names = [
    "Paths",
    "Final prices",
    "Payoffs",
    "Price convergence",
    "Variance of Mean",
]

# Heston produces a second path set for stochastic volatility. Keep this graph
# out of the UI for every other model.
if result.display_volatility_paths is not None:
    figures.append(
        volatility_path_chart(result.display_volatility_paths, result.time_grid)
    )
    tab_names.append("Volatility")

for tab, figure in zip(st.tabs(tab_names), figures):
    with tab:
        st.pyplot(figure, width="stretch")
        plt.close(figure)

# Run this file directly
# It will run the simulation for an increasing number of paths.
# Paths start from 10 and go upto a million
# It compares the price given by the simulation for each different number of paths
# Also shows a plot with the error

import matplotlib.pyplot as plt
import numpy as np

from monte_carlo import (
    DISCRETIZATIONS,
    MODELS,
    MonteCarloEngine,
    PAYOFFS,
    SimulationInputs,
    SimulationManager,
)

manager = SimulationManager()

blackScholesPrice = 10.451

# The final MC_price returned by the simulation is stored in this
resultArr = []
# Each MC_price's absolute difference from the black Scholes price multiplied by 10
diffs = []
# X Axis, each number maps to index+1 of paths


paths = [10000, 100000, 1000000]
samplers = ["Standard", "Quasi Random"]

for i in samplers:
    config = SimulationInputs(
    model= 'Geometric Brownian Motion',
    discretization= 'Euler',
    payoff= 'Vanilla',
    option_type= 'Call',
    sampling= i,
    start_price=100.0, strike=100.0, maturity=1.0, risk_free_rate=0.05, volatility=0.2,
    num_paths=1000000, num_steps= 100
    )
    for i in range(30):
        result = manager.run(config)
        price = result.option_price
        resultArr.append(price)
        # diffs.append(abs(blackScholesPrice - price) * 10)

print (resultArr, diffs)

# Data for plotting

array_y = np.array(resultArr)
xaxis = []
for i in range(0, len(array_y)):
    xaxis.append(i)
array_x = np.array(xaxis)

plt.plot(array_x, array_y, marker='o', linestyle='-', color='b', label='Data Points')

plt.title("1-to-1 Index Array Plot")
plt.xlabel("X Array Values")
plt.ylabel("Y Array Values")
plt.grid(True)
plt.legend()
plt.show()

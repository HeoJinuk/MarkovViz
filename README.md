# MarkovViz

A Python library for implementing and visualizing Markov models with a focus on reinforcement learning.

## Features

- **Multiple Input Formats**: Support for NumPy arrays, Pandas DataFrames, Python dictionaries, and lists
- **Markov Chain Implementation**: Calculate state convergence with customizable iterations and thresholds
- **Markov Reward Process**: Evaluate state values using linear equations or dynamic programming
- **Visualization**: Generate clear graphical representations of:
  - State transition probabilities
  - State distributions
  - Rewards and state values
  - Combined views for reinforcement learning analysis

## Installation

```bash
pip install markovviz  # Note: Update with actual package name
```

## Quick Start

### Basic Markov Chain

```python
import numpy as np
from markov import MarkovChain, PlotMarkov

# Define transition matrix
transitions = np.array([
    [0.7, 0.3, 0.0],
    [0.0, 0.5, 0.5],
    [0.3, 0.0, 0.7]
])

# Create MarkovChain instance
mc = MarkovChain(transitions, node_names=['A', 'B', 'C'])

# Visualize
plot = PlotMarkov(mc)
graph = plot.draw_graph()
graph.render('markov_chain', view=True, format='png')
```

### Markov Reward Process

```python
from markov import MarkovRewardProcess

# Define transitions and rewards
transitions = {
    'Start': {'Left': 0.4, 'Right': 0.6},
    'Left': {'Start': 0.3, 'End': 0.7},
    'Right': {'Start': 0.2, 'End': 0.8},
    'End': {'End': 1.0}
}

rewards = {
    'Start': 0,
    'Left': 1,
    'Right': 4,
    'End': 10
}

# Create MRP instance
mrp = MarkovRewardProcess(transitions, rewards, gamma=0.9)

# Calculate state values
values = mrp.evaluate_by_linear_equation()

# Visualize with rewards and values
plot = PlotMarkov(mrp)
graph = plot.draw_graph_with_rewards_and_values()
graph.render('markov_reward_process', view=True, format='png')
```

## Class Overview

### Markov (Base Class)
- Handles different input formats for transition matrices
- Provides basic data structure and validation

### MarkovChain
- Implements basic Markov Chain functionality
- Supports state probability convergence analysis
- Calculates steady-state probabilities

### MarkovRewardProcess
- Implements Markov Reward Process calculations
- Supports both linear equation and dynamic programming approaches
- Calculates state values with configurable discount factor

### PlotMarkov
- Generates graphical visualizations using Graphviz
- Supports multiple visualization modes:
  - Basic state transitions
  - State probabilities
  - Rewards
  - State values
  - Combined views

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Graphviz

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{markovviz2024,
  title = {MarkovViz: A Python Library for Markov Models and Reinforcement Learning Visualization},
  year = {2024},
  // Add other relevant citation information
}
```

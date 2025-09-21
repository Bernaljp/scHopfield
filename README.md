# scHopfield

A modular Python package for single-cell trajectory analysis using Hopfield-like dynamics.

## Overview

scHopfield provides a comprehensive framework for analyzing single-cell trajectories using energy landscape approaches inspired by Hopfield networks. The package is designed with modularity in mind, offering flexible components for data processing, optimization, simulation, and visualization.

## Features

- **Modular Architecture**: Well-structured components for different aspects of trajectory analysis
- **Energy Landscape Analysis**: Tools for computing and analyzing energy landscapes
- **Trajectory Optimization**: Advanced optimization algorithms for trajectory inference
- **Simulation Framework**: Comprehensive simulation tools for hypothesis testing
- **Visualization Suite**: Rich plotting and visualization capabilities
- **Type Safety**: Full type hints for better development experience

## Installation

### From Source

```bash
git clone https://github.com/schopfield/scHopfield.git
cd scHopfield
pip install -e .
```

### Development Installation

For development with all optional dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import scHopfield as sch
import numpy as np

# Load your single-cell data
data = sch.load_data("your_data.h5ad")

# Create an analyzer
analyzer = sch.LandscapeAnalyzer(data)

# Compute energy landscape
landscape = analyzer.compute_landscape()

# Visualize results
sch.plot_landscape(landscape)
```

## Package Structure

```
scHopfield/
├── core/           # Core functionality and base classes
├── analysis/       # Analysis algorithms and methods
├── optimization/   # Optimization algorithms
├── simulation/     # Simulation framework
├── utils/          # Utility functions and helpers
└── visualization/  # Plotting and visualization tools
```

## Core Components

### Analysis Module
- `LandscapeAnalyzer`: Main class for energy landscape analysis
- `TrajectoryAnalyzer`: Tools for trajectory-specific analysis
- Various specialized analyzers for different data types

### Optimization Module
- `EnergyOptimizer`: Energy-based optimization algorithms
- `TrajectoryOptimizer`: Specialized trajectory optimization
- Custom optimization strategies

### Simulation Module
- `DynamicsSimulator`: Simulate trajectory dynamics
- `EnergySimulator`: Energy landscape simulations
- Monte Carlo and deterministic simulation methods

### Visualization Module
- `LandscapePlotter`: Energy landscape visualization
- `TrajectoryPlotter`: Trajectory-specific plots
- Interactive plotting capabilities

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- AnnData >= 0.8.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0

## Documentation

Full documentation is available at [https://schopfield.readthedocs.io](https://schopfield.readthedocs.io)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use scHopfield in your research, please cite:

```bibtex
@software{scHopfield,
  title={scHopfield: A modular package for single-cell trajectory analysis},
  author={scHopfield Team},
  url={https://github.com/schopfield/scHopfield},
  year={2024}
}
```

## Support

- Documentation: [https://schopfield.readthedocs.io](https://schopfield.readthedocs.io)
- Issues: [https://github.com/schopfield/scHopfield/issues](https://github.com/schopfield/scHopfield/issues)
- Discussions: [https://github.com/schopfield/scHopfield/discussions](https://github.com/schopfield/scHopfield/discussions)
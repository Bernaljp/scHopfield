#!/usr/bin/env python3
"""
Test script to verify all scHopfield module imports work correctly.
Run this script from the scHopfield directory to test all module imports.
"""

print('Testing scHopfield module imports...')
print('=' * 50)

# Test core imports
try:
    from core.base_models import BaseAnalyzer, BaseEnergyCalculator, BaseOptimizer, BaseSimulator
    from core.data_processing import DataProcessor
    print('✓ Core modules imported successfully')
except Exception as e:
    print(f'✗ Core module import error: {e}')

# Test analysis imports
try:
    from analysis import LandscapeAnalyzer, EnergyCalculator
    print('✓ Analysis modules imported successfully')
except Exception as e:
    print(f'✗ Analysis module import error: {e}')

# Test optimization imports
try:
    from optimization import ScaffoldOptimizer, EnergyOptimizer, TrajectoryOptimizer
    print('✓ Optimization modules imported successfully')
except Exception as e:
    print(f'✗ Optimization module import error: {e}')

# Test simulation imports
try:
    from simulation import ODESolver, DynamicsSimulator, AttractorAnalyzer
    print('✓ Simulation modules imported successfully')
except Exception as e:
    print(f'✗ Simulation module import error: {e}')

# Test visualization imports
try:
    from visualization import EnergyPlotter, TrajectoryPlotter, LandscapePlotter
    print('✓ Visualization modules imported successfully')
except Exception as e:
    print(f'✗ Visualization module import error: {e}')

# Test utils imports
try:
    from utils.utilities import sigmoid, fit_sigmoid, to_numpy
    print('✓ Utility functions imported successfully')
except Exception as e:
    print(f'✗ Utility function import error: {e}')

print('\n' + '=' * 50)
print('Module import testing completed!')

# Test that main package import works when installed
print('\nTesting main package import...')
try:
    import sys
    sys.path.insert(0, '..')  # Add parent directory to path
    import scHopfield
    print('✓ Main scHopfield package imported successfully')

    # Test accessing some main classes
    print('Available classes in scHopfield:')
    for attr in dir(scHopfield):
        if not attr.startswith('_'):
            print(f'  - {attr}')

except Exception as e:
    print(f'✗ Main package import error: {e}')

print('\nAll tests completed!')
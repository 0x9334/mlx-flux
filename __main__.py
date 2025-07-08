#!/usr/bin/env python3
"""
MLX-Flux package entry point
Allows running: python -m mlx_flux
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cli import cli
except ImportError as e:
    print(f"Error importing CLI: {e}")
    print("Make sure all dependencies are installed: pip install -e .")
    sys.exit(1)

if __name__ == '__main__':
    cli() 
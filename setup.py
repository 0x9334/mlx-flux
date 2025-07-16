#!/usr/bin/env python3
"""
MLX-Flux Setup Script
"""

from version import __version__
from setuptools import setup, find_packages
import os

# Read the README file for the long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "MLX-Flux: FastAPI server for Flux image generation using MLX"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="mlx-flux",
    version=__version__,
    author="EternalAI",
    description="FastAPI server for Flux image generation using MLX",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mlx-flux",
    packages=find_packages(),
    py_modules=[
        "app",
        "flux", 
        "schema",
        "cli"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements() + ["click>=8.0.0"],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.950",
        ],
        "test": [
            "pytest>=6.0.0",
            "httpx>=0.23.0",
            "pytest-asyncio>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlx-flux=cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="mlx flux image generation fastapi ai ml",
    project_urls={
        "Bug Reports": "https://github.com/your-username/mlx-flux/issues",
        "Source": "https://github.com/your-username/mlx-flux",
        "Documentation": "https://github.com/your-username/mlx-flux#readme",
    },
) 
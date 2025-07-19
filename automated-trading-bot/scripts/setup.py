#!/usr/bin/env python3
"""
Setup script for Automated Trading Bot
Handles installation with proper dependency management
"""

import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages


def check_talib_c_library():
    """Check if TA-Lib C library is installed"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Check if TA-Lib is installed via Homebrew
        try:
            result = subprocess.run(['brew', 'list', 'ta-lib'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            print("Homebrew not found. Please install Homebrew first.")
            return False
            
    elif system == "Linux":
        # Check common locations for TA-Lib
        lib_paths = ['/usr/lib/libta_lib.so', '/usr/local/lib/libta_lib.so']
        return any(os.path.exists(path) for path in lib_paths)
    
    return False


def install_talib_c_library():
    """Provide instructions for installing TA-Lib C library"""
    system = platform.system()
    
    print("\n" + "="*60)
    print("TA-Lib C Library Installation Required")
    print("="*60)
    
    if system == "Darwin":  # macOS
        print("\nFor macOS, run:")
        print("  brew install ta-lib")
        
    elif system == "Linux":
        print("\nFor Ubuntu/Debian, run:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y ta-lib")
        print("\nFor other Linux distributions, install from source:")
        print("  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
        print("  tar -xzf ta-lib-0.4.0-src.tar.gz")
        print("  cd ta-lib/")
        print("  ./configure --prefix=/usr")
        print("  make")
        print("  sudo make install")
    
    print("\nAfter installing TA-Lib C library, run:")
    print("  pip install -r requirements.txt")
    print("\n" + "="*60)


# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]


# Separate requirements
core_requirements = [
    'pandas>=2.0.0',
    'numpy>=1.24.0',
    'scipy>=1.10.0',
    'scikit-learn>=1.2.0',
    'sqlalchemy>=2.0.0',
    'fastapi>=0.95.0',
    'uvicorn>=0.21.0',
    'requests>=2.28.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'click>=8.1.0',
    'python-dotenv>=1.0.0',
    'pydantic>=1.10.0',
    'loguru>=0.7.0',
]

# Optional requirements that might fail
optional_requirements = [
    'TA-Lib>=0.4.28',  # Requires C library
    'pandas-ta>=0.3.14b0',  # Alternative to TA-Lib
]


if __name__ == "__main__":
    # Check if TA-Lib C library is installed
    if not check_talib_c_library():
        install_talib_c_library()
        print("\nInstalling without TA-Lib for now...")
        print("Using pandas-ta as alternative")
        # Remove TA-Lib from requirements
        install_requires = core_requirements + ['pandas-ta>=0.3.14b0']
    else:
        install_requires = core_requirements + optional_requirements
    
    setup(
        name="automated-trading-bot",
        version="2.0.0",
        author="Trading Bot Team",
        description="Automated Trading Bot with Enhanced Indicators",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/your-org/automated-trading-bot",
        packages=find_packages(),
        install_requires=install_requires,
        extras_require={
            'dev': [
                'pytest>=7.3.0',
                'pytest-asyncio>=0.21.0',
                'pytest-cov>=4.0.0',
                'black>=23.3.0',
                'flake8>=6.0.0',
                'mypy>=1.2.0',
            ],
            'ml': [
                'tensorflow>=2.12.0',
                'lightgbm>=3.3.0',
                'xgboost>=1.7.0',
            ],
        },
        python_requires='>=3.8',
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Financial and Insurance Industry",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        entry_points={
            'console_scripts': [
                'trading-bot=src.main:main',
                'bot-optimize=run_optimization:main',
            ],
        },
    )
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure")
    print("2. Run 'python main.py --help' to see options")
    print("3. Run 'python run_optimization.py' to optimize parameters")
    print("4. Start with paper trading mode")
    print("\nFor full documentation, see docs/PROJECT_STRUCTURE.md")
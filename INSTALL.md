# Astro-AI Environment Setup Guide
# Based on CAM-SUST Summer School Installation Guide

## Option 1: Using pip (Current approach)
```bash
pip install -r requirements.txt
```

## Option 2: Using conda (Recommended for astronomical packages)
```bash
# Create environment
conda create -n astro-ai python=3.10
conda activate astro-ai

# Install conda packages first
conda install -c conda-forge numpy pandas matplotlib scipy astropy h5py
conda install -c conda-forge streamlit seaborn

# Install pip packages
pip install bagpipes
pip install openai>=0.27.0
pip install pytest>=7.0.0 black>=23.0.0
```

## Option 3: Full astronomical setup (Advanced)
```bash
# Create environment for full astronomy stack
conda create -n astro-ai-full python=3.10
conda activate astro-ai-full

# Install system dependencies (if on Linux/WSL)
# sudo apt-get install libfftw3-dev libgsl-dev build-essential

# Install conda packages
conda install -c conda-forge cython numpy pandas matplotlib scipy astropy h5py
conda install -c conda-forge streamlit seaborn

# Install specialized astronomy packages
pip install bagpipes
pip install tools21cm
pip install nautilus-sampler

# Optional: Install 21cmFAST (Linux/WSL only)
# pip install git+https://github.com/21cmfast/21cmFAST.git

# Optional: Install JWST pipeline (requires C++ compiler)
# conda install -c conda-forge jwst

# Install remaining packages
pip install openai>=0.27.0
pip install pytest>=7.0.0 black>=23.0.0
```

## Environment Files

For reproducible installations, you can also create:
- environment.yml (for conda)
- requirements.txt (for pip)
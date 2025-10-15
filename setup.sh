#!/bin/bash
# setup.sh - Pre-installation setup script for Streamlit Community Cloud
#
# This script runs before pip installs Python packages, ensuring the Linux
# environment is properly configured for scientific computing and astronomy.
# 
# Streamlit Cloud executes this automatically when deploying the app.

set -e  # Exit on any error

echo "🚀 Setting up Astro-AI deployment environment..."

# ========================================
# ENVIRONMENT VARIABLES
# ========================================

# Configure scientific computing optimizations
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Set up Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Configure matplotlib for headless operation (no GUI)
export MPLBACKEND=Agg

# ========================================
# SYSTEM PACKAGE VERIFICATION
# ========================================

echo "📦 Verifying system packages installation..."

# Check critical libraries are available
if ! pkg-config --exists fftw3; then
    echo "⚠️  Warning: FFTW3 not found - py21cmfast will not be available"
fi

if ! pkg-config --exists gsl; then
    echo "⚠️  Warning: GSL not found - advanced simulations may be limited"  
fi

if ! pkg-config --exists hdf5; then
    echo "⚠️  Warning: HDF5 not found - some data formats may not be supported"
fi

# ========================================
# DIRECTORY SETUP
# ========================================

echo "📁 Creating application directories..."

# Create cache directories for astronomical data
mkdir -p ~/.astropy/cache
mkdir -p ~/.matplotlib/cache

# Create temporary directories for processing
mkdir -p /tmp/astro-ai-workspace
mkdir -p /tmp/astro-ai-plots

# ========================================
# PERMISSIONS & SECURITY
# ========================================

echo "🔒 Setting up permissions..."

# Ensure cache directories are writable
chmod 755 ~/.astropy/cache
chmod 755 ~/.matplotlib/cache

# Set up secure temp directories
chmod 755 /tmp/astro-ai-workspace
chmod 755 /tmp/astro-ai-plots

# ========================================
# SCIENTIFIC LIBRARY CONFIGURATION
# ========================================

echo "⚙️  Configuring scientific libraries..."

# Configure NumPy/SciPy for optimized performance
cat > ~/.numpy-site.cfg << EOF
[DEFAULT]
library_dirs = /usr/lib/x86_64-linux-gnu
include_dirs = /usr/include

[openblas]
libraries = openblas
library_dirs = /usr/lib/x86_64-linux-gnu
include_dirs = /usr/include/openblas

[fftw]
libraries = fftw3
library_dirs = /usr/lib/x86_64-linux-gnu
include_dirs = /usr/include
EOF

# ========================================
# ASTRONOMICAL DATA SETUP
# ========================================

echo "🌌 Setting up astronomical data environment..."

# Set astropy data download location (if needed)
export ASTROPY_CACHE_DIR=~/.astropy/cache

# Configure any FITS file optimizations
export FITS_CACHE_SIZE=128

# ========================================
# MEMORY & PERFORMANCE OPTIMIZATION
# ========================================

echo "💾 Configuring memory optimizations..."

# Set memory-efficient defaults for scientific computing
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Configure garbage collection for large datasets
export PYTHONGC=1

# ========================================
# LOGGING CONFIGURATION
# ========================================

echo "📝 Setting up logging..."

# Configure application logging level
export ASTRO_AI_LOG_LEVEL=INFO

# Reduce verbose output from scientific libraries during startup
export ASTROPY_SILENCE_WARNINGS=1

# ========================================
# FINAL VERIFICATION
# ========================================

echo "✅ Verifying setup completion..."

# Test Python is available
python3 --version

# Test critical libraries can be found
echo "🔍 Checking library availability:"
ldconfig -p | grep -E "(fftw|gsl|hdf5|openblas)" || echo "Some optional libraries not found"

echo ""
echo "🎯 Astro-AI environment setup complete!"
echo "   Platform: Linux (Streamlit Community Cloud)"
echo "   Python: $(python3 --version)"
echo "   Ready for pip package installation..."
echo ""

# ========================================
# NOTES FOR MAINTENANCE
# ========================================
#
# This script prepares the Linux environment for:
# 
# 1. CORE SCIENTIFIC COMPUTING
#    - NumPy/SciPy with optimized BLAS
#    - Matplotlib in headless mode
#    - Efficient memory usage
#
# 2. ASTRONOMICAL PACKAGES  
#    - Astropy with proper caching
#    - FITS file support via CFITSIO
#    - HDF5 data format support
#
# 3. OPTIONAL HEAVY PACKAGES
#    - py21cmfast (if FFTW/GSL available)
#    - JWST pipeline (if dependencies met)
#    - Bagpipes SED fitting
#
# 4. PERFORMANCE OPTIMIZATION
#    - Single-threaded operation (Cloud CPU limits)
#    - Memory-efficient garbage collection
#    - Reduced library verbosity
#
# The app will gracefully fall back to mock modes if heavy
# packages cannot be installed, ensuring reliability.
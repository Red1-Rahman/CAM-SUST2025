# 🌌 Astro-AI: Streamlit Cloud Deployment Guide

This directory contains deployment configuration files optimized for **Streamlit Community Cloud** running in Linux containers.

## 📋 Deployment Files Overview

### `requirements.txt` - Python Dependencies

**Purpose**: Specifies all Python packages needed for the application
**Optimizations**:

- ✅ **Linux-compatible**: All packages work in Ubuntu/Debian containers
- ✅ **Lightweight core**: Essential packages only, with optional heavy deps commented
- ✅ **Version pinning**: Stable versions that work together
- ✅ **Fallback ready**: App works even if optional packages fail to install

**Key packages**:

- `streamlit` - Web framework
- `numpy`, `pandas`, `scipy` - Scientific computing core
- `matplotlib`, `seaborn` - Visualization
- `astropy` - Astronomy calculations
- `bagpipes` - SED fitting (lightweight astronomy package)
- `openai` - AI integration

### `packages.txt` - System Dependencies

**Purpose**: Linux system packages installed via `apt-get` before Python packages
**Critical for**:

- 🔬 **Scientific libraries**: FFTW, GSL, BLAS for mathematical computing
- 📊 **Data formats**: HDF5, CFITSIO for astronomical file support
- 🛠️ **Build tools**: Compilers needed for packages with C/Fortran extensions
- 🖼️ **Graphics**: Libraries for matplotlib plotting

### `setup.sh` - Environment Configuration

**Purpose**: Bash script that runs before package installation
**Configures**:

- 🔧 **Environment variables**: Optimizes scientific computing for Cloud limits
- 📁 **Directories**: Creates cache and workspace folders
- ⚡ **Performance**: Single-threaded operation, memory optimization
- 🔇 **Logging**: Reduces verbose startup messages

## 🚀 Deployment Strategy

### Tier-Based Architecture

The platform supports multiple deployment tiers:

| Tier              | Packages            | Use Case                    |
| ----------------- | ------------------- | --------------------------- |
| **Minimal**       | Core only           | Demo, UI testing            |
| **Science-Light** | + bagpipes          | SED fitting, basic analysis |
| **Full**          | + py21cmfast + jwst | Complete research platform  |

### Streamlit Cloud Configuration

1. **Automatic Detection**: Platform detects available packages at runtime
2. **Graceful Fallback**: Missing heavy packages trigger mock/demo modes
3. **User Feedback**: UI shows which capabilities are available
4. **Error Handling**: App never crashes due to missing optional dependencies

## 🔧 How It Works

### 1. Container Startup Sequence

```bash
# 1. System packages installed (packages.txt)
apt-get update && apt-get install -y libfftw3-dev libgsl-dev ...

# 2. Environment setup (setup.sh)
./setup.sh  # Configures paths, permissions, optimizations

# 3. Python packages installed (requirements.txt)
pip install -r requirements.txt

# 4. Streamlit app starts
streamlit run app.py
```

### 2. Runtime Capability Detection

```python
# In app.py - feature_flags.py handles this
capabilities = detect_capabilities()
if capabilities["py21cmfast"].available:
    # Use real 21cm simulations
else:
    # Use mock/demo mode
```

### 3. User Experience

- ✅ **Always works**: Core functionality available regardless of missing packages
- 🎯 **Clear feedback**: Status panel shows what's available
- 🚀 **Fast startup**: Lightweight by default, heavy packages optional
- 📱 **Mobile friendly**: Responsive UI works on all devices

## 🛠️ Customizing for Your Deployment

### Enable Heavy Packages

To enable full scientific capability, uncomment in `requirements.txt`:

```bash
# Uncomment these lines:
py21cmfast>=3.3.0
jwst>=1.12.0
tools21cm>=2.0.0
```

### Strict Mode

For research deployments requiring all packages:

```python
# In app.py sidebar
strict_mode = True  # Fail if any packages missing
```

### Environment Variables

Set in Streamlit Cloud dashboard:

```bash
ASTRO_AI_LOG_LEVEL=INFO
ASTRO_AI_STRICT=false
OPENAI_API_KEY=your_key_here
```

## 📊 Performance Considerations

### Memory Optimization

- Single-threaded operation (Cloud CPU sharing)
- Efficient caching for astronomical data
- Garbage collection tuning for large datasets

### Startup Time

- **Fast (< 30s)**: Minimal tier with core packages
- **Medium (1-2 min)**: Science-light with bagpipes
- **Slow (3-5 min)**: Full tier with all heavy packages

### Storage Efficiency

- Cache directories for reused astronomical data
- Temporary processing space for large files
- Automatic cleanup of intermediate results

## 🚨 Troubleshooting

### Common Issues

**Build fails on heavy packages**:

- ✅ Normal behavior - app will use fallback modes
- Check logs for specific package causing issues
- Consider reducing to science-light tier

**Memory errors during startup**:

- Reduce concurrent operations in setup.sh
- Increase garbage collection frequency
- Use smaller default datasets

**Import errors at runtime**:

- Check feature_flags.py output in sidebar
- Verify all required packages in requirements.txt
- Enable strict mode to catch missing dependencies early

### Debug Information

The app provides extensive debugging through:

- Sidebar status panel (always visible)
- Console logs (check Streamlit Cloud logs)
- Runtime capability detection (feature_flags.py)

## 🎯 Success Metrics

A successful deployment shows:

- ✅ App loads without Python import errors
- ✅ Core modules (data upload, basic analysis) functional
- ✅ Status panel shows available/missing capabilities clearly
- ✅ At least one analysis module works fully (even in fallback mode)
- ✅ AI integration responds (if OpenAI key provided)

---

## 🔗 Related Files

- `app.py` - Main application with module routing
- `utils/feature_flags.py` - Dependency detection and fallback logic
- `INSTALL.md` - Complete installation guide for all environments
- `README.md` - Project overview and scientific background

**Questions?** Check the main README.md or raise an issue on GitHub!

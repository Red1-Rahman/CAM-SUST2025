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
pip install openai>=1.0.0
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
pip install openai>=1.0.0
pip install pytest>=7.0.0 black>=23.0.0
```

## Environment Files

For reproducible installations, you can also create:

- environment.yml (for conda)
- requirements.txt (for pip)

---

## Deployment Guidance & Eliminating Runtime Warnings

The application supports optional heavy scientific dependencies. In production you can choose a capability tier:

| Tier          | Intended Use             | Required Packages                                                |
| ------------- | ------------------------ | ---------------------------------------------------------------- |
| Minimal       | Demo, AI summaries only  | core requirements (see requirements.txt), NO py21cmfast, NO jwst |
| Science-Light | Add Bagpipes SED fitting | + bagpipes (already in requirements)                             |
| Full 21cm     | Simulated reionization   | + py21cmfast (+ tools21cm optional)                              |
| JWST + SED    | JWST pipeline reduction  | + jwst + bagpipes                                                |
| Complete      | Everything               | all above                                                        |

### Recommended Strategy

1. Build a base Docker / conda image for the Minimal or Science-Light tier (fast CI).
2. Create a separate image for Full science features (longer build) used only where needed.
3. Use environment variable `ASTRO_AI_STRICT=1` (you can add logic) to enforce mandatory capabilities.

### Strict Mode (in app)

The sidebar "Environment Status" expander offers a checkbox for Strict Mode which raises an error if required modules are missing. Adjust required list in `app.py`:

```python
all_required_or_raise(["py21cmfast", "bagpipes", "jwst_pipeline", "astropy"])
```

Edit this list per deployment tier.

### Suppressing Warnings Cleanly

All module availability messages now use `logging`. Configure log level via environment:

```bash
export ASTRO_AI_LOG_LEVEL=WARNING  # Linux / WSL
```

Or adapt `_configure_logging()` in `app.py` to read that.

### Installing Heavy Dependencies

| Package          | Notes                                                                   |
| ---------------- | ----------------------------------------------------------------------- |
| py21cmfast       | Prefer Linux/WSL; needs FFTW, GSL. Use conda for libs then pip install. |
| jwst             | Large; adds CRDS downloads. Set `CRDS_PATH` & `CRDS_SERVER_URL`.        |
| bagpipes         | Pure Python + some scientific stack; LaTeX optional for nicer plots.    |
| tools21cm        | Utility library; optional for extra analysis.                           |
| nautilus-sampler | Alternative nested sampler if PyMultiNest unavailable.                  |

### JWST Pipeline Environment Variables

```bash
export CRDS_PATH="$HOME/.crds"
export CRDS_SERVER_URL="https://jwst-crds.stsci.edu"
```

On Windows PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable('CRDS_PATH', "$Env:USERPROFILE\.crds", 'User')
[System.Environment]::SetEnvironmentVariable('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu', 'User')
```

### LaTeX (Bagpipes Plot Quality)

Install a lightweight TeX distro (MiKTeX on Windows) or disable TeX in code:

```python
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
```

### Docker (Example Skeleton)

```dockerfile
FROM mambaorg/micromamba:1.5.8
COPY environment.yml /tmp/environment.yml
RUN micromamba env create -f /tmp/environment.yml -y && \
	micromamba clean --all --yes
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV OPENAI_API_KEY=changeme
WORKDIR /app
COPY . /app
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Preflight CI Check (Optional)

Add a small script to assert required tier before deployment:

```python
# scripts/preflight.py
from utils.feature_flags import detect_capabilities, all_required_or_raise
all_required_or_raise(["bagpipes"])  # minimal tier example
print("Preflight passed")
```

Run in CI:

```bash
python scripts/preflight.py
```

### OpenAI Client Version

The codebase expects `openai>=1.0.0`. Ensure the deployed image doesn’t pin an older cached layer.

---

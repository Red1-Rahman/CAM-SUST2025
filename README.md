# Astro-AI: Galaxy Evolution Analysis Platform 🌌

A comprehensive web-based platform for multi-wavelength galaxy evolution analysis, combining 21cm cosmological simulations, galaxy cluster environment studies, and JWST spectroscopic analysis with AI-powered scientific insights.

## 🚀 Features

### Core Analysis Modules

1. **🌌 Cosmic Evolution (21cm Analysis)**

   - 21cm brightness temperature simulations
   - Power spectrum analysis
   - Reionization epoch modeling
   - Integration with py21cmfast

2. **🌟 Cluster Environment Analyzer**

   - Galaxy cluster vs field environment comparison
   - SED fitting with Bagpipes
   - Environmental quenching analysis
   - Stellar mass assembly studies

3. **🔭 JWST Spectroscopic Analyzer**

   - NIRSpec pipeline integration
   - Multi-stage data reduction
   - Optimal 1D spectral extraction
   - Stellar population analysis

4. **📊 Integrated Dashboard**

   - Cross-module result visualization
   - Timeline integration plots
   - Comparative environment analysis
   - JWST showcase panels

5. **🤖 AI-Powered Insights**
   - OpenAI GPT-4 integration
   - Scientific interpretation and reporting
   - Automated next-steps suggestions
   - Research synthesis across modules

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Quick Setup

```bash
# Clone or download the Astro-AI directory
cd Astro-AI

# Install core requirements
pip install -r requirements.txt

# Optional: Install astronomical packages
pip install py21cmfast bagpipes jwst
```

### Environment Configuration

1. Set up OpenAI API key in Streamlit secrets:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

## 🚀 Quick Start

### Launch the Application

```bash
streamlit run app.py
```

### Basic Workflow

1. **Navigation**: Use the sidebar to select analysis modules
2. **Data Input**: Upload data files or use built-in mock data
3. **Parameter Configuration**: Adjust analysis parameters as needed
4. **Run Analysis**: Execute selected modules
5. **View Results**: Explore visualizations and AI insights
6. **Dashboard**: Compare results across modules
7. **Generate Reports**: Use AI assistant for scientific reporting

## 📁 Project Structure

```
Astro-AI/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
│
├── modules/                  # Analysis modules
│   ├── cos_evo/             # Cosmic evolution (21cm)
│   │   └── cosmic_evolution.py
│   ├── cluster_analyzer/     # Galaxy cluster analysis
│   │   └── cluster_analysis.py
│   └── jwst_analyzer/       # JWST spectroscopy
│       └── jwst_pipeline.py
│
├── dashboard/               # Integrated visualization
│   └── dashboard.py
│
├── api/                     # AI integration
│   └── openai_integration.py
│
└── utils/                   # Utility functions
    └── data_utils.py
```

## 🔬 Scientific Applications

### Research Areas

- **Galaxy Formation & Evolution**: Multi-epoch analysis from reionization to present
- **Environmental Effects**: Cluster vs field galaxy populations
- **Early Universe Studies**: JWST high-redshift galaxy observations
- **Cosmological Simulations**: 21cm intensity mapping and reionization
- **Stellar Population Synthesis**: SED fitting and galaxy properties

### Analysis Capabilities

- **Statistical Analysis**: Population studies, red fraction evolution
- **Spectroscopic Analysis**: Emission line diagnostics, stellar ages
- **Environmental Studies**: Quenching mechanisms, mass assembly
- **Cosmological Modeling**: Power spectra, brightness temperature evolution

## 🤖 AI-Powered Features

### Scientific Insights

- **Automated Interpretation**: AI analysis of results with scientific context
- **Cross-Module Synthesis**: Comparative analysis across different wavelengths
- **Research Suggestions**: Next-steps recommendations based on findings

### Report Generation

- **Scientific Writing**: AI-assisted report sections (Introduction, Methods, Results, Discussion)
- **Literature Context**: Integration with current research trends
- **Publication Ready**: Professional formatting and scientific language

## 📊 Data Formats

### Input Data

- **21cm Simulations**: HDF5 brightness temperature cubes
- **Galaxy Catalogs**: CSV/FITS files with photometry and redshifts
- **JWST Observations**: FITS files from STScI archive
- **Spectroscopic Data**: 1D/2D spectra in standard formats

### Output Products

- **Visualizations**: High-quality matplotlib figures
- **Analysis Results**: JSON/CSV summary statistics
- **Scientific Reports**: Formatted text with AI insights
- **Interactive Plots**: Streamlit widgets for exploration

## 🔧 Configuration

### Analysis Parameters

- **Cosmological**: Ωm, Ωb, H0, σ8 for simulations
- **SED Fitting**: Star formation histories, metallicity ranges
- **Spectroscopic**: Wavelength ranges, line fitting parameters
- **Statistical**: Sample selection, error estimation

### Performance Options

- **Mock Data**: Fast testing with synthetic datasets
- **Parallel Processing**: Multi-core analysis where available
- **Memory Management**: Efficient handling of large datasets

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install pytest black

# Run tests
pytest

# Format code
black .
```

### Code Organization

- **Modular Design**: Independent analysis modules
- **Error Handling**: Graceful fallbacks for missing dependencies
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit tests for core functionality

## 📚 Scientific Background

### Key References

- **21cm Cosmology**: Furlanetto et al. (2006), Mesinger et al. (2011)
- **Galaxy Clusters**: Peng et al. (2010), Wetzel et al. (2012)
- **JWST Science**: Robertson et al. (2022), Curtis-Lake et al. (2023)
- **SED Fitting**: Carnall et al. (2018), Leja et al. (2017)

### Theoretical Framework

- **Reionization Models**: Semi-numerical simulations with py21cmfast
- **Environmental Quenching**: Satellite quenching in cluster environments
- **Stellar Population Synthesis**: Composite stellar population models
- **Observational Selection**: Survey completeness and selection effects

## 🔬 Example Workflows

### Multi-Wavelength Galaxy Study

1. Run 21cm simulation for cosmic context
2. Analyze cluster vs field populations
3. Compare with JWST high-redshift observations
4. Generate AI synthesis report

### Environmental Quenching Analysis

1. Load galaxy cluster catalog
2. Separate cluster/field populations
3. Run SED fitting for stellar masses/ages
4. Analyze quenching efficiency vs environment

### JWST Early Galaxy Analysis

1. Process NIRSpec observations
2. Extract 1D spectra with optimal weighting
3. Fit stellar population models
4. Compare with simulation predictions

## 📞 Support

### Documentation

- **In-App Help**: Tooltips and explanations throughout interface
- **Error Messages**: Descriptive error handling with suggestions
- **Examples**: Built-in mock data and parameter sets

### Troubleshooting

- **Dependencies**: Check requirements.txt installation
- **Data Formats**: Verify input file formats and headers
- **Memory Issues**: Use mock data for testing large analyses
- **API Limits**: Monitor OpenAI usage and rate limits

## 📄 License

This project builds upon and integrates several open-source astronomical packages:

- py21cmfast: GPL-3.0 license
- Bagpipes: MIT license
- JWST Pipeline: BSD-3-Clause license
- Astropy: BSD-3-Clause license

Please cite appropriate packages when using this platform for research.

## 🌟 Acknowledgments

Built using functions and methodologies from the analyzed codebase including:

- 21cm simulation notebooks and cosmological analysis tools
- Galaxy cluster workshop materials and SED fitting pipelines
- JWST data reduction scripts and spectroscopic analysis methods
- Statistical analysis utilities and visualization functions

---

**Astro-AI Platform** - Advancing galaxy evolution research through integrated multi-wavelength analysis and AI-powered scientific insights.

# 🧠 Architecture Overview

## 🏁 Entry Point & User Interface

### Main Application

* **File:** `app.py`
* **Framework:** [Streamlit](https://streamlit.io/)
* **Purpose:** Web-based entry point and navigation hub for all analysis modules.

### Navigation Functions

* `main`
* `show_home`
* `show_data_upload`
* `show_cosmic_evolution`
* `show_cluster_analyzer`
* `show_jwst_analyzer`
* `show_dashboard`
* `show_report`

### State Management

Utilizes **Streamlit’s `st.session_state`** for maintaining runtime state across navigation steps:

* `cosmology_params` – Cosmological parameters (H₀, Ωₘ, redshift ranges)
* `uploaded_data` – User-uploaded catalogs or JWST files
* `cos_evo_results` – 21cm simulation results
* `extracted_spectrum` – JWST spectrum extraction results
* `spectral_fit_results` – Bagpipes fitting outputs

---

## 🔬 Core Analysis Modules

### 🌌 21cm Cosmic Evolution

* **File:** `cosmic_evolution.py`
* **Class:** `CosmicEvolution` (`modules.cos_evo.cosmic_evolution.CosmicEvolution`)
* **Key Methods:**

  * `run_simulation()` – Execute 21cmFAST simulations
  * `plot_global_evolution()` – Visualize global 21cm signal evolution
  * `plot_power_spectra_evolution()` – Analyze power spectra evolution
  * `plot_brightness_temperature_slices()` – Generate temperature maps
  * `generate_summary_report()` – Summarize simulation results
  * `export_results()` – Export outputs
* **Fallback:** Operates in **mock/simulation mode** when `py21cmfast` is unavailable.

---

### 🌟 Cluster Environment Analyzer

* **File:** `cluster_analysis.py`
* **Class:** `ClusterAnalyzer` (`modules.cluster_analyzer.cluster_analysis.ClusterAnalyzer`)
* **Core Features:**

  * `setup_bagpipes_model()` – Configure SED fitting model
  * Galaxy spatial distribution and clustering analysis
  * Color–magnitude diagram generation
  * Mass–SFR relation visualization
  * Environmental quenching analysis

---

### 🔭 JWST Spectrum Analyzer

* **File:** `jwst_pipeline.py`
* **Class:** `JWSTAnalyzer` (`modules.jwst_analyzer.jwst_pipeline.JWSTAnalyzer`)
* **Key Methods:**

  * `setup_spectral_fitting_model()` – Initialize spectral models
  * `fit_spectrum_bagpipes()` – Fit extracted spectra with Bagpipes
  * `plot_spectral_fit_results()` – Visualize fitting outputs
  * `generate_pipeline_summary()` – Summarize pipeline results
  * `export_results()` – Export analysis data
* **Pipeline Stages:** Stage 1–3 JWST reduction + optimal 1D extraction
* **Fallback:** Runs in **mock mode** if STScI JWST pipeline or Bagpipes are missing.

---

## 📊 Integration & Visualization

### Dashboard & Comparative Analysis

* **File:** `dashboard.py`
* **Class:** `Dashboard` (`dashboard.dashboard.Dashboard`)
* **Features:**

  * `load_results()` – Load and synchronize results across modules
  * `create_timeline_integration()` – Cross-module temporal visualization
  * `create_environment_comparison()` – Cluster vs. field comparison
  * `create_jwst_showcase()` – Showcase JWST spectral results
  * `create_summary_dashboard()` – Unified scientific overview
  * `generate_integration_report()` – Cross-domain reporting
  * `export_dashboard_plots()` – Export integrated visualizations

---

## ⚙️ Utility Infrastructure

### 🛠 Core Utilities

| Function                   | File                 | Class            | Key Methods                                                                           |
| -------------------------- | -------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| **Cosmology Calculations** | `cosmology_utils.py` | `CosmologyUtils` | `redshift_to_age`, `luminosity_distance`, `angular_diameter_distance`, `scale_factor` |
| **Data Handling**          | `data_handler.py`    | `DataHandler`    | `list_files`, `delete_file`                                                           |
| **Plotting Utilities**     | `plotting_utils.py`  | `PlottingUtils`  | `plot_corner`                                                                         |

---

### 🔧 Feature Management

* **File:** `feature_flags.py`
* **Functions:**

  * `detect_capabilities()` – Detect available dependencies
  * `all_required_or_raise()` – Enforce strict dependency mode
  * `summarize_status()` – Generate runtime capability report
* **Supported Scientific Libraries:**

  * `py21cmfast`, `tools21cm`, `bagpipes`, `jwst`, `astropy`

---

## 🤖 AI Integration

### OpenAI Assistant

* **File:** `openai_integration.py`
* **Class:** `OpenAIAssistant` (`api.openai_integration.OpenAIAssistant`)
* **Features:**

  * `generate_report_section()` – Automated report writing
  * `generate_comparative_analysis()` – Cross-module insights
  * `suggest_next_steps()` – Research recommendations
  * `check_api_status()` – Verify API connectivity

---

## 🚀 Deployment & Execution

### Launch & Installation

* **Launcher:** `launch.bat` – Local startup script
* **Requirements:** `requirements.txt`, optional `environment.yml`
* **Setup Guide:** `INSTALL.md` (includes dependency tiers)

#### Deployment Tiers

| Tier                 | Description                                               |
| -------------------- | --------------------------------------------------------- |
| **Minimal**          | Basic Streamlit UI without heavy scientific dependencies  |
| **Science-Light**    | Core astronomy tools only                                 |
| **Full (21cm/JWST)** | Complete setup with 21cmFAST, JWST pipeline, and Bagpipes |
| **Complete**         | All modules + AI integration                              |

---

## 🔄 Data Flow Architecture

### Typical Analysis Workflow

1. **Input:**
   `show_data_upload` → User uploads files → Parameters stored in `st.session_state`.
2. **Analysis Execution:**
   `show_cosmic_evolution`, `show_cluster_analyzer`, and `show_jwst_analyzer` perform analyses and store results.
3. **Integration & Visualization:**
   `show_dashboard` aggregates and visualizes outputs via `Dashboard`.
4. **Reporting & Insights:**
   `show_report` uses `OpenAIAssistant` for generating summaries, analyses, and next-step recommendations.

---

## 🧬 Module Dependencies

### Core Dependencies

* `streamlit`, `numpy`, `pandas`, `matplotlib`

### AI Dependencies

* `openai>=1.0.0`

### Optional Scientific Dependencies

| Purpose                 | Libraries                 |
| ----------------------- | ------------------------- |
| **21cm Simulations**    | `py21cmfast`, `tools21cm` |
| **SED Fitting**         | `bagpipes`                |
| **JWST Pipeline**       | `jwst` (STScI pipeline)   |
| **Astronomy Utilities** | `astropy`, `scipy`        |

### Deployment Flexibility

* **Environment Detection:** via `feature_flags.py`
* **Strict Mode:** optional dependency enforcement
* **Mock Modes:** automatic fallback when heavy packages are unavailable

---

## 📘 Documentation & Notes

* **Overview:** `README.md`
* **Team Ideation & Goals:** `ideation.md`
* **Setup & Execution:** `INSTALL.md`

Entry point and UI (Streamlit)

Main app and navigation in app.py: main, show_home, show_data_upload, show_cosmic_evolution, show_cluster_analyzer, show_jwst_analyzer, show_dashboard, show_report.
State management via Streamlit’s st.session_state (e.g., cosmology_params, uploaded_data, cos_evo_results, extracted_spectrum, spectral_fit_results).
Domain modules

21cm Cosmic Evolution in cosmic_evolution.py
modules.cos_evo.cosmic_evolution.CosmicEvolution: run_simulation, plot_global_evolution, plot_power_spectra_evolution, plot_brightness_temperature_slices, generate_summary_report, export_results.
Falls back to mock/simulation mode if py21cmfast is absent.
Cluster analysis in cluster_analysis.py
modules.cluster_analyzer.cluster_analysis.ClusterAnalyzer: setup_bagpipes_model and related SED fitting utilities.
JWST spectroscopy in jwst_pipeline.py
modules.jwst_analyzer.jwst_pipeline.JWSTAnalyzer: setup_spectral_fitting_model, fit_spectrum_bagpipes, plot_spectral_fit_results, generate_pipeline_summary, export_results.
Works in a mock mode when JWST STScI pipeline or Bagpipes aren’t installed.
Integration and dashboards

Comparative visualizations in dashboard.py
dashboard.dashboard.Dashboard: load_results, create_timeline_integration, create_environment_comparison, create_jwst_showcase, create_summary_dashboard, generate_integration_report, export_dashboard_plots.
Utilities

Cosmology helpers in cosmology_utils.py
utils.cosmology_utils.CosmologyUtils: e.g., redshift_to_age, luminosity_distance, angular_diameter_distance, scale_factor.
Data I/O in data_handler.py
utils.data_handler.DataHandler: list_files, delete_file.
Plotting helpers in plotting_utils.py
utils.plotting_utils.PlottingUtils: plot_corner.
Feature flags/capability detection in feature_flags.py
utils.feature_flags.detect_capabilities, utils.feature_flags.all_required_or_raise, utils.feature_flags.summarize_status.
AI integration

OpenAI assistant in openai_integration.py
api.openai_integration.OpenAIAssistant: generate_report_section, generate_comparative_analysis, suggest_next_steps, check_api_status.
Execution and deployment

Local launcher in launch.bat.
Setup guidance in INSTALL.md and dependency tiers (Minimal/Science-Light/Full 21cm/JWST/Complete).
Requirements in requirements.txt and optional environment.yml.
Data flow (typical path)

Inputs via show_data_upload → params saved into st.session_state.
Simulation/analysis via show_cosmic_evolution, show_cluster_analyzer, show_jwst_analyzer producing results in session state.
Integration/visualization via show_dashboard using Dashboard.
Reporting and AI synthesis via show_report using OpenAIAssistant.
Documentation and notes

Project overview in README.md.
Team ideation and goals in ideation.md.

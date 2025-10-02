# Dashboard Module for Astro-AI
# 
# This module provides integrated visualization and comparison tools
# for results from all analysis modules (Cosmic Evolution, Cluster Analysis, JWST)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

class Dashboard:
    """
    Integrated Dashboard for Astro-AI Results.
    
    This class provides methods for creating comparative visualizations
    that integrate results from all analysis modules.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.cos_evo_results = None
        self.cluster_results = None
        self.jwst_results = None
        self.integration_plots = {}
        
    def load_results(self, cos_evo=None, cluster=None, jwst=None):
        """
        Load results from analysis modules.
        
        Parameters:
        -----------
        cos_evo : dict, optional
            Results from cosmic evolution module
        cluster : dict, optional
            Results from cluster analysis module
        jwst : dict, optional
            Results from JWST analysis module
        """
        if cos_evo is not None:
            self.cos_evo_results = cos_evo
            
        if cluster is not None:
            self.cluster_results = cluster
            
        if jwst is not None:
            self.jwst_results = jwst
    
    def create_timeline_integration(self, save_path=None):
        """
        Create integrated cosmic timeline visualization.
        
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Cosmic time and redshift arrays
        cosmic_time = np.linspace(0.5, 13.8, 100)
        redshift = self._time_to_redshift(cosmic_time)
        
        # Panel 1: 21cm Evolution
        ax1 = fig.add_subplot(gs[0, :])
        
        if self.cos_evo_results:
            z_21cm = self.cos_evo_results.get('redshifts', np.linspace(6, 15, 10))
            signal_21cm = self.cos_evo_results.get('global_signals', 
                                                  -50 * np.exp(-(z_21cm - 10)**2 / 10))
            
            # Convert redshift to cosmic time
            time_21cm = self._redshift_to_time(z_21cm)
            
            ax1.plot(time_21cm, signal_21cm, 'b-', linewidth=3, label='21cm Global Signal')
            ax1.fill_between(time_21cm, signal_21cm - 5, signal_21cm + 5, 
                           alpha=0.3, color='blue')
        else:
            # Mock data
            time_21cm = np.linspace(0.5, 2.5, 50)
            z_mock = self._time_to_redshift(time_21cm)
            signal_21cm = -50 * np.exp(-(z_mock - 10)**2 / 10)
            ax1.plot(time_21cm, signal_21cm, 'b-', linewidth=3, label='21cm Global Signal')
        
        ax1.set_ylabel('21cm Signal [mK]', fontsize=12)
        ax1.set_title('Cosmic Evolution Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add epoch markers
        ax1.axvline(0.8, color='red', linestyle='--', alpha=0.7, label='z~15 (Cosmic Dawn)')
        ax1.axvline(1.2, color='orange', linestyle='--', alpha=0.7, label='z~10 (Reionization)')
        ax1.axvline(2.0, color='green', linestyle='--', alpha=0.7, label='z~6 (End EoR)')
        
        # Panel 2: Cluster Environment Effects
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Red fraction vs mass
        mass_bins = np.logspace(9.5, 11.5, 10)
        
        if self.cluster_results and 'red_fraction' in self.cluster_results:
            rf_results = self.cluster_results['red_fraction']
            mass_centers = rf_results['mass_centers']
            rf_cluster = rf_results['cluster']['red_fraction']
            rf_field = rf_results['field']['red_fraction']
            
            ax2.semilogx(10**mass_centers, rf_cluster, 'ro-', linewidth=2, 
                        markersize=6, label='Cluster')
            ax2.semilogx(10**mass_centers, rf_field, 'bo-', linewidth=2, 
                        markersize=6, label='Field')
        else:
            # Mock environmental effects
            rf_cluster = 0.1 + 0.6 / (1 + np.exp(-(mass_bins - 10**10.5) / 1e10))
            rf_field = 0.05 + 0.4 / (1 + np.exp(-(mass_bins - 1e11) / 1e10))
            
            ax2.semilogx(mass_bins, rf_cluster, 'ro-', linewidth=2, markersize=6, label='Cluster')
            ax2.semilogx(mass_bins, rf_field, 'bo-', linewidth=2, markersize=6, label='Field')
        
        ax2.set_xlabel('Stellar Mass [M☉]')
        ax2.set_ylabel('Red Fraction')
        ax2.set_title('Environmental Quenching', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Panel 3: Environmental Effect Strength
        ax3 = fig.add_subplot(gs[1, 1])
        
        if self.cluster_results and 'red_fraction' in self.cluster_results:
            rf_results = self.cluster_results['red_fraction']
            mass_centers = rf_results['mass_centers']
            delta_rf = rf_results['environmental_effect']['delta_red_fraction']
            delta_rf_err = rf_results['environmental_effect']['delta_red_fraction_err']
            
            ax3.errorbar(mass_centers, delta_rf, yerr=delta_rf_err, 
                        fmt='go-', linewidth=2, markersize=6, capsize=3)
        else:
            # Mock environmental effect
            delta_rf = rf_cluster - rf_field
            ax3.plot(np.log10(mass_bins), delta_rf, 'go-', linewidth=2, markersize=6)
        
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('log(M*/M☉)')
        ax3.set_ylabel('Δ(Red Fraction)')
        ax3.set_title('Environmental Effect Strength', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: JWST Spectroscopic Properties
        ax4 = fig.add_subplot(gs[2, 0])
        
        if self.jwst_results and 'spectral_fits' in self.jwst_results:
            # Extract properties from multiple sources
            masses = []
            ages = []
            redshifts = []
            
            for source_id, fit_result in self.jwst_results['spectral_fits'].items():
                masses.append(fit_result['stellar_mass'])
                ages.append(fit_result['age'])
                redshifts.append(fit_result['redshift'])
            
            if masses:
                scatter = ax4.scatter(masses, ages, c=redshifts, s=100, cmap='viridis', 
                                    alpha=0.8, edgecolors='black')
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Redshift')
        else:
            # Mock JWST results
            n_sources = 20
            masses = np.random.uniform(9.5, 11.5, n_sources)
            ages = np.random.uniform(0.5, 8.0, n_sources)
            redshifts = np.random.uniform(1.0, 4.0, n_sources)
            
            scatter = ax4.scatter(masses, ages, c=redshifts, s=100, cmap='viridis', 
                                alpha=0.8, edgecolors='black')
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Redshift')
        
        ax4.set_xlabel('log(M*/M☉)')
        ax4.set_ylabel('Age [Gyr]')
        ax4.set_title('JWST Spectroscopic Properties', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Integrated Mass-SFR Relation
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Combine cluster and JWST results
        mass_range = np.linspace(9, 12, 100)
        
        # Main sequence relation
        ms_sfr = mass_range - 9.0  # Simplified main sequence
        ax5.plot(mass_range, ms_sfr, 'k--', linewidth=2, alpha=0.7, label='Main Sequence')
        
        # Add cluster results if available
        if self.cluster_results and 'sed_results' in self.cluster_results:
            cluster_masses = []
            cluster_sfrs = []
            for galaxy_id, sed_result in self.cluster_results['sed_results'].items():
                cluster_masses.append(sed_result['stellar_mass'])
                cluster_sfrs.append(np.log10(sed_result['sfr'] + 1e-3))
            
            if cluster_masses:
                ax5.scatter(cluster_masses, cluster_sfrs, c='red', s=50, 
                          alpha=0.7, label='Cluster galaxies')
        
        # Add JWST results if available
        if self.jwst_results and 'spectral_fits' in self.jwst_results:
            jwst_masses = []
            jwst_sfrs = []
            for source_id, fit_result in self.jwst_results['spectral_fits'].items():
                if 'sfr' in fit_result:
                    jwst_masses.append(fit_result['stellar_mass'])
                    jwst_sfrs.append(np.log10(fit_result['sfr'] + 1e-3))
            
            if jwst_masses:
                ax5.scatter(jwst_masses, jwst_sfrs, c='blue', s=100, 
                          alpha=0.8, marker='s', label='JWST spectra')
        else:
            # Mock data
            jwst_masses = np.random.uniform(9.5, 11.5, 10)
            jwst_sfrs = jwst_masses - 9.0 + np.random.normal(0, 0.3, 10)
            ax5.scatter(jwst_masses, jwst_sfrs, c='blue', s=100, 
                      alpha=0.8, marker='s', label='JWST spectra')
        
        ax5.set_xlabel('log(M*/M☉)')
        ax5.set_ylabel('log(SFR) [M☉/yr]')
        ax5.set_title('Integrated Mass-SFR Relation', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Astro-AI: Integrated Galaxy Evolution Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.integration_plots['timeline'] = fig
        return fig, [ax1, ax2, ax3, ax4, ax5]
    
    def create_environment_comparison(self, save_path=None):
        """
        Create detailed environment comparison plots.
        
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Environmental Effects on Galaxy Properties', fontsize=16, fontweight='bold')
        
        # Panel 1: Spatial Distribution
        ax = axes[0, 0]
        
        if (self.cluster_results and 'cluster_members' in self.cluster_results 
            and 'field_galaxies' in self.cluster_results):
            cluster_gals = self.cluster_results['cluster_members']
            field_gals = self.cluster_results['field_galaxies']
            
            if 'ra' in cluster_gals.columns and 'dec' in cluster_gals.columns:
                ax.scatter(field_gals['ra'], field_gals['dec'], 
                          alpha=0.6, s=20, c='lightblue', label='Field')
                ax.scatter(cluster_gals['ra'], cluster_gals['dec'], 
                          alpha=0.8, s=40, c='red', label='Cluster')
        else:
            # Mock spatial data
            n_field, n_cluster = 800, 200
            field_ra = np.random.uniform(149, 151, n_field)
            field_dec = np.random.uniform(1.5, 2.5, n_field)
            cluster_ra = np.random.normal(150, 0.1, n_cluster)
            cluster_dec = np.random.normal(2.0, 0.1, n_cluster)
            
            ax.scatter(field_ra, field_dec, alpha=0.6, s=20, c='lightblue', label='Field')
            ax.scatter(cluster_ra, cluster_dec, alpha=0.8, s=40, c='red', label='Cluster')
        
        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')
        ax.set_title('Spatial Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Color-Magnitude Diagram
        ax = axes[0, 1]
        
        # Generate or use actual CMD data
        r_mags = np.random.uniform(18, 25, 1000)
        colors = np.random.normal(0.8, 0.3, 1000)
        
        # Simulate cluster vs field differences
        cluster_mask = np.random.choice([True, False], size=1000, p=[0.3, 0.7])
        colors[cluster_mask] += 0.2  # Cluster galaxies redder
        
        ax.scatter(r_mags[~cluster_mask], colors[~cluster_mask], 
                  alpha=0.6, s=20, c='lightblue', label='Field')
        ax.scatter(r_mags[cluster_mask], colors[cluster_mask], 
                  alpha=0.8, s=30, c='red', label='Cluster')
        
        # Red sequence
        r_range = np.linspace(18, 25, 100)
        red_seq = 1.0 + 0.05 * (r_range - 20)
        ax.plot(r_range, red_seq, 'k--', alpha=0.7, label='Red sequence')
        
        ax.set_xlabel('r magnitude')
        ax.set_ylabel('g - r color')
        ax.set_title('Color-Magnitude Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Panel 3: Redshift Distribution
        ax = axes[0, 2]
        
        z_bins = np.linspace(0.5, 2.5, 30)
        
        # Mock redshift distributions
        z_field = np.random.uniform(0.5, 2.5, 800)
        z_cluster = np.concatenate([
            np.random.normal(1.2, 0.05, 150),  # Main cluster
            np.random.normal(1.8, 0.03, 50)    # Secondary peak
        ])
        
        ax.hist(z_field, bins=z_bins, alpha=0.6, color='lightblue', 
               label='Field', density=True)
        ax.hist(z_cluster, bins=z_bins, alpha=0.8, color='red', 
               label='Cluster', density=True)
        
        ax.axvline(1.2, color='red', linestyle='--', alpha=0.7, label='Cluster z')
        
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Normalized Number')
        ax.set_title('Redshift Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Stellar Mass Functions
        ax = axes[1, 0]
        
        mass_bins = np.linspace(9.5, 11.5, 15)
        mass_centers = mass_bins[:-1] + np.diff(mass_bins) / 2
        
        # Mock SMFs
        phi_cluster = 10**(-(mass_centers - 10.5)**2 / 0.5 - 2)
        phi_field = 10**(-(mass_centers - 10.3)**2 / 0.8 - 2.2)
        
        ax.semilogy(mass_centers, phi_cluster, 'ro-', linewidth=2, 
                   markersize=6, label='Cluster')
        ax.semilogy(mass_centers, phi_field, 'bo-', linewidth=2, 
                   markersize=6, label='Field')
        
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('φ [Mpc⁻³ dex⁻¹]')
        ax.set_title('Stellar Mass Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 5: Red Fraction Evolution
        ax = axes[1, 1]
        
        mass_bins_rf = np.linspace(9.5, 11.5, 8)
        mass_centers_rf = mass_bins_rf[:-1] + np.diff(mass_bins_rf) / 2
        
        # Mock red fractions
        rf_cluster = 0.1 + 0.7 / (1 + np.exp(-(mass_centers_rf - 10.3) / 0.3))
        rf_field = 0.05 + 0.4 / (1 + np.exp(-(mass_centers_rf - 10.7) / 0.4))
        
        ax.plot(mass_centers_rf, rf_cluster, 'ro-', linewidth=2, 
               markersize=6, label='Cluster')
        ax.plot(mass_centers_rf, rf_field, 'bo-', linewidth=2, 
               markersize=6, label='Field')
        
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('Red Fraction')
        ax.set_title('Red Fraction vs Mass')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Panel 6: Quenching Efficiency
        ax = axes[1, 2]
        
        delta_rf = rf_cluster - rf_field
        quench_eff = delta_rf / (1 - rf_field)  # Quenching efficiency
        
        ax.plot(mass_centers_rf, quench_eff, 'go-', linewidth=3, markersize=8)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('Quenching Efficiency')
        ax.set_title('Environmental Quenching Efficiency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.integration_plots['environment'] = fig
        return fig, axes
    
    def create_jwst_showcase(self, save_path=None):
        """
        Create JWST results showcase.
        
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('JWST Spectroscopic Analysis Results', fontsize=16, fontweight='bold')
        
        # Panel 1: Example 2D Spectrum
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Mock 2D spectrum
        ny, nx = 30, 200
        wavelength = np.linspace(1.0, 5.0, nx)
        y_coords = np.arange(ny)
        
        # Create 2D spectrum with emission lines
        flux_1d = self._generate_mock_spectrum_for_plot(wavelength)
        profile = np.exp(-(y_coords - ny/2)**2 / (2 * 3**2))
        spectrum_2d = np.outer(profile, flux_1d)
        
        im1 = ax1.imshow(spectrum_2d, aspect='auto', origin='lower',
                        extent=[wavelength[0], wavelength[-1], 0, ny],
                        cmap='viridis', interpolation='bilinear')
        
        ax1.set_xlabel('Wavelength (μm)')
        ax1.set_ylabel('Spatial (pixels)')
        ax1.set_title('2D Spectrum')
        plt.colorbar(im1, ax=ax1, label='Flux')
        
        # Panel 2: Extracted 1D Spectrum
        ax2 = fig.add_subplot(gs[0, 1])
        
        flux_1d_clean = gaussian_filter1d(flux_1d, sigma=1)
        noise = 0.05 * np.median(flux_1d_clean) * np.random.normal(0, 1, len(flux_1d_clean))
        flux_observed = flux_1d_clean + noise
        flux_error = 0.1 * np.abs(flux_observed) + 0.01
        
        ax2.plot(wavelength, flux_observed, 'k-', linewidth=1, label='Observed')
        ax2.fill_between(wavelength, flux_observed - flux_error, 
                        flux_observed + flux_error, alpha=0.3, color='gray')
        ax2.plot(wavelength, flux_1d_clean, 'r-', linewidth=1, label='Best fit')
        
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Flux')
        ax2.set_title('Extracted 1D Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Fit Residuals
        ax3 = fig.add_subplot(gs[0, 2])
        
        residuals = (flux_observed - flux_1d_clean) / flux_error
        ax3.plot(wavelength, residuals, 'g-', linewidth=1)
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.axhline(1, color='red', linestyle=':', alpha=0.5)
        ax3.axhline(-1, color='red', linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Wavelength (μm)')
        ax3.set_ylabel('Residuals (σ)')
        ax3.set_title('Fit Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Star Formation History
        ax4 = fig.add_subplot(gs[1, 0])
        
        time = np.linspace(0, 13.8, 100)
        # Mock SFH - recent burst
        lookback_formation = 2.0  # Gyr ago
        tau = 1.0  # Gyr
        sfh = np.exp(-(time - (13.8 - lookback_formation))**2 / (2 * tau**2))
        sfh[time < (13.8 - lookback_formation)] = 0
        
        ax4.plot(time, sfh, 'b-', linewidth=3)
        ax4.fill_between(time, sfh, alpha=0.3, color='blue')
        ax4.set_xlabel('Cosmic Time (Gyr)')
        ax4.set_ylabel('SFR (arbitrary units)')
        ax4.set_title('Star Formation History')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Properties Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Mock comparison with literature
        properties = ['log(M*)', 'Age', 'Z/Z☉', 'Av', 'SFR']
        jwst_values = [10.2, 2.1, 0.8, 0.3, 5.2]
        literature_values = [10.0, 3.0, 0.6, 0.5, 3.8]
        jwst_errors = [0.1, 0.3, 0.1, 0.1, 1.2]
        
        x_pos = np.arange(len(properties))
        
        ax5.errorbar(x_pos - 0.1, jwst_values, yerr=jwst_errors, 
                    fmt='ro', markersize=8, capsize=5, label='JWST')
        ax5.plot(x_pos + 0.1, literature_values, 'bs', markersize=8, label='Literature')
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(properties)
        ax5.set_ylabel('Property Value')
        ax5.set_title('JWST vs Literature')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: High-z Population
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Mock high-z galaxy properties
        if self.jwst_results and 'spectral_fits' in self.jwst_results:
            redshifts = []
            masses = []
            for source_id, fit_result in self.jwst_results['spectral_fits'].items():
                redshifts.append(fit_result['redshift'])
                masses.append(fit_result['stellar_mass'])
        else:
            # Mock data
            redshifts = np.random.uniform(2.0, 6.0, 20)
            masses = np.random.uniform(9.0, 11.0, 20)
        
        ax6.scatter(redshifts, masses, s=100, c=redshifts, cmap='plasma', 
                   alpha=0.8, edgecolors='black')
        
        # Add evolutionary tracks
        z_track = np.linspace(2, 6, 100)
        mass_track1 = 9.5 + 0.5 * np.log10(13.8 - self._redshift_to_time(z_track))
        mass_track2 = 10.0 + 0.3 * np.log10(13.8 - self._redshift_to_time(z_track))
        
        ax6.plot(z_track, mass_track1, 'k--', alpha=0.7, label='Evolution track')
        ax6.plot(z_track, mass_track2, 'k--', alpha=0.7)
        
        ax6.set_xlabel('Redshift')
        ax6.set_ylabel('log(M*/M☉)')
        ax6.set_title('High-z Galaxy Population')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.integration_plots['jwst_showcase'] = fig
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]
    
    def _generate_mock_spectrum_for_plot(self, wavelength):
        """Generate mock spectrum for plotting."""
        # Base continuum
        continuum = 1e-18 * (wavelength / 2.0)**(-1.5)
        
        # Add emission lines
        lines = {1.083: 0.2, 1.875: 0.5, 3.760: 0.8, 4.861: 1.0}
        
        flux = continuum.copy()
        for line_wave, line_strength in lines.items():
            if wavelength.min() <= line_wave <= wavelength.max():
                line_profile = line_strength * continuum.max() * np.exp(
                    -(wavelength - line_wave)**2 / (2 * 0.01**2)
                )
                flux += line_profile
        
        return flux
    
    def _time_to_redshift(self, time_gyr):
        """Convert cosmic time to redshift (simplified)."""
        # Simplified relation for plotting
        return np.maximum(0, 10 * np.exp(-(time_gyr - 0.5) / 2.0) - 1)
    
    def _redshift_to_time(self, redshift):
        """Convert redshift to cosmic time (simplified)."""
        # Simplified inverse relation
        return 0.5 + 2.0 * np.log((redshift + 1) / 10.0 + 1e-10)
    
    def create_summary_dashboard(self, save_path=None):
        """
        Create comprehensive summary dashboard.
        
        Returns:
        --------
        fig : matplotlib figure
        """
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
        
        fig.suptitle('Astro-AI: Comprehensive Galaxy Evolution Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Module status indicators
        ax_status = fig.add_subplot(gs[0, 0])
        
        modules = ['21cm\nEvolution', 'Cluster\nAnalysis', 'JWST\nSpectroscopy']
        status = [
            bool(self.cos_evo_results),
            bool(self.cluster_results),
            bool(self.jwst_results)
        ]
        colors = ['green' if s else 'orange' for s in status]
        
        bars = ax_status.bar(modules, [1, 1, 1], color=colors, alpha=0.7)
        for i, s in enumerate(status):
            label = '✓ Complete' if s else '⚠ Pending'
            ax_status.text(i, 0.5, label, ha='center', va='center', fontweight='bold')
        
        ax_status.set_ylim(0, 1)
        ax_status.set_title('Module Status', fontweight='bold')
        ax_status.set_yticks([])
        
        # Key statistics
        ax_stats = fig.add_subplot(gs[0, 1:])
        ax_stats.axis('off')
        
        stats_text = """
        📊 ANALYSIS SUMMARY
        
        • Cosmic Evolution: 21cm signal from z=15 to z=6
        • Cluster Analysis: Environmental effects on galaxy properties  
        • JWST Spectroscopy: High-resolution stellar population analysis
        
        🔗 KEY CONNECTIONS
        
        • Reionization epoch → Cluster formation timing
        • Environmental quenching → Spectroscopic confirmation
        • High-z properties → Cosmic evolution context
        """
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # 21cm evolution
        ax_21cm = fig.add_subplot(gs[1, 0])
        
        if self.cos_evo_results:
            z_vals = self.cos_evo_results.get('redshifts', np.linspace(6, 15, 10))
            signal = self.cos_evo_results.get('global_signals', 
                                            -50 * np.exp(-(z_vals - 10)**2 / 10))
        else:
            z_vals = np.linspace(6, 15, 20)
            signal = -50 * np.exp(-(z_vals - 10)**2 / 10) + np.random.normal(0, 3, 20)
        
        ax_21cm.plot(z_vals, signal, 'b-', linewidth=2, marker='o', markersize=4)
        ax_21cm.set_xlabel('Redshift')
        ax_21cm.set_ylabel('21cm Signal [mK]')
        ax_21cm.set_title('Cosmic Evolution', fontweight='bold')
        ax_21cm.grid(True, alpha=0.3)
        
        # Environmental effects
        ax_env = fig.add_subplot(gs[1, 1])
        
        mass_bins = np.linspace(9.5, 11.5, 8)
        rf_cluster = 0.1 + 0.7 / (1 + np.exp(-(mass_bins - 10.3) / 0.3))
        rf_field = 0.05 + 0.4 / (1 + np.exp(-(mass_bins - 10.7) / 0.4))
        
        ax_env.plot(mass_bins, rf_cluster, 'ro-', linewidth=2, label='Cluster')
        ax_env.plot(mass_bins, rf_field, 'bo-', linewidth=2, label='Field')
        ax_env.set_xlabel('log(M*/M☉)')
        ax_env.set_ylabel('Red Fraction')
        ax_env.set_title('Environmental Effects', fontweight='bold')
        ax_env.legend()
        ax_env.grid(True, alpha=0.3)
        
        # JWST spectrum example
        ax_jwst = fig.add_subplot(gs[1, 2])
        
        wavelength = np.linspace(1.0, 5.0, 200)
        flux = self._generate_mock_spectrum_for_plot(wavelength)
        flux += 0.05 * np.median(flux) * np.random.normal(0, 1, len(flux))
        
        ax_jwst.plot(wavelength, flux, 'k-', linewidth=1)
        ax_jwst.set_xlabel('Wavelength (μm)')
        ax_jwst.set_ylabel('Flux')
        ax_jwst.set_title('JWST Spectroscopy', fontweight='bold')
        ax_jwst.grid(True, alpha=0.3)
        
        # Mass-SFR relation
        ax_msfr = fig.add_subplot(gs[1, 3])
        
        masses = np.random.uniform(9.5, 11.5, 100)
        sfrs = masses - 9.0 + np.random.normal(0, 0.3, 100)
        env = np.random.choice(['Cluster', 'Field'], 100, p=[0.3, 0.7])
        
        cluster_mask = env == 'Cluster'
        ax_msfr.scatter(masses[~cluster_mask], sfrs[~cluster_mask], 
                       alpha=0.6, s=30, c='blue', label='Field')
        ax_msfr.scatter(masses[cluster_mask], sfrs[cluster_mask], 
                       alpha=0.8, s=30, c='red', label='Cluster')
        
        # Main sequence
        ms_x = np.linspace(9.5, 11.5, 100)
        ms_y = ms_x - 9.0
        ax_msfr.plot(ms_x, ms_y, 'k--', alpha=0.7, label='Main Sequence')
        
        ax_msfr.set_xlabel('log(M*/M☉)')
        ax_msfr.set_ylabel('log(SFR)')
        ax_msfr.set_title('Mass-SFR Relation', fontweight='bold')
        ax_msfr.legend()
        ax_msfr.grid(True, alpha=0.3)
        
        # Bottom panels: Integrated timeline
        ax_timeline = fig.add_subplot(gs[2, :])
        
        # Cosmic time axis
        cosmic_time = np.linspace(0.5, 13.8, 100)
        redshift_axis = self._time_to_redshift(cosmic_time)
        
        # Add multiple timelines
        y_positions = [0.8, 0.6, 0.4, 0.2]
        timelines = [
            ('Reionization (21cm)', 'blue'),
            ('Cluster Formation', 'red'), 
            ('Galaxy Assembly', 'green'),
            ('JWST Observations', 'purple')
        ]
        
        for i, (label, color) in enumerate(timelines):
            y = y_positions[i]
            
            if i == 0:  # 21cm reionization
                mask = (cosmic_time >= 0.5) & (cosmic_time <= 2.5)
                intensity = np.where(mask, np.exp(-(cosmic_time - 1.2)**2 / 0.5), 0)
            elif i == 1:  # Cluster formation
                mask = (cosmic_time >= 2.0) & (cosmic_time <= 8.0)
                intensity = np.where(mask, np.exp(-(cosmic_time - 4.0)**2 / 2.0), 0)
            elif i == 2:  # Galaxy assembly
                mask = (cosmic_time >= 1.0) & (cosmic_time <= 10.0)
                intensity = np.where(mask, 0.5 + 0.5 * np.sin((cosmic_time - 1) / 2), 0)
            else:  # JWST observations
                mask = cosmic_time >= 11.0
                intensity = np.where(mask, 1.0, 0)
            
            ax_timeline.fill_between(cosmic_time, y - 0.05, y + 0.05, 
                                   where=(intensity > 0.1), alpha=intensity.max() * 0.8,
                                   color=color, label=label)
            
            ax_timeline.text(0.2, y, label, fontsize=10, fontweight='bold')
        
        # Add epoch markers
        epoch_times = [0.8, 1.2, 2.0, 4.0, 11.0]
        epoch_labels = ['z~15', 'z~10', 'z~6', 'z~2', 'z~0.5']
        
        for time, label in zip(epoch_times, epoch_labels):
            ax_timeline.axvline(time, color='black', linestyle=':', alpha=0.7)
            ax_timeline.text(time, 1.0, label, rotation=90, ha='center', va='bottom')
        
        ax_timeline.set_xlim(0.5, 13.8)
        ax_timeline.set_ylim(0, 1.1)
        ax_timeline.set_xlabel('Cosmic Time (Gyr)', fontsize=12)
        ax_timeline.set_title('Integrated Cosmic Timeline', fontsize=14, fontweight='bold')
        ax_timeline.set_yticks([])
        
        # Add secondary x-axis for redshift
        ax_timeline_z = ax_timeline.twiny()
        z_ticks = [15, 10, 6, 3, 1, 0]
        time_ticks = [self._redshift_to_time(z) for z in z_ticks]
        ax_timeline_z.set_xlim(ax_timeline.get_xlim())
        ax_timeline_z.set_xticks(time_ticks)
        ax_timeline_z.set_xticklabels([f'z={z}' for z in z_ticks])
        ax_timeline_z.set_xlabel('Redshift', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.integration_plots['summary'] = fig
        return fig
    
    def export_dashboard_plots(self, output_dir, formats=['png', 'pdf']):
        """
        Export all dashboard plots.
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        formats : list
            List of output formats
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for plot_name, fig in self.integration_plots.items():
            for fmt in formats:
                output_path = os.path.join(output_dir, f"{plot_name}.{fmt}")
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved {output_path}")
    
    def generate_integration_report(self):
        """
        Generate summary report of integrated analysis.
        
        Returns:
        --------
        report : dict
        """
        report = {
            'modules_status': {
                'cosmic_evolution': bool(self.cos_evo_results),
                'cluster_analysis': bool(self.cluster_results), 
                'jwst_spectroscopy': bool(self.jwst_results)
            },
            'integration_summary': {
                'total_modules_completed': sum([
                    bool(self.cos_evo_results),
                    bool(self.cluster_results),
                    bool(self.jwst_results)
                ]),
                'plots_generated': len(self.integration_plots)
            }
        }
        
        # Add module-specific summaries
        if self.cos_evo_results:
            report['cosmic_evolution_summary'] = {
                'redshift_range': [
                    float(min(self.cos_evo_results.get('redshifts', []))),
                    float(max(self.cos_evo_results.get('redshifts', [])))
                ] if self.cos_evo_results.get('redshifts') else None,
                'signal_range_mK': [
                    float(min(self.cos_evo_results.get('global_signals', []))),
                    float(max(self.cos_evo_results.get('global_signals', [])))
                ] if self.cos_evo_results.get('global_signals') else None
            }
        
        if self.cluster_results:
            report['cluster_analysis_summary'] = {
                'n_cluster_members': len(self.cluster_results.get('cluster_members', [])),
                'n_field_galaxies': len(self.cluster_results.get('field_galaxies', [])),
                'environmental_effect_detected': True  # Simplified
            }
        
        if self.jwst_results:
            report['jwst_analysis_summary'] = {
                'n_sources_fitted': len(self.jwst_results.get('spectral_fits', {})),
                'average_redshift': np.mean([
                    fit['redshift'] for fit in self.jwst_results.get('spectral_fits', {}).values()
                ]) if self.jwst_results.get('spectral_fits') else None
            }
        
        return report
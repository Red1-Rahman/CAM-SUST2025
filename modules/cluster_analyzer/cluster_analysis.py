# Cluster Environment Analyzer Module for Astro-AI
# 
# This module integrates galaxy cluster analysis functionality including
# spatial distribution analysis, redshift detection, SED fitting with Bagpipes,
# and environmental effect studies.
# 
# Based on functions from galaxy_clusters_workshop_notebook_Participants.ipynb,
# Galaxy SEDs Fitting.ipynb, and Making model galaxies.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import bagpipes as pipes
    HAVE_BAGPIPES = True
except ImportError:
    HAVE_BAGPIPES = False
    logger.warning("bagpipes not available. Using simulation mode.")

try:
    from astropy.io import fits
    from astropy.table import Table
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False

class ClusterAnalyzer:
    """
    Galaxy Cluster Environment Analysis.
    
    This class provides methods for analyzing galaxy clusters including
    spatial distribution, redshift analysis, SED fitting, and environmental
    effects on galaxy properties.
    """
    
    def __init__(self):
        """Initialize the cluster analyzer."""
        self.data = None
        self.cluster_members = {}
        self.sed_results = {}
        self.analysis_results = {}
        
        # Default column mapping
        self.column_mapping = {
            "cluster_id": "cl_id",
            "halo_mass": "halo_mass", 
            "stellar_mass": "stellar_mass",
            "log10_stellar_mass": None,
            "g_mag": "g_mag",
            "r_mag": "r_mag", 
            "redshift": "redshift",
            "ra": "ra",
            "dec": "dec"
        }
        
        # Analysis parameters
        self.color_cut = 0.9  # g-r color cut for red/blue separation
        self.mass_bins = np.arange(10.0, 12.5, 0.2)  # Log stellar mass bins
    
    def load_data(self, data_source, file_format='csv'):
        """
        Load galaxy catalog data.
        
        Parameters:
        -----------
        data_source : str or pandas.DataFrame
            Path to data file or DataFrame
        file_format : str
            Format of input file ('csv', 'fits')
            
        Returns:
        --------
        success : bool
        """
        try:
            if isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            elif file_format.lower() == 'csv':
                self.data = pd.read_csv(data_source)
            elif file_format.lower() == 'fits' and HAVE_ASTROPY:
                with fits.open(data_source) as hdul:
                    self.data = Table(hdul[1].data).to_pandas()
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Apply column mapping and compute derived quantities
            self._process_data()
            
            print(f"Loaded {len(self.data):,} galaxies")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _process_data(self):
        """Process loaded data and compute derived quantities."""
        # Standardize column names
        for key, col in self.column_mapping.items():
            if col and col in self.data.columns:
                self.data[key] = self.data[col]
        
        # Compute log stellar mass if needed
        if 'log10_stellar_mass' in self.data.columns:
            self.data['log10M'] = self.data['log10_stellar_mass']
        elif 'stellar_mass' in self.data.columns:
            self.data['log10M'] = np.log10(self.data['stellar_mass'])
        
        # Compute colors
        if 'g_mag' in self.data.columns and 'r_mag' in self.data.columns:
            self.data['g_r'] = self.data['g_mag'] - self.data['r_mag']
            self.data['is_red'] = (self.data['g_r'] >= self.color_cut)
        
        # Convert numeric columns
        numeric_cols = ['halo_mass', 'redshift', 'ra', 'dec', 'log10M']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
    
    def generate_mock_data(self, n_galaxies=1000, n_clusters=5):
        """
        Generate mock galaxy cluster data for testing.
        
        Parameters:
        -----------
        n_galaxies : int
            Total number of galaxies
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        success : bool
        """
        np.random.seed(42)
        
        # Cluster properties
        cluster_z = np.random.uniform(0.8, 2.0, n_clusters)
        cluster_ra = np.random.uniform(149, 151, n_clusters)
        cluster_dec = np.random.uniform(1.5, 2.5, n_clusters)
        cluster_mass = np.random.uniform(1e14, 1e15, n_clusters)
        
        # Generate galaxies
        data = []
        
        for i in range(n_galaxies):
            # Assign to cluster or field
            if np.random.random() < 0.3:  # 30% cluster members
                cluster_idx = np.random.randint(0, n_clusters)
                cl_id = f"cluster_{cluster_idx}"
                halo_mass = cluster_mass[cluster_idx]
                
                # Cluster members concentrated around center
                ra = np.random.normal(cluster_ra[cluster_idx], 0.1)
                dec = np.random.normal(cluster_dec[cluster_idx], 0.1)
                redshift = np.random.normal(cluster_z[cluster_idx], 0.02)
                
                # Cluster galaxies tend to be more massive and redder
                log10M = np.random.normal(10.5, 0.5)
                is_red_prob = 0.7
                
            else:  # Field galaxies
                cl_id = "field"
                halo_mass = np.random.uniform(1e12, 1e14)
                
                ra = np.random.uniform(149, 151)
                dec = np.random.uniform(1.5, 2.5)
                redshift = np.random.uniform(0.5, 2.5)
                
                # Field galaxies more diverse
                log10M = np.random.normal(10.0, 0.8)
                is_red_prob = 0.3
            
            # Generate photometry
            stellar_mass = 10**log10M
            
            # Basic color-magnitude relation
            r_mag = 25 - 2.5 * (log10M - 9) + np.random.normal(0, 0.3)
            
            if np.random.random() < is_red_prob:
                g_r = np.random.normal(1.0, 0.2)  # Red galaxies
            else:
                g_r = np.random.normal(0.5, 0.3)  # Blue galaxies
            
            g_mag = r_mag + g_r
            
            data.append({
                'galaxy_id': i,
                'cl_id': cl_id,
                'halo_mass': halo_mass,
                'stellar_mass': stellar_mass,
                'log10M': log10M,
                'ra': ra,
                'dec': dec,
                'redshift': redshift,
                'g_mag': g_mag,
                'r_mag': r_mag,
                'g_r': g_r,
                'is_red': g_r >= self.color_cut
            })
        
        self.data = pd.DataFrame(data)
        print(f"Generated {len(self.data):,} mock galaxies")
        return True
    
    def analyze_spatial_distribution(self):
        """
        Analyze spatial distribution of galaxies.
        
        Returns:
        --------
        results : dict
            Analysis results including overdensity maps
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Compute basic statistics
        ra_range = [self.data['ra'].min(), self.data['ra'].max()]
        dec_range = [self.data['dec'].min(), self.data['dec'].max()]
        
        # Create spatial density map
        H, xedges, yedges = np.histogram2d(
            self.data['ra'], self.data['dec'], 
            bins=20, density=True
        )
        
        # Find overdense regions
        threshold = np.percentile(H, 80)
        overdense_regions = H > threshold
        
        results = {
            'ra_range': ra_range,
            'dec_range': dec_range,
            'density_map': H,
            'ra_edges': xedges,
            'dec_edges': yedges,
            'overdense_regions': overdense_regions,
            'n_galaxies': len(self.data)
        }
        
        self.analysis_results['spatial'] = results
        return results
    
    def detect_clusters_redshift(self, z_bins=50, significance_threshold=3.0):
        """
        Detect galaxy clusters using redshift overdensities.
        
        Parameters:
        -----------
        z_bins : int
            Number of redshift bins
        significance_threshold : float
            Significance threshold for cluster detection
            
        Returns:
        --------
        clusters : dict
            Detected cluster properties
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Create redshift histogram
        z_range = [self.data['redshift'].min(), self.data['redshift'].max()]
        counts, bin_edges = np.histogram(self.data['redshift'], bins=z_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        # Smooth background
        from scipy.ndimage import gaussian_filter1d
        smooth_counts = gaussian_filter1d(counts.astype(float), sigma=2)
        
        # Find peaks
        residuals = counts - smooth_counts
        significance = residuals / np.sqrt(smooth_counts + 1)
        
        # Detect significant peaks
        peak_indices = []
        for i in range(1, len(significance) - 1):
            if (significance[i] > significance_threshold and
                significance[i] > significance[i-1] and
                significance[i] > significance[i+1]):
                peak_indices.append(i)
        
        # Extract cluster information
        detected_clusters = []
        for i, peak_idx in enumerate(peak_indices):
            z_cluster = bin_centers[peak_idx]
            n_members = counts[peak_idx]
            significance_val = significance[peak_idx]
            
            # Find members within ±3σ of peak
            z_width = 0.05  # Typical cluster velocity dispersion
            member_mask = np.abs(self.data['redshift'] - z_cluster) < z_width
            members = self.data[member_mask].copy()
            
            detected_clusters.append({
                'cluster_id': f'detected_{i}',
                'redshift': z_cluster,
                'n_members': len(members),
                'significance': significance_val,
                'members': members
            })
        
        results = {
            'z_histogram': {
                'counts': counts,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges
            },
            'background': smooth_counts,
            'significance': significance,
            'detected_clusters': detected_clusters,
            'n_clusters_detected': len(detected_clusters)
        }
        
        self.analysis_results['redshift_clustering'] = results
        return results
    
    def separate_cluster_field(self, cluster_z_tolerance=0.05):
        """
        Separate cluster and field galaxies.
        
        Parameters:
        -----------
        cluster_z_tolerance : float
            Redshift tolerance for cluster membership
            
        Returns:
        --------
        separation_results : dict
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # If we have detected clusters, use those
        if 'redshift_clustering' in self.analysis_results:
            cluster_results = self.analysis_results['redshift_clustering']
            
            cluster_members = pd.DataFrame()
            for cluster in cluster_results['detected_clusters']:
                cluster_members = pd.concat([cluster_members, cluster['members']])
            
            field_galaxies = self.data[~self.data.index.isin(cluster_members.index)]
            
        else:
            # Simple separation based on existing cluster_id
            if 'cl_id' in self.data.columns:
                cluster_members = self.data[self.data['cl_id'] != 'field']
                field_galaxies = self.data[self.data['cl_id'] == 'field']
            else:
                # No cluster info available, assign randomly for demonstration
                cluster_mask = np.random.choice([True, False], size=len(self.data), p=[0.3, 0.7])
                cluster_members = self.data[cluster_mask]
                field_galaxies = self.data[~cluster_mask]
        
        results = {
            'cluster_members': cluster_members,
            'field_galaxies': field_galaxies,
            'n_cluster': len(cluster_members),
            'n_field': len(field_galaxies),
            'cluster_fraction': len(cluster_members) / len(self.data)
        }
        
        self.cluster_members = results
        return results
    
    def setup_bagpipes_model(self, sfh_model='exponential', dust_model='Calzetti'):
        """
        Set up Bagpipes SED fitting model.
        
        Parameters:
        -----------
        sfh_model : str
            Star formation history model
        dust_model : str
            Dust attenuation model
            
        Returns:
        --------
        fit_instructions : dict
        """
        # SFH component
        if sfh_model == 'exponential':
            exp = {
                "age": (0.1, 15.),
                "tau": (0.3, 10.),
                "massformed": (1., 15.),
                "metallicity": (0., 2.5)
            }
            sfh_component = {"exponential": exp}
        elif sfh_model == 'double_power_law':
            dblplaw = {
                "tau": (0.3, 10.),
                "alpha": (0.01, 1000.),
                "beta": (0.01, 1000.),
                "massformed": (1., 15.),
                "metallicity": (0., 2.5)
            }
            sfh_component = {"dblplaw": dblplaw}
        else:
            raise ValueError(f"Unsupported SFH model: {sfh_model}")
        
        # Dust component
        dust = {
            "type": dust_model,
            "Av": (0., 2.)
        }
        
        # Complete fit instructions
        fit_instructions = {
            "redshift": (0., 10.),
            "dust": dust,
            **sfh_component
        }
        
        return fit_instructions
    
    def run_sed_fitting(self, galaxy_subset=None, fit_instructions=None, n_live=400):
        """
        Run SED fitting with Bagpipes.
        
        Parameters:
        -----------
        galaxy_subset : DataFrame, optional
            Subset of galaxies to fit. If None, fits all.
        fit_instructions : dict, optional
            Bagpipes fit instructions
        n_live : int
            Number of live points for nested sampling
            
        Returns:
        --------
        results : dict
        """
        if not HAVE_BAGPIPES:
            return self._mock_sed_fitting_results(galaxy_subset)
        
        if galaxy_subset is None:
            galaxy_subset = self.data
        
        if fit_instructions is None:
            fit_instructions = self.setup_bagpipes_model()
        
        # Prepare photometry (mock for now)
        results = {}
        
        for idx, galaxy in galaxy_subset.iterrows():
            try:
                # Mock photometry array (flux, flux_err)
                photometry = self._prepare_photometry(galaxy)
                
                # Create galaxy object
                galaxy_obj = pipes.galaxy(
                    str(galaxy['galaxy_id']), 
                    lambda x: photometry,
                    spectrum_exists=False
                )
                
                # Run fit
                fit = pipes.fit(galaxy_obj, fit_instructions)
                fit.fit(verbose=False, n_live=n_live)
                
                # Extract results
                posterior = fit.posterior
                results[galaxy['galaxy_id']] = {
                    'stellar_mass': posterior.samples['stellar_mass'].mean(),
                    'stellar_mass_err': posterior.samples['stellar_mass'].std(),
                    'sfr': posterior.samples['sfr'].mean() if 'sfr' in posterior.samples else 0,
                    'sfr_err': posterior.samples['sfr'].std() if 'sfr' in posterior.samples else 0,
                    'age': posterior.samples['age'].mean(),
                    'age_err': posterior.samples['age'].std(),
                    'metallicity': posterior.samples['metallicity'].mean(),
                    'metallicity_err': posterior.samples['metallicity'].std(),
                    'av': posterior.samples['dust:Av'].mean(),
                    'av_err': posterior.samples['dust:Av'].std(),
                }
                
            except Exception as e:
                print(f"Error fitting galaxy {galaxy['galaxy_id']}: {e}")
                continue
        
        self.sed_results = results
        return results
    
    def _mock_sed_fitting_results(self, galaxy_subset):
        """Generate mock SED fitting results for testing."""
        if galaxy_subset is None:
            galaxy_subset = self.data
        
        results = {}
        
        for idx, galaxy in galaxy_subset.iterrows():
            # Generate realistic mock results based on input properties
            log_mass = galaxy.get('log10M', np.random.normal(10.0, 0.5))
            is_red = galaxy.get('is_red', np.random.choice([True, False]))
            
            # Mock stellar mass (with some scatter)
            stellar_mass = log_mass + np.random.normal(0, 0.2)
            
            # Mock SFR (red galaxies have lower SFR)
            if is_red:
                sfr = np.random.lognormal(-1, 0.5)  # Lower SFR
                age = np.random.uniform(3, 10)      # Older
            else:
                sfr = np.random.lognormal(0, 0.5)   # Higher SFR
                age = np.random.uniform(0.5, 5)     # Younger
            
            # Mock other properties
            metallicity = np.random.uniform(0.5, 1.5)
            av = np.random.exponential(0.3)
            
            results[galaxy['galaxy_id']] = {
                'stellar_mass': stellar_mass,
                'stellar_mass_err': 0.1,
                'sfr': sfr,
                'sfr_err': sfr * 0.3,
                'age': age,
                'age_err': age * 0.2,
                'metallicity': metallicity,
                'metallicity_err': 0.1,
                'av': av,
                'av_err': 0.05
            }
        
        self.sed_results = results
        return results
    
    def _prepare_photometry(self, galaxy):
        """Prepare photometry array for Bagpipes fitting."""
        # Mock photometry based on galaxy properties
        bands = ['g', 'r', 'i', 'z']  # Simplified
        fluxes = []
        flux_errs = []
        
        for band in bands:
            if f'{band}_mag' in galaxy:
                mag = galaxy[f'{band}_mag']
                flux = 10**(-0.4 * (mag - 25))  # Convert to microJy
                flux_err = flux * 0.1  # 10% error
            else:
                flux = np.random.lognormal(0, 1)
                flux_err = flux * 0.1
            
            fluxes.append(flux)
            flux_errs.append(flux_err)
        
        return np.column_stack([fluxes, flux_errs])
    
    def analyze_stellar_mass_functions(self, mass_bins=None):
        """
        Compute stellar mass functions for cluster and field populations.
        
        Parameters:
        -----------
        mass_bins : array, optional
            Stellar mass bins
            
        Returns:
        --------
        smf_results : dict
        """
        if mass_bins is None:
            mass_bins = self.mass_bins
        
        if not self.cluster_members:
            self.separate_cluster_field()
        
        cluster_gals = self.cluster_members['cluster_members']
        field_gals = self.cluster_members['field_galaxies']
        
        # Compute mass functions
        def compute_smf(galaxies, bins):
            if 'log10M' not in galaxies.columns:
                return np.zeros(len(bins)-1), bins[:-1] + np.diff(bins)/2
            
            counts, _ = np.histogram(galaxies['log10M'].dropna(), bins=bins)
            bin_centers = bins[:-1] + np.diff(bins) / 2
            
            # Convert to number density (simplified)
            volume = 1.0  # Assume unit volume for now
            phi = counts / volume / np.diff(bins)
            
            return phi, bin_centers
        
        phi_cluster, mass_centers = compute_smf(cluster_gals, mass_bins)
        phi_field, _ = compute_smf(field_gals, mass_bins)
        
        # Separate by color if available
        phi_cluster_red = phi_cluster_blue = np.zeros_like(phi_cluster)
        phi_field_red = phi_field_blue = np.zeros_like(phi_field)
        
        if 'is_red' in cluster_gals.columns:
            phi_cluster_red, _ = compute_smf(cluster_gals[cluster_gals['is_red']], mass_bins)
            phi_cluster_blue, _ = compute_smf(cluster_gals[~cluster_gals['is_red']], mass_bins)
        
        if 'is_red' in field_gals.columns:
            phi_field_red, _ = compute_smf(field_gals[field_gals['is_red']], mass_bins)
            phi_field_blue, _ = compute_smf(field_gals[~field_gals['is_red']], mass_bins)
        
        results = {
            'mass_bins': mass_bins,
            'mass_centers': mass_centers,
            'cluster': {
                'total': phi_cluster,
                'red': phi_cluster_red,
                'blue': phi_cluster_blue
            },
            'field': {
                'total': phi_field,
                'red': phi_field_red,
                'blue': phi_field_blue
            }
        }
        
        self.analysis_results['stellar_mass_functions'] = results
        return results
    
    def compute_red_fraction(self, mass_bins=None):
        """
        Compute red fraction as a function of stellar mass and environment.
        
        Parameters:
        -----------
        mass_bins : array, optional
            Stellar mass bins
            
        Returns:
        --------
        red_fraction_results : dict
        """
        if mass_bins is None:
            mass_bins = self.mass_bins
        
        if not self.cluster_members:
            self.separate_cluster_field()
        
        cluster_gals = self.cluster_members['cluster_members']
        field_gals = self.cluster_members['field_galaxies']
        
        def compute_red_frac(galaxies, bins):
            if 'is_red' not in galaxies.columns or 'log10M' not in galaxies.columns:
                return np.zeros(len(bins)-1), np.zeros(len(bins)-1)
            
            bin_centers = bins[:-1] + np.diff(bins) / 2
            red_fractions = []
            red_frac_errs = []
            
            for i in range(len(bins)-1):
                mask = (galaxies['log10M'] >= bins[i]) & (galaxies['log10M'] < bins[i+1])
                gals_in_bin = galaxies[mask]
                
                if len(gals_in_bin) > 0:
                    n_red = gals_in_bin['is_red'].sum()
                    n_total = len(gals_in_bin)
                    red_frac = n_red / n_total
                    
                    # Binomial error
                    red_frac_err = np.sqrt(red_frac * (1 - red_frac) / n_total)
                else:
                    red_frac = 0
                    red_frac_err = 0
                
                red_fractions.append(red_frac)
                red_frac_errs.append(red_frac_err)
            
            return np.array(red_fractions), np.array(red_frac_errs)
        
        rf_cluster, rf_cluster_err = compute_red_frac(cluster_gals, mass_bins)
        rf_field, rf_field_err = compute_red_frac(field_gals, mass_bins)
        
        # Compute environmental effect (cluster - field)
        delta_rf = rf_cluster - rf_field
        delta_rf_err = np.sqrt(rf_cluster_err**2 + rf_field_err**2)
        
        results = {
            'mass_bins': mass_bins,
            'mass_centers': mass_bins[:-1] + np.diff(mass_bins) / 2,
            'cluster': {
                'red_fraction': rf_cluster,
                'red_fraction_err': rf_cluster_err
            },
            'field': {
                'red_fraction': rf_field,
                'red_fraction_err': rf_field_err
            },
            'environmental_effect': {
                'delta_red_fraction': delta_rf,
                'delta_red_fraction_err': delta_rf_err
            }
        }
        
        self.analysis_results['red_fraction'] = results
        return results
    
    def plot_spatial_distribution(self, save_path=None):
        """Plot spatial distribution of galaxies."""
        if 'spatial' not in self.analysis_results:
            self.analyze_spatial_distribution()
        
        results = self.analysis_results['spatial']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        if not self.cluster_members:
            self.separate_cluster_field()
        
        cluster_gals = self.cluster_members['cluster_members']
        field_gals = self.cluster_members['field_galaxies']
        
        ax1.scatter(field_gals['ra'], field_gals['dec'], 
                   alpha=0.6, s=20, c='gray', label='Field galaxies')
        ax1.scatter(cluster_gals['ra'], cluster_gals['dec'], 
                   alpha=0.8, s=30, c='red', label='Cluster members')
        
        ax1.set_xlabel('RA (degrees)')
        ax1.set_ylabel('Dec (degrees)')
        ax1.set_title('Galaxy Spatial Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Density map
        im = ax2.imshow(results['density_map'].T, origin='lower',
                       extent=[results['ra_edges'][0], results['ra_edges'][-1],
                              results['dec_edges'][0], results['dec_edges'][-1]],
                       cmap='viridis', aspect='auto')
        
        ax2.set_xlabel('RA (degrees)')
        ax2.set_ylabel('Dec (degrees)')
        ax2.set_title('Galaxy Density Map')
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Number Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_color_magnitude_diagram(self, save_path=None):
        """Plot color-magnitude diagram."""
        if not self.cluster_members:
            self.separate_cluster_field()
        
        cluster_gals = self.cluster_members['cluster_members']
        field_gals = self.cluster_members['field_galaxies']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot field galaxies
        if 'r_mag' in field_gals.columns and 'g_r' in field_gals.columns:
            scatter_field = ax.scatter(field_gals['r_mag'], field_gals['g_r'],
                                     alpha=0.6, s=20, c='lightblue', 
                                     label=f'Field ({len(field_gals)})')
        
        # Plot cluster galaxies  
        if 'r_mag' in cluster_gals.columns and 'g_r' in cluster_gals.columns:
            scatter_cluster = ax.scatter(cluster_gals['r_mag'], cluster_gals['g_r'],
                                       alpha=0.8, s=30, c='red',
                                       label=f'Cluster ({len(cluster_gals)})')
        
        # Add red sequence line
        r_range = np.linspace(18, 25, 100)
        red_sequence = 1.0 + 0.05 * (r_range - 20)  # Simple red sequence
        ax.plot(r_range, red_sequence, 'k--', alpha=0.7, label='Red sequence')
        
        # Add color cut line
        ax.axhline(self.color_cut, color='orange', linestyle=':', 
                  alpha=0.7, label=f'Color cut (g-r = {self.color_cut})')
        
        ax.set_xlabel('r magnitude')
        ax.set_ylabel('g - r color')
        ax.set_title('Color-Magnitude Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_stellar_mass_functions(self, save_path=None):
        """Plot stellar mass functions."""
        if 'stellar_mass_functions' not in self.analysis_results:
            self.analyze_stellar_mass_functions()
        
        results = self.analysis_results['stellar_mass_functions']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        mass_centers = results['mass_centers']
        
        # Total SMFs
        ax1.semilogy(mass_centers, results['cluster']['total'], 
                    'ro-', label='Cluster total', markersize=6)
        ax1.semilogy(mass_centers, results['field']['total'], 
                    'bo-', label='Field total', markersize=6)
        
        ax1.set_xlabel('log(M*/M☉)')
        ax1.set_ylabel('φ [Mpc⁻³ dex⁻¹]')
        ax1.set_title('Stellar Mass Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # By color
        ax2.semilogy(mass_centers, results['cluster']['red'], 
                    'r-', label='Cluster red', linewidth=2)
        ax2.semilogy(mass_centers, results['cluster']['blue'], 
                    'r--', label='Cluster blue', linewidth=2)
        ax2.semilogy(mass_centers, results['field']['red'], 
                    'b-', label='Field red', linewidth=2)
        ax2.semilogy(mass_centers, results['field']['blue'], 
                    'b--', label='Field blue', linewidth=2)
        
        ax2.set_xlabel('log(M*/M☉)')
        ax2.set_ylabel('φ [Mpc⁻³ dex⁻¹]')
        ax2.set_title('SMFs by Color')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_red_fraction(self, save_path=None):
        """Plot red fraction vs stellar mass."""
        if 'red_fraction' not in self.analysis_results:
            self.compute_red_fraction()
        
        results = self.analysis_results['red_fraction']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        mass_centers = results['mass_centers']
        
        # Red fraction by environment
        ax1.errorbar(mass_centers, results['cluster']['red_fraction'],
                    yerr=results['cluster']['red_fraction_err'],
                    fmt='ro-', label='Cluster', markersize=6, capsize=3)
        ax1.errorbar(mass_centers, results['field']['red_fraction'],
                    yerr=results['field']['red_fraction_err'],
                    fmt='bo-', label='Field', markersize=6, capsize=3)
        
        ax1.set_xlabel('log(M*/M☉)')
        ax1.set_ylabel('Red Fraction')
        ax1.set_title('Red Fraction vs Stellar Mass')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Environmental effect
        ax2.errorbar(mass_centers, results['environmental_effect']['delta_red_fraction'],
                    yerr=results['environmental_effect']['delta_red_fraction_err'],
                    fmt='go-', markersize=6, capsize=3)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('log(M*/M☉)')
        ax2.set_ylabel('Δ(Red Fraction) = Cluster - Field')
        ax2.set_title('Environmental Quenching Effect')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def generate_summary_report(self):
        """Generate summary report of cluster analysis."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Basic statistics
        if not self.cluster_members:
            self.separate_cluster_field()
        
        cluster_gals = self.cluster_members['cluster_members']
        field_gals = self.cluster_members['field_galaxies']
        
        report = {
            'data_summary': {
                'total_galaxies': len(self.data),
                'cluster_members': len(cluster_gals),
                'field_galaxies': len(field_gals),
                'cluster_fraction': len(cluster_gals) / len(self.data)
            }
        }
        
        # Color statistics
        if 'is_red' in self.data.columns:
            cluster_red_frac = cluster_gals['is_red'].mean() if len(cluster_gals) > 0 else 0
            field_red_frac = field_gals['is_red'].mean() if len(field_gals) > 0 else 0
            
            report['color_statistics'] = {
                'cluster_red_fraction': cluster_red_frac,
                'field_red_fraction': field_red_frac,
                'environmental_effect': cluster_red_frac - field_red_frac
            }
        
        # Mass statistics
        if 'log10M' in self.data.columns:
            report['mass_statistics'] = {
                'cluster_median_mass': cluster_gals['log10M'].median() if len(cluster_gals) > 0 else 0,
                'field_median_mass': field_gals['log10M'].median() if len(field_gals) > 0 else 0,
                'mass_range': [self.data['log10M'].min(), self.data['log10M'].max()]
            }
        
        # SED fitting results summary
        if self.sed_results:
            n_fitted = len(self.sed_results)
            avg_stellar_mass = np.mean([r['stellar_mass'] for r in self.sed_results.values()])
            avg_sfr = np.mean([r['sfr'] for r in self.sed_results.values()])
            
            report['sed_fitting'] = {
                'n_galaxies_fitted': n_fitted,
                'average_stellar_mass': avg_stellar_mass,
                'average_sfr': avg_sfr
            }
        
        return report
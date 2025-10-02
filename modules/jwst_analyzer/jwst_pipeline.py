# JWST Spectrum Analyzer Module for Astro-AI
# 
# This module integrates JWST NIRSpec pipeline functionality including
# data reduction, optimal 1D extraction, and spectral fitting with Bagpipes.
# 
# Based on functions from reduce_data_st.ipynb, optimal_1d_extraction.ipynb,
# and NGSF spectral fitting tools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

try:
    from astropy.io import fits
    from astropy.table import Table
    import astropy.units as u
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False
    print("Warning: astropy not available.")

try:
    import bagpipes as pipes
    HAVE_BAGPIPES = True
except ImportError:
    HAVE_BAGPIPES = False
    print("Warning: bagpipes not available. Using simulation mode.")

try:
    from jwst.datamodels import ImageModel
    from jwst import datamodels
    HAVE_JWST_PIPELINE = True
except ImportError:
    HAVE_JWST_PIPELINE = False
    print("Warning: JWST pipeline not available. Using simulation mode.")

class JWSTAnalyzer:
    """
    JWST NIRSpec Data Analysis Pipeline.
    
    This class provides methods for reducing JWST NIRSpec data,
    extracting 1D spectra, and performing spectral fitting.
    """
    
    def __init__(self):
        """Initialize the JWST analyzer."""
        self.raw_data = None
        self.reduced_data = {}
        self.extracted_spectra = {}
        self.spectral_fits = {}
        self.pipeline_status = {
            'stage1': False,
            'stage2': False, 
            'stage3': False,
            'extraction': False
        }
        
        # Default extraction parameters
        self.extraction_params = {
            'profile_sigma': 2.0,
            'bg_offset': 5,
            'snr_threshold': 3.0,
            'profile_slice': [1.0, 5.0],
            'center_limit': 3
        }
        
        # Default spectral fitting parameters
        self.fitting_params = {
            'z_range': (0.0, 10.0),
            'age_range': (0.1, 15.0),
            'tau_range': (0.3, 10.0),
            'mass_range': (1.0, 15.0),
            'av_range': (0.0, 3.0)
        }
    
    def load_jwst_data(self, data_path, data_type='rate'):
        """
        Load JWST data file.
        
        Parameters:
        -----------
        data_path : str
            Path to JWST data file
        data_type : str
            Type of JWST data ('uncal', 'rate', 'cal', 'x1d')
            
        Returns:
        --------
        success : bool
        """
        try:
            if HAVE_ASTROPY:
                if data_type in ['uncal', 'rate', 'cal']:
                    # Load 2D spectroscopic data
                    with fits.open(data_path) as hdul:
                        self.raw_data = {
                            'data': hdul['SCI'].data,
                            'error': hdul['ERR'].data if 'ERR' in hdul else None,
                            'dq': hdul['DQ'].data if 'DQ' in hdul else None,
                            'header': hdul['PRIMARY'].header,
                            'sci_header': hdul['SCI'].header,
                            'data_type': data_type,
                            'file_path': data_path
                        }
                elif data_type == 'x1d':
                    # Load 1D extracted spectrum
                    with fits.open(data_path) as hdul:
                        table_data = Table(hdul['EXTRACT1D'].data)
                        self.extracted_spectra['loaded'] = {
                            'wavelength': table_data['WAVELENGTH'],
                            'flux': table_data['FLUX'],
                            'flux_error': table_data['FLUX_ERROR'],
                            'header': hdul['PRIMARY'].header
                        }
                        self.pipeline_status['extraction'] = True
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
            else:
                # Mock data loading
                self._generate_mock_jwst_data(data_type)
            
            print(f"Loaded JWST {data_type} data from {data_path}")
            return True
            
        except Exception as e:
            print(f"Error loading JWST data: {e}")
            return False
    
    def _generate_mock_jwst_data(self, data_type='rate'):
        """Generate mock JWST data for testing."""
        if data_type in ['uncal', 'rate', 'cal']:
            # Generate 2D spectroscopic data
            ny, nx = 50, 1000  # Typical NIRSpec dimensions
            
            # Create wavelength array (1-5 microns)
            wavelength = np.linspace(1.0, 5.0, nx)
            
            # Generate mock spectrum with emission lines
            flux_1d = self._generate_mock_spectrum(wavelength)
            
            # Create 2D spectrum with spatial profile
            y_profile = np.exp(-(np.arange(ny) - ny/2)**2 / (2 * 3**2))
            flux_2d = np.outer(y_profile, flux_1d)
            
            # Add noise
            noise = np.random.normal(0, 0.1 * np.median(flux_2d), flux_2d.shape)
            flux_2d += noise
            
            # Create error array
            error_2d = 0.1 * np.abs(flux_2d) + 0.01
            
            self.raw_data = {
                'data': flux_2d,
                'error': error_2d,
                'dq': np.zeros_like(flux_2d, dtype=int),
                'wavelength_2d': np.tile(wavelength, (ny, 1)),
                'data_type': data_type,
                'mock': True
            }
        
        elif data_type == 'x1d':
            # Generate 1D extracted spectrum
            wavelength = np.linspace(1.0, 5.0, 1000)
            flux = self._generate_mock_spectrum(wavelength)
            flux_error = 0.1 * np.abs(flux) + 0.01
            
            self.extracted_spectra['loaded'] = {
                'wavelength': wavelength,
                'flux': flux,
                'flux_error': flux_error,
                'mock': True
            }
            self.pipeline_status['extraction'] = True
    
    def _generate_mock_spectrum(self, wavelength):
        """Generate realistic mock galaxy spectrum."""
        # Base continuum (blackbody-like)
        continuum = 1e-18 * (wavelength / 2.0)**(-2) * np.exp(-wavelength / 3.0)
        
        # Add emission lines (common NIR lines)
        lines = {
            1.083: 0.2,   # He I
            1.282: 0.3,   # Pa β  
            1.875: 0.5,   # Pa α
            2.166: 0.1,   # Br γ
            3.760: 0.8,   # [O III]
            4.861: 1.0,   # Hβ
            4.959: 0.3,   # [O III]
            5.007: 0.9    # [O III]
        }
        
        flux = continuum.copy()
        
        for line_wave, line_strength in lines.items():
            if wavelength.min() <= line_wave <= wavelength.max():
                line_profile = line_strength * continuum.max() * np.exp(
                    -(wavelength - line_wave)**2 / (2 * 0.001**2)
                )
                flux += line_profile
        
        # Add noise
        flux += np.random.normal(0, 0.05 * np.median(flux), len(flux))
        
        return np.maximum(flux, 0)  # Ensure positive flux
    
    def run_pipeline_stage1(self, output_path=None):
        """
        Run JWST Pipeline Stage 1: Detector-level processing.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save Stage 1 output
            
        Returns:
        --------
        success : bool
        """
        if self.raw_data is None:
            raise ValueError("No raw data loaded")
        
        if not HAVE_JWST_PIPELINE:
            # Mock Stage 1 processing
            print("Mock Stage 1: Detector processing...")
            self.reduced_data['stage1'] = {
                'data': self.raw_data['data'].copy(),
                'error': self.raw_data['error'].copy() if self.raw_data['error'] is not None else None,
                'dq': self.raw_data['dq'].copy() if self.raw_data['dq'] is not None else None,
                'processing_steps': ['dq_init', 'saturation', 'superbias', 'refpix', 'linearity', 'dark_current', 'jump', 'ramp_fit']
            }
            self.pipeline_status['stage1'] = True
            return True
        
        try:
            # Actual JWST pipeline Stage 1
            from jwst.pipeline import Detector1Pipeline
            
            # Initialize pipeline
            pipeline = Detector1Pipeline()
            
            # Configure steps (example)
            pipeline.save_results = bool(output_path)
            if output_path:
                pipeline.output_dir = output_path
            
            # Run pipeline
            result = pipeline.run(self.raw_data['file_path'])
            
            # Store results
            self.reduced_data['stage1'] = result
            self.pipeline_status['stage1'] = True
            
            print("Stage 1 processing completed")
            return True
            
        except Exception as e:
            print(f"Error in Stage 1 processing: {e}")
            return False
    
    def run_pipeline_stage2(self, output_path=None):
        """
        Run JWST Pipeline Stage 2: Spectroscopic processing.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save Stage 2 output
            
        Returns:
        --------
        success : bool
        """
        if not self.pipeline_status['stage1']:
            print("Stage 1 must be completed first")
            return False
        
        if not HAVE_JWST_PIPELINE:
            # Mock Stage 2 processing
            print("Mock Stage 2: Spectroscopic processing...")
            
            # Apply wavelength calibration, flat fielding, etc.
            stage1_data = self.reduced_data['stage1']['data']
            
            # Mock flux calibration (multiply by sensitivity curve)
            sensitivity = np.ones_like(stage1_data) * 0.8
            calibrated_data = stage1_data * sensitivity
            
            self.reduced_data['stage2'] = {
                'data': calibrated_data,
                'error': self.reduced_data['stage1']['error'],
                'dq': self.reduced_data['stage1']['dq'],
                'wavelength': self.raw_data.get('wavelength_2d'),
                'processing_steps': ['assign_wcs', 'msa_flagging', 'extract_2d', 'flat_field', 'pathloss', 'photom', 'resample_spec']
            }
            self.pipeline_status['stage2'] = True
            return True
        
        try:
            # Actual JWST pipeline Stage 2
            from jwst.pipeline import Spec2Pipeline
            
            pipeline = Spec2Pipeline()
            pipeline.save_results = bool(output_path)
            if output_path:
                pipeline.output_dir = output_path
            
            result = pipeline.run(self.reduced_data['stage1'])
            
            self.reduced_data['stage2'] = result
            self.pipeline_status['stage2'] = True
            
            print("Stage 2 processing completed")
            return True
            
        except Exception as e:
            print(f"Error in Stage 2 processing: {e}")
            return False
    
    def run_pipeline_stage3(self, output_path=None):
        """
        Run JWST Pipeline Stage 3: Combine multiple exposures.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save Stage 3 output
            
        Returns:
        --------
        success : bool
        """
        if not self.pipeline_status['stage2']:
            print("Stage 2 must be completed first")
            return False
        
        if not HAVE_JWST_PIPELINE:
            # Mock Stage 3 processing
            print("Mock Stage 3: Combining exposures...")
            
            # For single exposure, just copy Stage 2 results
            self.reduced_data['stage3'] = self.reduced_data['stage2'].copy()
            self.reduced_data['stage3']['processing_steps'] = ['outlier_detection', 'resample_spec', 'extract_1d']
            self.pipeline_status['stage3'] = True
            return True
        
        try:
            # Actual JWST pipeline Stage 3
            from jwst.pipeline import Spec3Pipeline
            
            pipeline = Spec3Pipeline()
            pipeline.save_results = bool(output_path)
            if output_path:
                pipeline.output_dir = output_path
            
            result = pipeline.run([self.reduced_data['stage2']])
            
            self.reduced_data['stage3'] = result
            self.pipeline_status['stage3'] = True
            
            print("Stage 3 processing completed")
            return True
            
        except Exception as e:
            print(f"Error in Stage 3 processing: {e}")
            return False
    
    def extract_1d_spectrum(self, extraction_method='optimal', source_id='target'):
        """
        Extract 1D spectrum from 2D data.
        
        Parameters:
        -----------
        extraction_method : str
            Extraction method ('optimal', 'aperture', 'profile')
        source_id : str
            Identifier for the extracted source
            
        Returns:
        --------
        spectrum : dict
            Extracted 1D spectrum
        """
        if not self.pipeline_status['stage2']:
            print("Stage 2 processing required for extraction")
            return None
        
        # Get 2D data
        if 'stage3' in self.reduced_data:
            data_2d = self.reduced_data['stage3']['data']
            error_2d = self.reduced_data['stage3'].get('error')
            wavelength_2d = self.reduced_data['stage3'].get('wavelength')
        else:
            data_2d = self.reduced_data['stage2']['data']
            error_2d = self.reduced_data['stage2'].get('error')
            wavelength_2d = self.reduced_data['stage2'].get('wavelength')
        
        # Handle wavelength array
        if wavelength_2d is None:
            # Generate wavelength array
            nx = data_2d.shape[1]
            wavelength_1d = np.linspace(1.0, 5.0, nx)
        else:
            # Take wavelength from middle row
            wavelength_1d = wavelength_2d[data_2d.shape[0]//2, :]
        
        if extraction_method == 'optimal':
            spectrum = self._optimal_extraction(data_2d, error_2d, wavelength_1d)
        elif extraction_method == 'aperture':
            spectrum = self._aperture_extraction(data_2d, error_2d, wavelength_1d)
        elif extraction_method == 'profile':
            spectrum = self._profile_extraction(data_2d, error_2d, wavelength_1d)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        # Store result
        self.extracted_spectra[source_id] = spectrum
        self.pipeline_status['extraction'] = True
        
        return spectrum
    
    def _optimal_extraction(self, data_2d, error_2d, wavelength):
        """
        Perform optimal extraction (Horne 1986).
        
        Parameters:
        -----------
        data_2d : ndarray
            2D spectroscopic data
        error_2d : ndarray
            2D error array
        wavelength : ndarray
            Wavelength array
            
        Returns:
        --------
        spectrum : dict
        """
        ny, nx = data_2d.shape
        
        if error_2d is None:
            error_2d = np.sqrt(np.abs(data_2d) + 1)
        
        # Compute spatial profile
        profile = np.median(data_2d, axis=1)
        profile /= np.max(profile)
        
        # Gaussian fit to profile
        y_coords = np.arange(ny)
        center = ny // 2
        
        try:
            def gaussian(x, amp, center, sigma):
                return amp * np.exp(-(x - center)**2 / (2 * sigma**2))
            
            popt, _ = curve_fit(gaussian, y_coords, profile, 
                              p0=[1.0, center, self.extraction_params['profile_sigma']])
            fitted_profile = gaussian(y_coords, *popt)
        except:
            # Use simple Gaussian if fitting fails
            sigma = self.extraction_params['profile_sigma']
            fitted_profile = np.exp(-(y_coords - center)**2 / (2 * sigma**2))
        
        # Optimal extraction
        weights = fitted_profile[:, np.newaxis] / (error_2d**2 + 1e-10)
        
        flux_1d = np.sum(weights * data_2d, axis=0) / np.sum(weights, axis=0)
        
        # Compute errors
        flux_error_1d = np.sqrt(1.0 / np.sum(weights, axis=0))
        
        return {
            'wavelength': wavelength,
            'flux': flux_1d,
            'flux_error': flux_error_1d,
            'extraction_method': 'optimal',
            'profile': fitted_profile
        }
    
    def _aperture_extraction(self, data_2d, error_2d, wavelength):
        """Simple aperture extraction."""
        ny, nx = data_2d.shape
        center = ny // 2
        aperture_size = int(self.extraction_params['profile_sigma'] * 2)
        
        y_min = max(0, center - aperture_size)
        y_max = min(ny, center + aperture_size + 1)
        
        flux_1d = np.sum(data_2d[y_min:y_max, :], axis=0)
        
        if error_2d is not None:
            flux_error_1d = np.sqrt(np.sum(error_2d[y_min:y_max, :]**2, axis=0))
        else:
            flux_error_1d = np.sqrt(np.abs(flux_1d))
        
        return {
            'wavelength': wavelength,
            'flux': flux_1d,
            'flux_error': flux_error_1d,
            'extraction_method': 'aperture',
            'aperture_size': aperture_size
        }
    
    def _profile_extraction(self, data_2d, error_2d, wavelength):
        """Profile-weighted extraction."""
        ny, nx = data_2d.shape
        
        # Compute profile for each wavelength bin
        profiles = []
        flux_1d = []
        flux_error_1d = []
        
        for i in range(nx):
            column = data_2d[:, i]
            if error_2d is not None:
                error_column = error_2d[:, i]
            else:
                error_column = np.sqrt(np.abs(column) + 1)
            
            # Normalized profile
            profile = column / np.sum(column)
            profiles.append(profile)
            
            # Profile-weighted extraction
            weights = profile / (error_column**2 + 1e-10)
            flux = np.sum(weights * column) / np.sum(weights)
            flux_err = np.sqrt(1.0 / np.sum(weights))
            
            flux_1d.append(flux)
            flux_error_1d.append(flux_err)
        
        return {
            'wavelength': wavelength,
            'flux': np.array(flux_1d),
            'flux_error': np.array(flux_error_1d),
            'extraction_method': 'profile',
            'profiles': profiles
        }
    
    def setup_spectral_fitting_model(self, sfh_model='exponential', include_nebular=True):
        """
        Set up Bagpipes model for spectral fitting.
        
        Parameters:
        -----------
        sfh_model : str
            Star formation history model
        include_nebular : bool
            Include nebular emission
            
        Returns:
        --------
        fit_instructions : dict
        """
        # SFH component
        if sfh_model == 'exponential':
            exp = {
                "age": self.fitting_params['age_range'],
                "tau": self.fitting_params['tau_range'],
                "massformed": self.fitting_params['mass_range'],
                "metallicity": (0., 2.5)
            }
            sfh_component = {"exponential": exp}
        else:
            raise ValueError(f"Unsupported SFH model: {sfh_model}")
        
        # Dust component
        dust = {
            "type": "Calzetti",
            "Av": self.fitting_params['av_range']
        }
        
        # Build fit instructions
        fit_instructions = {
            "redshift": self.fitting_params['z_range'],
            "dust": dust,
            **sfh_component
        }
        
        # Add nebular emission if requested
        if include_nebular:
            nebular = {
                "logU": (-4., -1.)
            }
            fit_instructions["nebular"] = nebular
            fit_instructions["t_bc"] = (0.01, 0.1)  # Birth cloud lifetime
        
        return fit_instructions
    
    def fit_spectrum_bagpipes(self, source_id='target', fit_instructions=None, 
                             spec_resolution=1000, n_live=400):
        """
        Fit spectrum with Bagpipes.
        
        Parameters:
        -----------
        source_id : str
            ID of extracted spectrum to fit
        fit_instructions : dict, optional
            Bagpipes fit instructions
        spec_resolution : float
            Spectral resolution
        n_live : int
            Number of live points for nested sampling
            
        Returns:
        --------
        fit_results : dict
        """
        if source_id not in self.extracted_spectra:
            raise ValueError(f"No extracted spectrum found for {source_id}")
        
        spectrum = self.extracted_spectra[source_id]
        
        if not HAVE_BAGPIPES:
            return self._mock_spectral_fitting(spectrum, source_id)
        
        if fit_instructions is None:
            fit_instructions = self.setup_spectral_fitting_model()
        
        try:
            # Prepare spectrum for Bagpipes
            wavelength = spectrum['wavelength'] * 1e4  # Convert to Angstrom
            flux = spectrum['flux']
            flux_error = spectrum['flux_error']
            
            # Create spectrum array (wavelength, flux, flux_error)
            spec_array = np.column_stack([wavelength, flux, flux_error])
            
            # Create galaxy object with spectrum
            def load_spectrum(galaxy_id):
                return spec_array
            
            galaxy = pipes.galaxy(source_id, load_spectrum, 
                                spectrum_exists=True, 
                                spec_wavs=wavelength)
            
            # Run fit
            fit = pipes.fit(galaxy, fit_instructions, 
                          spec_res=spec_resolution)
            fit.fit(verbose=False, n_live=n_live)
            
            # Extract results
            posterior = fit.posterior
            
            fit_results = {
                'stellar_mass': posterior.samples['stellar_mass'].mean(),
                'stellar_mass_err': posterior.samples['stellar_mass'].std(),
                'age': posterior.samples['age'].mean(),
                'age_err': posterior.samples['age'].std(),
                'tau': posterior.samples['tau'].mean(),
                'tau_err': posterior.samples['tau'].std(),
                'metallicity': posterior.samples['metallicity'].mean(),
                'metallicity_err': posterior.samples['metallicity'].std(),
                'av': posterior.samples['dust:Av'].mean(),
                'av_err': posterior.samples['dust:Av'].std(),
                'redshift': posterior.samples['redshift'].mean(),
                'redshift_err': posterior.samples['redshift'].std(),
                'posterior_samples': posterior.samples,
                'model_spectrum': fit.galaxy.spectrum_full,
                'chi2': posterior.samples['chisq_phot'].mean() if 'chisq_phot' in posterior.samples else None
            }
            
            if 'nebular' in fit_instructions:
                fit_results['log_u'] = posterior.samples['nebular:logU'].mean()
                fit_results['log_u_err'] = posterior.samples['nebular:logU'].std()
            
            # Store results
            self.spectral_fits[source_id] = fit_results
            
            return fit_results
            
        except Exception as e:
            print(f"Error in spectral fitting: {e}")
            return None
    
    def _mock_spectral_fitting(self, spectrum, source_id):
        """Generate mock spectral fitting results."""
        # Extract basic properties from spectrum
        wavelength = spectrum['wavelength']
        flux = spectrum['flux']
        
        # Estimate redshift from emission lines (mock)
        redshift = np.random.uniform(1.0, 4.0)
        
        # Generate realistic parameters
        stellar_mass = np.random.uniform(9.5, 11.5)
        age = np.random.uniform(0.5, 5.0)
        tau = np.random.uniform(0.5, 3.0)
        metallicity = np.random.uniform(0.5, 1.5)
        av = np.random.exponential(0.3)
        
        # Generate mock model spectrum
        model_flux = flux * (1 + 0.1 * np.random.normal(0, 1, len(flux)))
        
        fit_results = {
            'stellar_mass': stellar_mass,
            'stellar_mass_err': 0.1,
            'age': age,
            'age_err': age * 0.2,
            'tau': tau,
            'tau_err': tau * 0.3,
            'metallicity': metallicity,
            'metallicity_err': 0.1,
            'av': av,
            'av_err': 0.05,
            'redshift': redshift,
            'redshift_err': 0.01,
            'model_spectrum': {
                'wavelength': wavelength,
                'flux': model_flux
            },
            'chi2': np.random.uniform(0.8, 1.5),
            'mock': True
        }
        
        self.spectral_fits[source_id] = fit_results
        return fit_results
    
    def plot_2d_spectrum(self, data_stage='stage2', save_path=None):
        """Plot 2D spectrum."""
        if data_stage not in self.reduced_data:
            raise ValueError(f"No data available for {data_stage}")
        
        data_2d = self.reduced_data[data_stage]['data']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create wavelength axis
        nx = data_2d.shape[1]
        wavelength = np.linspace(1.0, 5.0, nx)
        
        im = ax.imshow(data_2d, aspect='auto', origin='lower',
                      extent=[wavelength[0], wavelength[-1], 0, data_2d.shape[0]],
                      cmap='viridis', interpolation='bilinear')
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Spatial Direction (pixels)')
        ax.set_title(f'2D Spectrum ({data_stage.title()})')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Flux')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_1d_spectrum(self, source_id='target', show_fit=True, save_path=None):
        """Plot extracted 1D spectrum."""
        if source_id not in self.extracted_spectra:
            raise ValueError(f"No extracted spectrum for {source_id}")
        
        spectrum = self.extracted_spectra[source_id]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main spectrum plot
        wavelength = spectrum['wavelength']
        flux = spectrum['flux']
        flux_error = spectrum['flux_error']
        
        axes[0].plot(wavelength, flux, 'k-', linewidth=1, label='Observed')
        axes[0].fill_between(wavelength, flux - flux_error, flux + flux_error,
                           alpha=0.3, color='gray', label='±1σ error')
        
        # Show model fit if available
        if show_fit and source_id in self.spectral_fits:
            fit_results = self.spectral_fits[source_id]
            if 'model_spectrum' in fit_results:
                model = fit_results['model_spectrum']
                axes[0].plot(model['wavelength'], model['flux'], 
                           'r-', linewidth=1, label='Best fit model')
        
        axes[0].set_ylabel('Flux')
        axes[0].set_title(f'1D Spectrum: {source_id}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        if show_fit and source_id in self.spectral_fits:
            fit_results = self.spectral_fits[source_id]
            if 'model_spectrum' in fit_results:
                model_flux = np.interp(wavelength, 
                                     fit_results['model_spectrum']['wavelength'],
                                     fit_results['model_spectrum']['flux'])
                residuals = (flux - model_flux) / flux_error
                
                axes[1].plot(wavelength, residuals, 'g-', linewidth=1)
                axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[1].set_ylabel('Residuals (σ)')
                axes[1].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Wavelength (μm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def plot_spectral_fit_results(self, source_id='target', save_path=None):
        """Plot spectral fitting results."""
        if source_id not in self.spectral_fits:
            raise ValueError(f"No fit results for {source_id}")
        
        fit_results = self.spectral_fits[source_id]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Fit parameters table
        params = [
            f"log(M*/M☉) = {fit_results['stellar_mass']:.2f} ± {fit_results['stellar_mass_err']:.2f}",
            f"Age = {fit_results['age']:.2f} ± {fit_results['age_err']:.2f} Gyr",
            f"τ = {fit_results['tau']:.2f} ± {fit_results['tau_err']:.2f} Gyr", 
            f"Z/Z☉ = {fit_results['metallicity']:.2f} ± {fit_results['metallicity_err']:.2f}",
            f"Av = {fit_results['av']:.2f} ± {fit_results['av_err']:.2f} mag",
            f"z = {fit_results['redshift']:.3f} ± {fit_results['redshift_err']:.3f}"
        ]
        
        axes[0,0].text(0.05, 0.95, '\n'.join(params), transform=axes[0,0].transAxes,
                      verticalalignment='top', fontsize=12, fontfamily='monospace')
        axes[0,0].set_xlim(0, 1)
        axes[0,0].set_ylim(0, 1)
        axes[0,0].axis('off')
        axes[0,0].set_title('Fitted Parameters')
        
        # Star formation history
        if not fit_results.get('mock', False):
            # Use actual SFH from fit
            time = np.linspace(0, 13.8, 100)
            sfh = np.exp(-time / fit_results['tau']) * (time > (13.8 - fit_results['age']))
        else:
            # Mock SFH
            time = np.linspace(0, 13.8, 100)
            lookback_time = 13.8 - fit_results['age']
            sfh = np.exp(-(time - lookback_time)**2 / (2 * fit_results['tau']**2))
            sfh[time < lookback_time] = 0
        
        axes[0,1].plot(time, sfh, 'b-', linewidth=2)
        axes[0,1].set_xlabel('Cosmic Time (Gyr)')
        axes[0,1].set_ylabel('SFR (arbitrary units)')
        axes[0,1].set_title('Star Formation History')
        axes[0,1].grid(True, alpha=0.3)
        
        # Spectrum comparison (already plotted in main spectrum plot)
        spectrum = self.extracted_spectra[source_id]
        wavelength = spectrum['wavelength']
        flux = spectrum['flux']
        
        axes[1,0].plot(wavelength, flux, 'k-', linewidth=1, label='Observed')
        if 'model_spectrum' in fit_results:
            model = fit_results['model_spectrum']
            axes[1,0].plot(model['wavelength'], model['flux'], 
                         'r-', linewidth=1, label='Model')
        
        axes[1,0].set_xlabel('Wavelength (μm)')
        axes[1,0].set_ylabel('Flux')
        axes[1,0].set_title('Spectral Fit')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Chi-squared or goodness of fit
        if fit_results.get('chi2') is not None:
            axes[1,1].text(0.5, 0.5, f"χ² = {fit_results['chi2']:.2f}", 
                          transform=axes[1,1].transAxes, ha='center', va='center',
                          fontsize=16)
        else:
            axes[1,1].text(0.5, 0.5, "No χ² available", 
                          transform=axes[1,1].transAxes, ha='center', va='center',
                          fontsize=16)
        
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Goodness of Fit')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def generate_pipeline_summary(self):
        """Generate summary of pipeline processing."""
        summary = {
            'pipeline_status': self.pipeline_status.copy(),
            'data_products': list(self.reduced_data.keys()),
            'extracted_sources': list(self.extracted_spectra.keys()),
            'fitted_sources': list(self.spectral_fits.keys())
        }
        
        # Add fit results summary
        if self.spectral_fits:
            fit_summary = {}
            for source_id, results in self.spectral_fits.items():
                fit_summary[source_id] = {
                    'stellar_mass': results['stellar_mass'],
                    'age': results['age'],
                    'redshift': results['redshift'],
                    'av': results['av']
                }
            summary['fit_results_summary'] = fit_summary
        
        return summary
    
    def export_results(self, output_path, format='fits'):
        """
        Export analysis results.
        
        Parameters:
        -----------
        output_path : str
            Output file path
        format : str
            Output format ('fits', 'csv', 'json')
        """
        if format.lower() == 'fits' and HAVE_ASTROPY:
            # Create FITS file with multiple extensions
            hdu_list = [fits.PrimaryHDU()]
            
            # Add extracted spectra
            for source_id, spectrum in self.extracted_spectra.items():
                table = Table([
                    spectrum['wavelength'],
                    spectrum['flux'], 
                    spectrum['flux_error']
                ], names=['WAVELENGTH', 'FLUX', 'FLUX_ERROR'])
                
                hdu = fits.BinTableHDU(table, name=f'SPECTRUM_{source_id.upper()}')
                hdu_list.append(hdu)
            
            # Add fit results
            if self.spectral_fits:
                fit_data = []
                for source_id, results in self.spectral_fits.items():
                    fit_data.append([
                        source_id,
                        results['stellar_mass'],
                        results['stellar_mass_err'],
                        results['age'],
                        results['age_err'],
                        results['redshift'],
                        results['redshift_err'],
                        results['av'],
                        results['av_err']
                    ])
                
                fit_table = Table(rows=fit_data, 
                                names=['SOURCE_ID', 'STELLAR_MASS', 'STELLAR_MASS_ERR',
                                      'AGE', 'AGE_ERR', 'REDSHIFT', 'REDSHIFT_ERR',
                                      'AV', 'AV_ERR'])
                
                hdu_list.append(fits.BinTableHDU(fit_table, name='FIT_RESULTS'))
            
            # Write FITS file
            fits.HDUList(hdu_list).writeto(output_path, overwrite=True)
            
        elif format.lower() == 'csv':
            # Export fit results as CSV
            if self.spectral_fits:
                fit_data = []
                for source_id, results in self.spectral_fits.items():
                    row = {'source_id': source_id}
                    row.update(results)
                    # Remove complex objects
                    for key in ['posterior_samples', 'model_spectrum']:
                        row.pop(key, None)
                    fit_data.append(row)
                
                df = pd.DataFrame(fit_data)
                df.to_csv(output_path, index=False)
        
        elif format.lower() == 'json':
            # Export as JSON
            export_data = {
                'pipeline_summary': self.generate_pipeline_summary(),
                'fit_results': {}
            }
            
            # Add fit results (excluding complex objects)
            for source_id, results in self.spectral_fits.items():
                clean_results = results.copy()
                for key in ['posterior_samples', 'model_spectrum']:
                    clean_results.pop(key, None)
                export_data['fit_results'][source_id] = clean_results
            
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {output_path}")
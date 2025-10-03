# Cosmic Evolution Module for Astro-AI
# 
# This module integrates py21cmfast functionality for simulating the early universe
# and generating brightness temperature maps, power spectra, and cosmic evolution timelines.
# 
# Based on functions from CosmoSim.ipynb and utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from numpy import pi
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import py21cmfast as p21c  # type: ignore
    from py21cmfast import plotting  # type: ignore
    from py21cmfast import cache_tools  # type: ignore
    HAVE_21CMFAST = True
except ImportError:
    HAVE_21CMFAST = False
    logger.warning("py21cmfast not available. Falling back to simulation mode.")

try:
    import tools21cm as t2c  # type: ignore
    HAVE_TOOLS21CM = True
except ImportError:
    HAVE_TOOLS21CM = False

class CosmicEvolution:
    """
    Cosmic Evolution Analysis using 21cmFAST simulations.
    
    This class provides methods for running 21cm simulations, generating
    brightness temperature maps, computing power spectra, and analyzing
    the evolution of the cosmic 21cm signal.
    """
    
    def __init__(self, cosmology_params):
        """
        Initialize the cosmic evolution module.
        
        Parameters:
        -----------
        cosmology_params : dict
            Dictionary containing cosmological parameters:
            - H0: Hubble constant (km/s/Mpc)
            - Om0: Matter density parameter
            - sigma8: Amplitude of matter fluctuations
            - z_range: [z_min, z_max] redshift range
        """
        self.cosmo_params = cosmology_params
        self.results = {}
        
        # Set up default simulation parameters
        self.default_params = {
            'HII_DIM': 50,
            'BOX_LEN': 50,  # Mpc
            'SIGMA_8': cosmology_params.get('sigma8', 0.8),
            'hlittle': cosmology_params.get('H0', 67.66) / 100.0,
            'OMm': cosmology_params.get('Om0', 0.31)
        }
        
    def brightness_temperature(self, box_size=50, redshift=10.0, hubble=None, matter=None, random_seed=54321):
        """
        Compute brightness temperature field at given redshift.
        
        Parameters:
        -----------
        box_size : int
            Size of simulation box in Mpc and grid points
        redshift : float
            Redshift for computation
        hubble : float, optional
            Hubble parameter override
        matter : float, optional
            Matter density parameter override
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        brightness_temp : object
            21cmFAST brightness temperature object
        """
        if not HAVE_21CMFAST:
            # Return mock data for testing
            return self._mock_brightness_temperature(box_size, redshift)
        
        # Set cosmological parameters
        cosmo_params = p21c.CosmoParams(
            SIGMA_8=self.default_params['SIGMA_8'],
            hlittle=hubble if hubble is not None else self.default_params['hlittle'],
            OMm=matter if matter is not None else self.default_params['OMm']
        )
        
        # Generate initial conditions
        initial_conditions = p21c.initial_conditions(
            user_params={"HII_DIM": box_size, "BOX_LEN": box_size},
            cosmo_params=cosmo_params,
            random_seed=random_seed
        )
        
        # Generate perturbed field
        perturbed_field = p21c.perturb_field(
            redshift=redshift,
            init_boxes=initial_conditions
        )
        
        # Generate ionized box
        ionized_box = p21c.ionize_box(
            perturbed_field=perturbed_field
        )
        
        # Compute brightness temperature
        brightness_temp = p21c.brightness_temperature(
            ionized_box=ionized_box, 
            perturbed_field=perturbed_field
        )
        
        return brightness_temp
    
    def _mock_brightness_temperature(self, box_size, redshift):
        """Generate mock brightness temperature data for testing."""
        class MockBrightnessTemp:
            def __init__(self, box_size, redshift):
                # Generate realistic-looking 21cm signal
                x = np.linspace(0, box_size, box_size)
                y = np.linspace(0, box_size, box_size)
                z = np.linspace(0, box_size, box_size)
                
                # Create 3D coordinates
                X, Y, Z = np.meshgrid(x, y, z)
                
                # Generate correlated noise pattern
                np.random.seed(int(redshift * 1000))
                noise = np.random.normal(0, 1, (box_size, box_size, box_size))
                
                # Apply smoothing to create structure
                from scipy.ndimage import gaussian_filter
                smoothed = gaussian_filter(noise, sigma=2.0)
                
                # Scale to realistic brightness temperature values
                self.brightness_temp = -50 * np.exp(-(redshift - 10)**2 / 20) * (1 + 0.3 * smoothed)
                
        return MockBrightnessTemp(box_size, redshift)
    
    def compute_power_spectrum_1d(self, cube, kbins=15, box_length=50):
        """
        Compute 1D power spectrum from 3D brightness temperature cube.
        
        Parameters:
        -----------
        cube : ndarray
            3D brightness temperature array
        kbins : int
            Number of k bins
        box_length : float
            Physical size of box in Mpc
            
        Returns:
        --------
        k : ndarray
            k values in h/Mpc
        ps : ndarray
            Power spectrum values in mK^2 Mpc^3
        """
        H = 67.66  
        h = H / 100.0
        dim = box_length
        
        box_dims = [dim] * len(cube.shape)
        
        # Fourier transform
        ft = fftpack.fftshift(fftpack.fftn(cube.astype('float64')))
        
        # Power spectrum
        power_spectrum = np.abs(ft)**2
        
        # Normalization
        boxvol = np.prod(box_dims)
        pixelsize = boxvol / (np.prod(cube.shape))
        
        power_spectrum *= pixelsize**2 / boxvol
        
        # Radial averaging
        ps, ks, n_modes = self._radial_average(
            power_spectrum, kbins=kbins, box_dims=box_dims
        )
        
        return ks, ps * ks**3 / (2 * pi**2)
    
    def _radial_average(self, input_array, box_dims, kbins=10):
        """Compute radially averaged power spectrum."""
        k_comp, k = self._get_k(input_array, box_dims)
        
        kbins = self._get_kbins(kbins, box_dims, k)
        dk = (kbins[1:] - kbins[:-1]) / 2.
        
        outdata = np.histogram(k.flatten(), bins=kbins, weights=input_array.flatten())[0]
        n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
        
        # Avoid division by zero
        n_modes[n_modes == 0] = 1
        outdata /= n_modes
        
        return outdata, kbins[:-1] + dk, n_modes
    
    def _get_k(self, input_array, box_dims):
        """Get k values for input array."""
        dim = len(input_array.shape)
        
        if dim == 3:
            nx, ny, nz = input_array.shape
            x, y, z = np.indices(input_array.shape, dtype='int32')
            center = np.array([nx/2, ny/2, nz/2])
            
            kx = 2. * np.pi * (x - center[0]) / box_dims[0]
            ky = 2. * np.pi * (y - center[1]) / box_dims[1]
            kz = 2. * np.pi * (z - center[2]) / box_dims[2]
            
            k = np.sqrt(kx**2 + ky**2 + kz**2)
            return [kx, ky, kz], k
        else:
            raise ValueError("Only 3D arrays supported")
    
    def _get_kbins(self, kbins, box_dims, k):
        """Generate k bins."""
        if isinstance(kbins, int):
            kmin = 2. * np.pi / min(box_dims)
            kbins = 10**np.linspace(np.log10(kmin), np.log10(k.max()), kbins + 1)
        return kbins
    
    def run_simulation(self, box_size=50, resolution=50, z_start=15.0, z_end=6.0, z_step=1.0):
        """
        Run complete cosmic evolution simulation across redshift range.
        
        Parameters:
        -----------
        box_size : float
            Physical size of simulation box in Mpc
        resolution : int
            Grid resolution (HII_DIM)
        z_start : float
            Starting redshift (highest z)
        z_end : float
            Ending redshift (lowest z)
        z_step : float
            Redshift step size
            
        Returns:
        --------
        results : dict
            Dictionary containing simulation results
        """
        # Generate redshift array
        z_array = np.arange(z_end, z_start + z_step, z_step)
        z_array = z_array[::-1]  # Reverse to go from high to low z
        
        print(f"Running simulation for redshifts: {z_array}")
        
        # Initialize result arrays
        brightness_temps = []
        power_spectra_k = []
        power_spectra_ps = []
        global_signals = []
        
        # Run simulation for each redshift
        for i, z in enumerate(z_array):
            print(f"Computing redshift z = {z:.1f} ({i+1}/{len(z_array)})")
            
            # Compute brightness temperature
            bt_result = self.brightness_temperature(
                box_size=resolution, 
                redshift=z
            )
            
            brightness_temps.append(bt_result.brightness_temp)
            
            # Compute power spectrum
            k, ps = self.compute_power_spectrum_1d(
                bt_result.brightness_temp, 
                box_length=box_size
            )
            power_spectra_k.append(k)
            power_spectra_ps.append(ps)
            
            # Compute global signal (spatial average)
            global_signal = np.mean(bt_result.brightness_temp)
            global_signals.append(global_signal)
        
        # Store results
        self.results = {
            'redshifts': z_array,
            'brightness_temperatures': brightness_temps,
            'power_spectra_k': power_spectra_k,
            'power_spectra_ps': power_spectra_ps,
            'global_signals': global_signals,
            'simulation_params': {
                'box_size': box_size,
                'resolution': resolution,
                'cosmology': self.cosmo_params
            }
        }
        
        return self.results
    
    def plot_global_evolution(self, save_path=None):
        """
        Plot the evolution of the global 21cm signal.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        fig, ax : matplotlib objects
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.results['redshifts'], self.results['global_signals'], 
                'b-', linewidth=2, marker='o', markersize=6)
        
        ax.set_xlabel('Redshift z', fontsize=14)
        ax.set_ylabel('Global 21cm Signal [mK]', fontsize=14)
        ax.set_title('Evolution of Global 21cm Signal', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key epochs
        ax.axvline(10, color='red', linestyle='--', alpha=0.7, label='Peak signal')
        ax.axvline(6, color='green', linestyle='--', alpha=0.7, label='End of reionization')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_power_spectra_evolution(self, save_path=None):
        """
        Plot evolution of 21cm power spectra.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        fig, ax : matplotlib objects
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results['redshifts'])))
        
        for i, z in enumerate(self.results['redshifts']):
            k = self.results['power_spectra_k'][i]
            ps = self.results['power_spectra_ps'][i]
            
            ax.loglog(k, ps, color=colors[i], linewidth=2, 
                     label=f'z = {z:.1f}')
        
        ax.set_xlabel('k [Mpc⁻¹]', fontsize=14)
        ax.set_ylabel('k³/(2π²) P(k) [mK² Mpc³]', fontsize=14)
        ax.set_title('Evolution of 21cm Power Spectrum', fontsize=16)
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)
        ax.legend(ncol=2, fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_brightness_temperature_slices(self, z_indices=None, save_path=None):
        """
        Plot 2D slices of brightness temperature at different redshifts.
        
        Parameters:
        -----------
        z_indices : list, optional
            Indices of redshifts to plot. If None, plots first, middle, and last.
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        if z_indices is None:
            n_z = len(self.results['redshifts'])
            z_indices = [0, n_z//2, n_z-1]
        
        n_plots = len(z_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        for i, z_idx in enumerate(z_indices):
            bt_cube = self.results['brightness_temperatures'][z_idx]
            z_val = self.results['redshifts'][z_idx]
            
            # Take middle slice
            slice_2d = bt_cube[:, :, bt_cube.shape[2]//2]
            
            im = axes[i].imshow(slice_2d, origin='lower', cmap='viridis', 
                               interpolation='bilinear')
            axes[i].set_title(f'z = {z_val:.1f}', fontsize=14)
            axes[i].set_xlabel('x [pixels]')
            if i == 0:
                axes[i].set_ylabel('y [pixels]')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Brightness Temperature [mK]', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def generate_summary_report(self):
        """
        Generate a summary report of the simulation results.
        
        Returns:
        --------
        report : dict
            Dictionary containing summary statistics and key findings
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        # Extract key statistics
        global_signals = np.array(self.results['global_signals'])
        redshifts = np.array(self.results['redshifts'])
        
        # Find extrema
        min_signal_idx = np.argmin(global_signals)
        max_signal_idx = np.argmax(global_signals)
        
        report = {
            'simulation_summary': {
                'redshift_range': [float(redshifts.min()), float(redshifts.max())],
                'n_redshift_steps': len(redshifts),
                'box_size_mpc': self.results['simulation_params']['box_size'],
                'resolution': self.results['simulation_params']['resolution']
            },
            'signal_characteristics': {
                'min_signal_mK': float(global_signals[min_signal_idx]),
                'min_signal_redshift': float(redshifts[min_signal_idx]),
                'max_signal_mK': float(global_signals[max_signal_idx]),
                'max_signal_redshift': float(redshifts[max_signal_idx]),
                'signal_range_mK': float(global_signals.max() - global_signals.min())
            },
            'physical_interpretation': {
                'reionization_signature': global_signals[min_signal_idx] < -20,
                'cosmic_dawn_detected': global_signals[max_signal_idx] > -100,
                'evolution_trend': 'heating' if global_signals[-1] > global_signals[0] else 'cooling'
            }
        }
        
        return report
    
    def export_results(self, output_path):
        """
        Export simulation results to file.
        
        Parameters:
        -----------
        output_path : str
            Path for output file (supports .npz, .h5)
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        if output_path.endswith('.npz'):
            np.savez(output_path, **self.results)
        elif output_path.endswith('.h5'):
            try:
                import h5py
                with h5py.File(output_path, 'w') as f:
                    for key, value in self.results.items():
                        if isinstance(value, (list, np.ndarray)):
                            f.create_dataset(key, data=value)
                        elif isinstance(value, dict):
                            grp = f.create_group(key)
                            for subkey, subvalue in value.items():
                                grp.create_dataset(subkey, data=subvalue)
            except ImportError:
                raise ImportError("h5py required for HDF5 export")
        else:
            raise ValueError("Unsupported file format. Use .npz or .h5")
        
        print(f"Results exported to {output_path}")
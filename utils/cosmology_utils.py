"""
Cosmology Utilities Module
Provides cosmological calculations and conversions
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy import units as u
from astropy.coordinates import SkyCoord
from typing import Union, Tuple


class CosmologyUtils:
    """
    Utility class for cosmological calculations
    Uses Astropy's cosmology module with Planck18 parameters as default
    """
    
    def __init__(self, H0: float = 67.66, Om0: float = 0.3111, Ob0: float = 0.0490):
        """
        Initialize cosmology
        
        Parameters:
        -----------
        H0 : float
            Hubble constant in km/s/Mpc (default: Planck18)
        Om0 : float
            Matter density parameter (default: Planck18)
        Ob0 : float
            Baryon density parameter (default: Planck18)
        """
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
        self.planck18 = Planck18
        
    def redshift_to_age(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert redshift to age of universe
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Age in Gyr
        """
        age = self.cosmo.age(z).value
        return age
    
    def age_to_redshift(self, age_gyr: float, z_range: Tuple[float, float] = (0, 20)) -> float:
        """
        Convert age of universe to redshift (inverse operation)
        
        Parameters:
        -----------
        age_gyr : float
            Age in Gyr
        z_range : tuple
            Range of redshifts to search
            
        Returns:
        --------
        float : Corresponding redshift
        """
        from scipy.optimize import brentq
        
        def age_diff(z):
            return self.cosmo.age(z).value - age_gyr
        
        try:
            z = brentq(age_diff, z_range[0], z_range[1])
            return z
        except ValueError:
            return np.nan
    
    def luminosity_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate luminosity distance
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Luminosity distance in Mpc
        """
        d_L = self.cosmo.luminosity_distance(z).value
        return d_L
    
    def angular_diameter_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate angular diameter distance
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Angular diameter distance in Mpc
        """
        d_A = self.cosmo.angular_diameter_distance(z).value
        return d_A
    
    def comoving_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate comoving distance
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Comoving distance in Mpc
        """
        d_C = self.cosmo.comoving_distance(z).value
        return d_C
    
    def comoving_volume(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate comoving volume
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Comoving volume in Gpc³
        """
        vol = self.cosmo.comoving_volume(z).value / 1e9  # Convert to Gpc³
        return vol
    
    def lookback_time(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate lookback time
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Lookback time in Gyr
        """
        t_lookback = self.cosmo.lookback_time(z).value
        return t_lookback
    
    def critical_density(self, z: Union[float, np.ndarray] = 0) -> Union[float, np.ndarray]:
        """
        Calculate critical density
        
        Parameters:
        -----------
        z : float or array
            Redshift (default: 0)
            
        Returns:
        --------
        float or array : Critical density in g/cm³
        """
        rho_crit = self.cosmo.critical_density(z).to(u.g / u.cm**3).value
        return rho_crit
    
    def angular_scale(self, z: float) -> float:
        """
        Calculate angular scale (kpc per arcsec)
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        float : Angular scale in kpc/arcsec
        """
        d_A = self.angular_diameter_distance(z)  # in Mpc
        scale = d_A * 1000 * (np.pi / 648000)  # Convert to kpc/arcsec
        return scale
    
    def absolute_to_apparent_magnitude(self, M: float, z: float) -> float:
        """
        Convert absolute magnitude to apparent magnitude
        
        Parameters:
        -----------
        M : float
            Absolute magnitude
        z : float
            Redshift
            
        Returns:
        --------
        float : Apparent magnitude
        """
        d_L = self.luminosity_distance(z)  # in Mpc
        m = M + 5 * np.log10(d_L * 1e6 / 10)
        return m
    
    def apparent_to_absolute_magnitude(self, m: float, z: float) -> float:
        """
        Convert apparent magnitude to absolute magnitude
        
        Parameters:
        -----------
        m : float
            Apparent magnitude
        z : float
            Redshift
            
        Returns:
        --------
        float : Absolute magnitude
        """
        d_L = self.luminosity_distance(z)  # in Mpc
        M = m - 5 * np.log10(d_L * 1e6 / 10)
        return M
    
    def flux_to_luminosity(self, flux: float, z: float) -> float:
        """
        Convert flux to luminosity
        
        Parameters:
        -----------
        flux : float
            Flux in erg/s/cm²
        z : float
            Redshift
            
        Returns:
        --------
        float : Luminosity in erg/s
        """
        d_L = self.luminosity_distance(z) * 3.086e24  # Convert Mpc to cm
        L = flux * 4 * np.pi * d_L**2
        return L
    
    def luminosity_to_flux(self, luminosity: float, z: float) -> float:
        """
        Convert luminosity to flux
        
        Parameters:
        -----------
        luminosity : float
            Luminosity in erg/s
        z : float
            Redshift
            
        Returns:
        --------
        float : Flux in erg/s/cm²
        """
        d_L = self.luminosity_distance(z) * 3.086e24  # Convert Mpc to cm
        flux = luminosity / (4 * np.pi * d_L**2)
        return flux
    
    def redshift_velocity(self, z: float) -> float:
        """
        Convert redshift to velocity (non-relativistic approximation)
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        float : Velocity in km/s
        """
        c = 299792.458  # Speed of light in km/s
        v = z * c
        return v
    
    def velocity_to_redshift(self, v: float) -> float:
        """
        Convert velocity to redshift (non-relativistic approximation)
        
        Parameters:
        -----------
        v : float
            Velocity in km/s
            
        Returns:
        --------
        float : Redshift
        """
        c = 299792.458  # Speed of light in km/s
        z = v / c
        return z
    
    def scale_factor(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert redshift to scale factor
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : Scale factor
        """
        a = 1.0 / (1.0 + z)
        return a
    
    def redshift_from_scale_factor(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert scale factor to redshift
        
        Parameters:
        -----------
        a : float or array
            Scale factor
            
        Returns:
        --------
        float or array : Redshift
        """
        z = (1.0 / a) - 1.0
        return z
    
    def hubble_parameter(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Hubble parameter at redshift z
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array : H(z) in km/s/Mpc
        """
        H_z = self.cosmo.H(z).value
        return H_z
    
    def get_cosmology_params(self) -> dict:
        """
        Get current cosmology parameters
        
        Returns:
        --------
        dict : Dictionary of cosmology parameters
        """
        params = {
            'H0': self.cosmo.H0.value,
            'Om0': self.cosmo.Om0,
            'Ob0': self.cosmo.Ob0,
            'Ode0': self.cosmo.Ode0,
            'Tcmb0': self.cosmo.Tcmb0.value,
            'h': self.cosmo.h
        }
        return params
    
    @staticmethod
    def planck18_params() -> dict:
        """
        Get Planck 2018 cosmology parameters
        
        Returns:
        --------
        dict : Dictionary of Planck18 parameters
        """
        params = {
            'H0': Planck18.H0.value,
            'Om0': Planck18.Om0,
            'Ob0': Planck18.Ob0,
            'Ode0': Planck18.Ode0,
            'Tcmb0': Planck18.Tcmb0.value,
            'h': Planck18.h
        }
        return params

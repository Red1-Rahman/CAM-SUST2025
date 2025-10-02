"""
Plotting Utilities Module
Provides consistent plotting functions and styling for astronomical data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, List, Union
import seaborn as sns


class PlottingUtils:
    """
    Utility class for creating consistent and publication-quality plots
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize plotting utilities with a style
        
        Parameters:
        -----------
        style : str
            Matplotlib style ('default', 'seaborn', 'dark')
        """
        self.style = style
        self.setup_style()
        
        # Define color palettes
        self.color_palettes = {
            'galaxy': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'redshift': plt.cm.viridis,
            'temperature': plt.cm.plasma,
            'mass': plt.cm.cividis
        }
        
    def setup_style(self):
        """Setup matplotlib style"""
        if self.style == 'seaborn':
            sns.set_style("whitegrid")
            sns.set_context("talk")
        elif self.style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12
            
    def create_subplot_grid(self, nrows: int, ncols: int, 
                           figsize: Optional[Tuple[float, float]] = None,
                           **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a grid of subplots
        
        Parameters:
        -----------
        nrows : int
            Number of rows
        ncols : int
            Number of columns
        figsize : tuple, optional
            Figure size (width, height)
        **kwargs : Additional arguments for plt.subplots
            
        Returns:
        --------
        fig, axes : Figure and axes array
        """
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
            
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes
    
    def plot_spectrum(self, wavelength: np.ndarray, flux: np.ndarray,
                     error: Optional[np.ndarray] = None,
                     ax: Optional[plt.Axes] = None,
                     xlabel: str = 'Wavelength (μm)',
                     ylabel: str = 'Flux (arbitrary units)',
                     title: str = 'Spectrum',
                     **kwargs) -> plt.Axes:
        """
        Plot a spectrum with optional error bars
        
        Parameters:
        -----------
        wavelength : array
            Wavelength array
        flux : array
            Flux array
        error : array, optional
            Error array
        ax : Axes, optional
            Matplotlib axes to plot on
        xlabel, ylabel, title : str
            Axis labels and title
        **kwargs : Additional plot arguments
            
        Returns:
        --------
        ax : Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if error is not None:
            ax.errorbar(wavelength, flux, yerr=error, fmt='-', **kwargs)
        else:
            ax.plot(wavelength, flux, **kwargs)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_sed(self, wavelength: np.ndarray, luminosity: np.ndarray,
                ax: Optional[plt.Axes] = None,
                loglog: bool = True,
                title: str = 'Spectral Energy Distribution',
                **kwargs) -> plt.Axes:
        """
        Plot Spectral Energy Distribution
        
        Parameters:
        -----------
        wavelength : array
            Wavelength array
        luminosity : array
            Luminosity array
        ax : Axes, optional
            Matplotlib axes
        loglog : bool
            Use log-log scale
        title : str
            Plot title
        **kwargs : Additional plot arguments
            
        Returns:
        --------
        ax : Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if loglog:
            ax.loglog(wavelength, luminosity, **kwargs)
        else:
            ax.plot(wavelength, luminosity, **kwargs)
            
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Luminosity (L☉/Hz)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_image(self, data: np.ndarray,
                  ax: Optional[plt.Axes] = None,
                  cmap: str = 'viridis',
                  scale: str = 'linear',
                  colorbar: bool = True,
                  title: str = '',
                  **kwargs) -> Tuple[plt.Axes, plt.cm.ScalarMappable]:
        """
        Plot 2D image with colorbar
        
        Parameters:
        -----------
        data : 2D array
            Image data
        ax : Axes, optional
            Matplotlib axes
        cmap : str
            Colormap name
        scale : str
            'linear', 'log', or 'sqrt'
        colorbar : bool
            Show colorbar
        title : str
            Plot title
        **kwargs : Additional imshow arguments
            
        Returns:
        --------
        ax, im : Axes and image object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        # Apply scaling
        if scale == 'log':
            data = np.log10(np.abs(data) + 1e-10)
        elif scale == 'sqrt':
            data = np.sqrt(np.abs(data))
            
        im = ax.imshow(data, cmap=cmap, origin='lower', **kwargs)
        
        if colorbar:
            plt.colorbar(im, ax=ax)
            
        ax.set_title(title)
        
        return ax, im
    
    def plot_redshift_evolution(self, redshifts: np.ndarray, values: np.ndarray,
                               ax: Optional[plt.Axes] = None,
                               ylabel: str = 'Value',
                               title: str = 'Redshift Evolution',
                               **kwargs) -> plt.Axes:
        """
        Plot evolution with redshift
        
        Parameters:
        -----------
        redshifts : array
            Redshift values
        values : array
            Values to plot
        ax : Axes, optional
            Matplotlib axes
        ylabel : str
            Y-axis label
        title : str
            Plot title
        **kwargs : Additional plot arguments
            
        Returns:
        --------
        ax : Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(redshifts, values, **kwargs)
        ax.set_xlabel('Redshift')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Invert x-axis for time increasing to the right
        ax.invert_xaxis()
        
        return ax
    
    def plot_histogram(self, data: np.ndarray,
                      ax: Optional[plt.Axes] = None,
                      bins: Union[int, str] = 'auto',
                      xlabel: str = 'Value',
                      ylabel: str = 'Frequency',
                      title: str = 'Histogram',
                      **kwargs) -> plt.Axes:
        """
        Plot histogram
        
        Parameters:
        -----------
        data : array
            Data to histogram
        ax : Axes, optional
            Matplotlib axes
        bins : int or str
            Number of bins or binning strategy
        xlabel, ylabel, title : str
            Axis labels and title
        **kwargs : Additional hist arguments
            
        Returns:
        --------
        ax : Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.hist(data, bins=bins, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_corner(self, samples: np.ndarray, labels: List[str],
                   title: str = 'Corner Plot') -> plt.Figure:
        """
        Create a corner plot for parameter distributions
        
        Parameters:
        -----------
        samples : 2D array
            Parameter samples (n_samples, n_params)
        labels : list
            Parameter labels
        title : str
            Plot title
            
        Returns:
        --------
        fig : Matplotlib figure
        """
        try:
            import corner
            fig = corner.corner(samples, labels=labels, 
                              quantiles=[0.16, 0.5, 0.84],
                              show_titles=True)
            fig.suptitle(title, y=1.02)
            return fig
        except ImportError:
            print("Corner package not installed. Install with: pip install corner")
            return None
    
    def plot_comparison(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                       ax: Optional[plt.Axes] = None,
                       label1: str = 'Data 1',
                       label2: str = 'Data 2',
                       xlabel: str = 'X',
                       ylabel: str = 'Y',
                       title: str = 'Comparison',
                       **kwargs) -> plt.Axes:
        """
        Plot two datasets for comparison
        
        Parameters:
        -----------
        x : array
            X values
        y1, y2 : array
            Y values for comparison
        ax : Axes, optional
            Matplotlib axes
        label1, label2 : str
            Data labels
        xlabel, ylabel, title : str
            Axis labels and title
        **kwargs : Additional plot arguments
            
        Returns:
        --------
        ax : Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(x, y1, label=label1, **kwargs)
        ax.plot(x, y2, label=label2, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def add_redshift_axis(self, ax: plt.Axes, z_values: List[float] = None):
        """
        Add secondary axis showing redshift
        
        Parameters:
        -----------
        ax : Axes
            Matplotlib axes
        z_values : list, optional
            Specific redshift values to mark
        """
        if z_values is None:
            z_values = [0, 1, 2, 5, 10]
            
        ax2 = ax.twiny()
        ax2.set_xlabel('Redshift')
        ax2.set_xlim(ax.get_xlim())
        
        return ax2
    
    @staticmethod
    def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, **kwargs):
        """
        Save figure with publication quality
        
        Parameters:
        -----------
        fig : Figure
            Matplotlib figure
        filename : str
            Output filename
        dpi : int
            Resolution in dots per inch
        **kwargs : Additional savefig arguments
        """
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"Figure saved to {filename}")
    
    @staticmethod
    def close_all():
        """Close all open figures"""
        plt.close('all')
    
    def get_colormap(self, name: str = 'viridis', n_colors: int = 10) -> List:
        """
        Get discrete colors from colormap
        
        Parameters:
        -----------
        name : str
            Colormap name
        n_colors : int
            Number of discrete colors
            
        Returns:
        --------
        list : List of colors
        """
        cmap = plt.cm.get_cmap(name)
        colors = [cmap(i / n_colors) for i in range(n_colors)]
        return colors
    
    def annotate_peak(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                     label: str = 'Peak'):
        """
        Annotate peak value in plot
        
        Parameters:
        -----------
        ax : Axes
            Matplotlib axes
        x, y : array
            Data arrays
        label : str
            Annotation label
        """
        peak_idx = np.argmax(y)
        peak_x, peak_y = x[peak_idx], y[peak_idx]
        
        ax.annotate(f'{label}\n({peak_x:.2f}, {peak_y:.2e})',
                   xy=(peak_x, peak_y),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

"""
Data Handler Module
Handles data loading, saving, and management for astronomical data
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import pickle
import json
from typing import Dict, Any, Optional, Union
import warnings


class DataHandler:
    """
    Handles data I/O operations for various astronomical data formats
    including FITS, HDF5, CSV, and custom formats
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataHandler
        
        Parameters:
        -----------
        data_dir : str
            Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_hdf5(self, data: Dict[str, Any], filename: str, overwrite: bool = True):
        """
        Save data to HDF5 format
        
        Parameters:
        -----------
        data : dict
            Dictionary containing data arrays to save
        filename : str
            Output filename
        overwrite : bool
            Whether to overwrite existing file
        """
        filepath = self.data_dir / filename
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filepath} already exists")
            
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                if isinstance(value, (np.ndarray, list, tuple)):
                    f.create_dataset(key, data=value)
                elif isinstance(value, (int, float, str)):
                    f.attrs[key] = value
                    
        return filepath
    
    def load_hdf5(self, filename: str) -> Dict[str, Any]:
        """
        Load data from HDF5 format
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        dict : Dictionary containing loaded data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        data = {}
        with h5py.File(filepath, 'r') as f:
            # Load datasets
            for key in f.keys():
                data[key] = f[key][:]
            
            # Load attributes
            for key in f.attrs.keys():
                data[key] = f.attrs[key]
                
        return data
    
    def save_csv(self, data: Union[pd.DataFrame, Dict], filename: str):
        """
        Save data to CSV format
        
        Parameters:
        -----------
        data : pd.DataFrame or dict
            Data to save
        filename : str
            Output filename
        """
        filepath = self.data_dir / filename
        
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        data.to_csv(filepath, index=False)
        return filepath
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV format
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        return pd.read_csv(filepath)
    
    def save_pickle(self, data: Any, filename: str):
        """
        Save data using pickle
        
        Parameters:
        -----------
        data : Any
            Python object to save
        filename : str
            Output filename
        """
        filepath = self.data_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        return filepath
    
    def load_pickle(self, filename: str) -> Any:
        """
        Load data from pickle file
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        Any : Loaded Python object
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_json(self, data: Dict, filename: str):
        """
        Save data to JSON format
        
        Parameters:
        -----------
        data : dict
            Dictionary to save
        filename : str
            Output filename
        """
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath
    
    def load_json(self, filename: str) -> Dict:
        """
        Load data from JSON format
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        dict : Loaded data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_files(self, pattern: str = "*") -> list:
        """
        List files in data directory matching pattern
        
        Parameters:
        -----------
        pattern : str
            Glob pattern for file matching
            
        Returns:
        --------
        list : List of matching file paths
        """
        return list(self.data_dir.glob(pattern))
    
    def delete_file(self, filename: str):
        """
        Delete a file from data directory
        
        Parameters:
        -----------
        filename : str
            File to delete
        """
        filepath = self.data_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    @staticmethod
    def validate_data(data: np.ndarray, expected_shape: Optional[tuple] = None) -> bool:
        """
        Validate data array
        
        Parameters:
        -----------
        data : np.ndarray
            Data array to validate
        expected_shape : tuple, optional
            Expected shape of data
            
        Returns:
        --------
        bool : True if valid
        """
        if not isinstance(data, np.ndarray):
            return False
            
        if np.any(np.isnan(data)):
            warnings.warn("Data contains NaN values")
            
        if np.any(np.isinf(data)):
            warnings.warn("Data contains infinite values")
            
        if expected_shape is not None and data.shape != expected_shape:
            return False
            
        return True
    
    @staticmethod
    def clean_data(data: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """
        Clean data by replacing NaN and inf values
        
        Parameters:
        -----------
        data : np.ndarray
            Data array to clean
        fill_value : float
            Value to replace NaN/inf with
            
        Returns:
        --------
        np.ndarray : Cleaned data
        """
        cleaned = data.copy()
        cleaned[np.isnan(cleaned)] = fill_value
        cleaned[np.isinf(cleaned)] = fill_value
        return cleaned

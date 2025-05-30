# In feature_service/processor_utils.py

import logging
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cudf
    import cupy as cp
    HAS_GPU = True
    logger.info("GPU acceleration enabled using RAPIDS cuDF")
except ImportError:
    # Fall back to CPU libraries
    logger.warning("GPU libraries (cudf, cupy) not available. Using CPU versions (pandas, numpy) instead.")
    # Create compatibility layer
    HAS_GPU = False
    
    # Make pandas simulate cudf interface
    class CudfCompat:
        """
        Compatibility layer for mimicking essential cuDF functionality with pandas
        """
        
        @staticmethod
        def DataFrame(*args, **kwargs):
            """Return pandas DataFrame instead of cuDF DataFrame"""
            return pd.DataFrame(*args, **kwargs)
            
        @staticmethod
        def Series(*args, **kwargs):
            """Return pandas Series instead of cuDF Series"""
            return pd.Series(*args, **kwargs)

        # Add any other needed compatibility methods
        @staticmethod
        def merge(left, right, *args, **kwargs):
            """Mimic ``cudf.merge`` using pandas."""
            return pd.merge(left, right, *args, **kwargs)

        @staticmethod
        def join(left, right, *args, **kwargs):
            """Mimic ``cudf.join`` using pandas DataFrame.join."""
            return left.join(right, *args, **kwargs)

        @staticmethod
        def groupby(df, *args, **kwargs):
            """Mimic ``cudf.groupby`` returning a pandas ``GroupBy`` object."""
            return df.groupby(*args, **kwargs)

        def __getattr__(self, name):
            """Forward any other attributes to pandas"""
            return getattr(pd, name)
    
    # Replace actual cudf with our compatibility layer  
    cudf = CudfCompat()
    cp = np  # Use numpy instead of cupy

# Export the GPU status flag and library references
__all__ = ['HAS_GPU', 'cudf', 'cp']

"""
Data Manager
============

Handles dataset loading, processing, and management for the application.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Manages dataset operations and state."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.current_dataset: Optional[pd.DataFrame] = None
        self.dataset_name: Optional[str] = None
        self.selected_rows: List[int] = []
        self.selected_columns: List[str] = []
        self.dataset_history: List[Dict[str, Any]] = []
        self.supported_formats = ['.csv', '.xlsx', '.json', '.parquet']
    
    def load_dataset(self, file_path: str) -> bool:
        """
        Load a dataset from file.
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            self.dataset_name = file_path.stem
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            # Load based on file type
            if file_path.suffix.lower() == '.csv':
                self.current_dataset = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                self.current_dataset = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                self.current_dataset = pd.read_json(file_path)
            elif file_path.suffix.lower() == '.parquet':
                self.current_dataset = pd.read_parquet(file_path)
            
            # Reset selections
            self.selected_rows = []
            self.selected_columns = []
            
            logger.info(f"Successfully loaded dataset: {self.dataset_name}")
            logger.info(f"Dataset shape: {self.current_dataset.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the current dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.current_dataset is None:
            return {"error": "No dataset loaded"}
        
        return {
            "name": self.dataset_name,
            "shape": self.current_dataset.shape,
            "columns": self.current_dataset.columns.tolist(),
            "dtypes": self.current_dataset.dtypes.to_dict(),
            "null_counts": self.current_dataset.isnull().sum().to_dict(),
            "memory_usage": self.current_dataset.memory_usage(deep=True).sum(),
            "sample_data": self.current_dataset.head().to_dict('records')
        }
    
    def select_rows(self, row_indices: List[int]) -> bool:
        """
        Select specific rows in the dataset.
        
        Args:
            row_indices (List[int]): List of row indices to select
            
        Returns:
            bool: True if successful
        """
        if self.current_dataset is None:
            return False
        
        # Validate indices
        valid_indices = [i for i in row_indices if 0 <= i < len(self.current_dataset)]
        self.selected_rows = valid_indices
        
        logger.info(f"Selected {len(self.selected_rows)} rows")
        return True
    
    def select_columns(self, column_names: List[str]) -> bool:
        """
        Select specific columns in the dataset.
        
        Args:
            column_names (List[str]): List of column names to select
            
        Returns:
            bool: True if successful
        """
        if self.current_dataset is None:
            return False
        
        # Validate column names
        valid_columns = [col for col in column_names if col in self.current_dataset.columns]
        self.selected_columns = valid_columns
        
        logger.info(f"Selected {len(self.selected_columns)} columns")
        return True
    
    def get_selected_data(self) -> Optional[pd.DataFrame]:
        """
        Get the currently selected data subset.
        
        Returns:
            Optional[pd.DataFrame]: Selected data or None if no selection
        """
        if self.current_dataset is None:
            return None
        
        df = self.current_dataset
        
        # Apply row selection
        if self.selected_rows:
            df = df.iloc[self.selected_rows]
        
        # Apply column selection
        if self.selected_columns:
            df = df[self.selected_columns]
        
        return df
    
    def get_column_statistics(self, column_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific column.
        
        Args:
            column_name (str): Name of the column
            
        Returns:
            Dict[str, Any]: Column statistics
        """
        if self.current_dataset is None or column_name not in self.current_dataset.columns:
            return {"error": "Column not found"}
        
        column = self.current_dataset[column_name]
        
        stats = {
            "name": column_name,
            "dtype": str(column.dtype),
            "count": len(column),
            "null_count": column.isnull().sum(),
            "unique_count": column.nunique(),
        }
        
        # Add numeric statistics if applicable
        if column.dtype in ['int64', 'float64']:
            stats.update({
                "mean": column.mean(),
                "median": column.median(),
                "std": column.std(),
                "min": column.min(),
                "max": column.max(),
                "q25": column.quantile(0.25),
                "q75": column.quantile(0.75),
            })
        
        return stats
    
    def clear_selection(self):
        """Clear all current selections."""
        self.selected_rows = []
        self.selected_columns = []
        logger.info("Cleared all selections") 
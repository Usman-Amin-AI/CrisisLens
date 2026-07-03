"""
Data loading and splitting module for disaster tweets classification.
Handles CSV loading, train-validation split, and data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os


class DataLoader:
    """Loads and manages disaster tweets dataset."""
    
    def __init__(self, train_path: str, test_path: str, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
            random_state: Random seed for reproducibility
        """
        self.train_path = train_path
        self.test_path = test_path
        self.random_state = random_state
        self.df_train = None
        self.df_test = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.
        
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            FileNotFoundError: If CSV files not found
        """
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training data not found at {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found at {self.test_path}")
            
        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)
        
        return self.df_train, self.df_test
    
    def get_data_info(self) -> dict:
        """
        Get information about loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return {
            "train_size": len(self.df_train),
            "test_size": len(self.df_test),
            "train_missing": self.df_train.isna().sum().to_dict(),
            "test_missing": self.df_test.isna().sum().to_dict(),
            "target_distribution": self.df_train["target"].value_counts().to_dict(),
        }
    
    def prepare_splits(
        self, 
        test_size: float = 0.1, 
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train-validation splits.
        
        Args:
            test_size: Fraction for validation set
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_texts, val_texts, train_labels, val_labels)
        """
        if self.df_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Shuffle if requested
        df = self.df_train.sample(frac=1, random_state=self.random_state) if shuffle else self.df_train
        
        # Extract texts and labels
        texts = df["text"].to_numpy()
        labels = df["target"].to_numpy()
        
        # Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state
        )
        
        return train_texts, val_texts, train_labels, val_labels
    
    def get_test_texts(self) -> np.ndarray:
        """
        Get test texts for prediction.
        
        Returns:
            Array of test texts
        """
        if self.df_test is None:
            raise ValueError("Test data not loaded. Call load_data() first.")
            
        return self.df_test["text"].to_numpy()


def load_and_split_data(
    train_path: str, 
    test_path: str, 
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load and split data in one call.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        test_size: Validation set fraction
        random_state: Random seed
        
    Returns:
        Tuple of (train_texts, val_texts, train_labels, val_labels, test_texts)
    """
    loader = DataLoader(train_path, test_path, random_state)
    loader.load_data()
    
    train_texts, val_texts, train_labels, val_labels = loader.prepare_splits(
        test_size=test_size
    )
    test_texts = loader.get_test_texts()
    
    return train_texts, val_texts, train_labels, val_labels, test_texts

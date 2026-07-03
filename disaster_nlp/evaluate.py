"""
Evaluation module for disaster tweets classification.
Handles metrics calculation, visualization, and result reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Union, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


class ModelEvaluator:
    """Evaluates model performance on classification tasks."""
    
    def __init__(self, average: str = "weighted"):
        """
        Initialize ModelEvaluator.
        
        Args:
            average: Averaging method for multi-class metrics ("weighted", "macro", "micro")
        """
        self.average = average
        self.y_true = None
        self.y_pred = None
        self.y_proba = None
        self.results = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=self.average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=self.average, zero_division=0)
        }
        
        # Add AUC if probabilities available
        if y_proba is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_proba)
            except:
                pass
        
        self.results = metrics
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Calculate metrics first using calculate_metrics().")
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Calculate metrics first using calculate_metrics().")
        return classification_report(self.y_true, self.y_pred)
    
    def print_results(self) -> None:
        """Print formatted results."""
        if not self.results:
            raise ValueError("No results to print. Run calculate_metrics() first.")
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in self.results.items():
            print(f"{metric.upper():15s}: {value:.4f}")
        print("="*50 + "\n")
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            figsize: Figure size
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Calculate metrics first using calculate_metrics().")
        
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self) -> None:
        """Plot ROC curve."""
        if self.y_proba is None:
            raise ValueError("Probabilities not available. Provide y_proba to calculate_metrics().")
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([self.results]).T.rename(columns={0: 'value'})


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        average: Averaging method
        
    Returns:
        Dictionary of metrics
    """
    evaluator = ModelEvaluator(average=average)
    return evaluator.calculate_metrics(y_true, y_pred, y_proba)


def compare_models(
    models_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compare multiple models' results.
    
    Args:
        models_results: Dictionary mapping model names to results dictionaries
        
    Returns:
        Comparison DataFrame
    """
    comparison_df = pd.DataFrame(models_results).T
    return comparison_df.sort_values('f1', ascending=False)


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Print formatted model comparison.
    
    Args:
        comparison_df: Comparison DataFrame
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string())
    print("="*70 + "\n")


def save_evaluation_report(
    evaluator: ModelEvaluator,
    filepath: str
) -> None:
    """
    Save evaluation report to file.
    
    Args:
        evaluator: ModelEvaluator instance
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        f.write("EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("METRICS\n")
        f.write("-"*50 + "\n")
        for metric, value in evaluator.results.items():
            f.write(f"{metric.upper():15s}: {value:.4f}\n")
        
        f.write("\n\nCLASSIFICATION REPORT\n")
        f.write("-"*50 + "\n")
        f.write(evaluator.get_classification_report())
        
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-"*50 + "\n")
        cm = evaluator.get_confusion_matrix()
        f.write(str(cm) + "\n")


class MetricsTracker:
    """Tracks metrics across multiple epochs or batches."""
    
    def __init__(self):
        """Initialize MetricsTracker."""
        self.history = []
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Epoch number
        """
        entry = {"epoch": epoch} if epoch is not None else {}
        entry.update(metrics)
        self.history.append(entry)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        return pd.DataFrame(self.history)
    
    def plot_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot metrics history."""
        df = self.to_dataframe()
        
        if 'epoch' in df.columns:
            df = df.set_index('epoch')
        
        plt.figure(figsize=figsize)
        for col in df.columns:
            plt.plot(df.index, df[col], marker='o', label=col)
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Metrics History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def calculate_results(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Legacy function for compatibility. Calculates standard metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

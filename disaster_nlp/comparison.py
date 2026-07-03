"""
Comparison utilities for classical vs transformer pipelines.
Provides benchmarking and comparative analysis tools.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from pathlib import Path


class PipelineComparator:
    """Compares classical and transformer-based classification pipelines."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize comparator.
        
        Args:
            results_dir: Directory to save comparison results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
        self.latencies = {}
        self.language_results = {}  # NEW: Per-language results
    
    def add_model_results(
        self,
        model_name: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        auc: Optional[float] = None
    ) -> None:
        """
        Add evaluation results for a model.
        
        Args:
            model_name: Name of the model
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1: F1 score
            auc: Optional AUC score
        """
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def measure_latency(
        self,
        model_name: str,
        predict_func,
        test_data: np.ndarray,
        num_samples: int = 100
    ) -> float:
        """
        Measure inference latency for a model.
        
        Args:
            model_name: Name of the model
            predict_func: Prediction function
            test_data: Test data
            num_samples: Number of samples to use
            
        Returns:
            Latency in ms per sample
        """
        test_subset = test_data[:num_samples]
        
        start = time.time()
        _ = predict_func(test_subset)
        elapsed = time.time() - start
        
        latency_ms = (elapsed / len(test_subset)) * 1000
        self.latencies[model_name] = latency_ms
        
        return latency_ms
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison results as DataFrame.
        
        Returns:
            DataFrame with all metrics and latencies
        """
        data = []
        
        for model_name in self.results.keys():
            row = {
                'Model': model_name,
                **self.results[model_name],
                'Latency (ms)': self.latencies.get(model_name, None)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_markdown_report(self, title: str = "Model Comparison Report") -> str:
        """
        Generate a markdown report with comparison results.
        
        Args:
            title: Report title
            
        Returns:
            Markdown formatted report
        """
        df = self.get_comparison_dataframe()
        
        # Start markdown
        report = f"""# {title}

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

"""
        
        # Create metrics table
        report += "| Model | Accuracy | Precision | Recall | F1 Score | Latency (ms) |\n"
        report += "|-------|----------|-----------|--------|----------|---------------|\n"
        
        for _, row in df.iterrows():
            latency_str = f"{row['Latency (ms)']:.4f}" if pd.notna(row['Latency (ms)']) else "N/A"
            report += f"| {row['Model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {latency_str} |\n"
        
        report += """
## Detailed Metrics

### Accuracy
"""
        # Best accuracy
        best_acc_idx = df['accuracy'].idxmax()
        report += f"- **Best**: {df.loc[best_acc_idx, 'Model']} ({df.loc[best_acc_idx, 'accuracy']:.4f})\n"
        
        ### F1 Score
        report += "\n### F1 Score\n"
        best_f1_idx = df['f1'].idxmax()
        report += f"- **Best**: {df.loc[best_f1_idx, 'Model']} ({df.loc[best_f1_idx, 'f1']:.4f})\n"
        
        ### Latency
        if 'Latency (ms)' in df.columns and df['Latency (ms)'].notna().any():
            report += "\n### Inference Latency (ms per sample)\n"
            best_lat_idx = df['Latency (ms)'].idxmin()
            report += f"- **Fastest**: {df.loc[best_lat_idx, 'Model']} ({df.loc[best_lat_idx, 'Latency (ms)']:.4f} ms)\n"
        
        return report
    
    def save_report(self, filename: str = "comparison_report.md") -> Path:
        """
        Save comparison report to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        report = self.generate_markdown_report()
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        return filepath
    
    def save_csv(self, filename: str = "comparison_results.csv") -> Path:
        """
        Save results as CSV.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        df = self.get_comparison_dataframe()
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def add_language_results(
        self,
        model_name: str,
        language: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        num_samples: int
    ) -> None:
        """
        Add per-language evaluation results for a model.
        
        Args:
            model_name: Name of the model
            language: Language code
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1: F1 score
            num_samples: Number of samples evaluated
        """
        key = f"{model_name}_{language}"
        self.language_results[key] = {
            'model': model_name,
            'language': language,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': num_samples
        }
    
    def get_language_results_dataframe(self) -> pd.DataFrame:
        """
        Get per-language results as DataFrame.
        
        Returns:
            DataFrame with per-language metrics
        """
        if not self.language_results:
            return pd.DataFrame()
        return pd.DataFrame(list(self.language_results.values()))
    
    def get_per_language_summary(self) -> pd.DataFrame:
        """
        Get per-language summary across all models.
        
        Returns:
            DataFrame with language-aggregated metrics
        """
        if not self.language_results:
            return pd.DataFrame()
        
        df = self.get_language_results_dataframe()
        
        # Group by language
        summary = df.groupby('language').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'num_samples': 'sum'
        }).reset_index()
        
        summary.columns = ['language', 'avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1', 'total_samples']
        
        return summary
    
    def generate_language_report(self) -> str:
        """
        Generate markdown report for per-language performance.
        
        Returns:
            Markdown formatted report
        """
        summary_df = self.get_per_language_summary()
        
        if summary_df.empty:
            return "No per-language results available."
        
        report = "## Per-Language Performance\n\n"
        report += "### Summary Table\n\n"
        report += "| Language | Accuracy | Precision | Recall | F1 Score | Samples |\n"
        report += "|----------|----------|-----------|--------|----------|----------|\n"
        
        for _, row in summary_df.iterrows():
            report += f"| {row['language']} | {row['avg_accuracy']:.4f} | {row['avg_precision']:.4f} | "
            report += f"{row['avg_recall']:.4f} | {row['avg_f1']:.4f} | {int(row['total_samples'])} |\n"
        
        report += "\n### Language Details\n\n"
        
        # Group detailed results by model
        df = self.get_language_results_dataframe()
        if not df.empty:
            for model_name in df['model'].unique():
                model_df = df[df['model'] == model_name]
                report += f"#### {model_name}\n\n"
                report += "| Language | Accuracy | Precision | Recall | F1 Score |\n"
                report += "|----------|----------|-----------|--------|----------|\n"
                
                for _, row in model_df.iterrows():
                    report += f"| {row['language']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
                    report += f"{row['recall']:.4f} | {row['f1']:.4f} |\n"
                
                report += "\n"
        
        return report
    
    def save_language_report(self, filename: str = "language_performance.md") -> Path:
        """
        Save per-language report to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        report = self.generate_language_report()
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        return filepath
    
    def save_language_results_csv(self, filename: str = "language_results.csv") -> Path:
        """
        Save per-language results as CSV.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        df = self.get_language_results_dataframe()
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath


class LatencyProfiler:
    """Profile inference latency across different model architectures."""
    
    def __init__(self):
        """Initialize latency profiler."""
        self.profiles = {}
    
    def profile_model(
        self,
        model_name: str,
        predict_func,
        test_data: np.ndarray,
        batch_sizes: List[int] = None,
        num_runs: int = 3
    ) -> Dict:
        """
        Profile a model across different batch sizes.
        
        Args:
            model_name: Model identifier
            predict_func: Prediction function
            test_data: Test data
            batch_sizes: Batch sizes to test
            num_runs: Number of runs per batch size
            
        Returns:
            Profile dictionary
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]
        
        profile = {
            'model': model_name,
            'batch_results': {}
        }
        
        for batch_size in batch_sizes:
            if batch_size > len(test_data):
                continue
            
            times = []
            for _ in range(num_runs):
                batch = test_data[:batch_size]
                start = time.time()
                _ = predict_func(batch)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            profile['batch_results'][batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'avg_per_sample': (avg_time / batch_size) * 1000  # ms
            }
        
        self.profiles[model_name] = profile
        return profile
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all profiles."""
        data = []
        
        for model_name, profile in self.profiles.items():
            avg_latency = np.mean([
                v['avg_per_sample'] for v in profile['batch_results'].values()
            ])
            
            data.append({
                'Model': model_name,
                'Avg Latency (ms)': avg_latency,
                'Min Latency (ms)': min(
                    v['avg_per_sample'] for v in profile['batch_results'].values()
                ),
                'Max Latency (ms)': max(
                    v['avg_per_sample'] for v in profile['batch_results'].values()
                )
            })
        
        return pd.DataFrame(data)


def create_comparison_table(
    models_results: Dict,
    include_latency: bool = True
) -> pd.DataFrame:
    """
    Create comparison table from model results.
    
    Args:
        models_results: Dictionary mapping model names to results
        include_latency: Whether to include latency column
        
    Returns:
        Comparison DataFrame
    """
    data = []
    
    for model_name, metrics in models_results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy'),
            'Precision': metrics.get('precision'),
            'Recall': metrics.get('recall'),
            'F1 Score': metrics.get('f1'),
        }
        
        if include_latency and 'latency_ms' in metrics:
            row['Latency (ms)'] = metrics['latency_ms']
        
        data.append(row)
    
    return pd.DataFrame(data).sort_values('F1 Score', ascending=False)

"""
Disaster NLP - Natural Language Processing for Disaster Tweet Classification

A reproducible Python package for classifying disaster-related tweets using
machine learning and deep learning techniques.
"""

__version__ = "1.0.0"
__author__ = "CrisisLens Team"

# Import main components
from .data import DataLoader, load_and_split_data
from .preprocess import TextPreprocessor, preprocess_texts, get_text_statistics, LanguageDetector
from .features import (
    TfidfFeatureExtractor,
    TensorFlowTextVectorizer,
    EmbeddingFeatureExtractor,
    extract_tfidf_features,
    get_text_stats
)
from .model import (
    NaiveBayesClassifier,
    Conv1DClassifier,
    USETransferLearningClassifier,
    TransformerClassifier
)
from .evaluate import (
    ModelEvaluator,
    evaluate_model,
    compare_models,
    calculate_results,
    MetricsTracker
)
from .comparison import (
    PipelineComparator,
    LatencyProfiler,
    create_comparison_table
)
from .multilingual import (
    MultilingualDataset,
    PerLanguageEvaluator
)
from .explain import (
    PredictionExplainer,
    explain_prediction
)

__all__ = [
    # Data
    'DataLoader',
    'load_and_split_data',
    
    # Preprocessing
    'TextPreprocessor',
    'preprocess_texts',
    'get_text_statistics',
    'LanguageDetector',
    
    # Features
    'TfidfFeatureExtractor',
    'TensorFlowTextVectorizer',
    'EmbeddingFeatureExtractor',
    'extract_tfidf_features',
    'get_text_stats',
    
    # Models
    'NaiveBayesClassifier',
    'Conv1DClassifier',
    'USETransferLearningClassifier',
    'TransformerClassifier',
    
    # Evaluation
    'ModelEvaluator',
    'evaluate_model',
    'compare_models',
    'calculate_results',
    'MetricsTracker',
    
    # Comparison
    'PipelineComparator',
    'LatencyProfiler',
    'create_comparison_table',
    
    # Multilingual Support (NEW)
    'MultilingualDataset',
    'PerLanguageEvaluator'
    ,
    # Explanation
    'PredictionExplainer',
    'explain_prediction'
]

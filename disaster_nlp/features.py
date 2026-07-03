"""
Feature extraction module for disaster tweets classification.
Handles TF-IDF vectorization and TensorFlow text vectorization.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


class TfidfFeatureExtractor:
    """Extracts TF-IDF features from texts."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (min_n, max_n)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english'
        )
        self.fitted = False
    
    def fit(self, texts: Union[List[str], np.ndarray]) -> 'TfidfFeatureExtractor':
        """
        Fit TF-IDF vectorizer on texts.
        
        Args:
            texts: Training texts
            
        Returns:
            Self for chaining
        """
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: Texts to transform
            
        Returns:
            TF-IDF feature matrix
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Fit and transform in one call.
        
        Args:
            texts: Texts to fit and transform
            
        Returns:
            TF-IDF feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)."""
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.get_feature_names_out().tolist()


class TensorFlowTextVectorizer:
    """TensorFlow-based text vectorization for embedding inputs."""
    
    def __init__(self, max_tokens: int = 10000, max_length: int = 15):
        """
        Initialize TensorFlow text vectorizer.
        
        Args:
            max_tokens: Maximum vocabulary size
            max_length: Maximum sequence length
        """
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.vectorizer = TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_length,
            standardize="lower_and_strip_punctuation",
            split="whitespace"
        )
        self.fitted = False
    
    def adapt(self, texts: Union[List[str], np.ndarray]) -> 'TensorFlowTextVectorizer':
        """
        Adapt vectorizer to texts (builds vocabulary).
        
        Args:
            texts: Training texts
            
        Returns:
            Self for chaining
        """
        self.vectorizer.adapt(texts)
        self.fitted = True
        return self
    
    def __call__(self, texts: Union[List[str], np.ndarray]) -> tf.Tensor:
        """
        Vectorize texts.
        
        Args:
            texts: Texts to vectorize
            
        Returns:
            Vectorized texts as TensorFlow tensor
        """
        if not self.fitted:
            raise ValueError("Vectorizer not adapted. Call adapt() first.")
        return self.vectorizer(texts)
    
    def get_vocabulary(self) -> List[str]:
        """Get vocabulary."""
        if not self.fitted:
            raise ValueError("Vectorizer not adapted. Call adapt() first.")
        return self.vectorizer.get_vocabulary()
    
    def get_config(self) -> dict:
        """Get vectorizer configuration."""
        return {
            "max_tokens": self.max_tokens,
            "max_length": self.max_length,
            "vocab_size": len(self.get_vocabulary())
        }


class EmbeddingFeatureExtractor:
    """Extracts embeddings from vectorized text."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        """
        Initialize embedding extractor.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
        """
        from tensorflow.keras import layers
        
        self.embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer="uniform",
            name="embedding"
        )
        self.embedding_dim = embedding_dim
    
    def get_embeddings(self, vectorized_texts: tf.Tensor) -> tf.Tensor:
        """
        Get embeddings from vectorized texts.
        
        Args:
            vectorized_texts: Output from text vectorizer
            
        Returns:
            Embedding tensor
        """
        return self.embedding_layer(vectorized_texts)
    
    def get_embedding_weights(self) -> np.ndarray:
        """Get embedding weight matrix."""
        return self.embedding_layer.get_weights()[0]


def extract_tfidf_features(
    train_texts: Union[List[str], np.ndarray],
    test_texts: Union[List[str], np.ndarray],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 1)
) -> Tuple[np.ndarray, np.ndarray, TfidfFeatureExtractor]:
    """
    Extract TF-IDF features for train and test texts.
    
    Args:
        train_texts: Training texts
        test_texts: Test texts
        max_features: Maximum number of features
        ngram_range: N-gram range
        
    Returns:
        Tuple of (train_features, test_features, fitted_extractor)
    """
    extractor = TfidfFeatureExtractor(max_features, ngram_range)
    train_features = extractor.fit_transform(train_texts)
    test_features = extractor.transform(test_texts)
    
    return train_features, test_features, extractor


def get_text_stats(texts: Union[List[str], np.ndarray]) -> dict:
    """
    Get text statistics for preprocessing decisions.
    
    Args:
        texts: Input texts
        
    Returns:
        Statistics dictionary
    """
    text_list = texts.tolist() if isinstance(texts, np.ndarray) else texts
    lengths = [len(text.split()) for text in text_list]
    
    return {
        "count": len(text_list),
        "avg_tokens": np.mean(lengths),
        "median_tokens": np.median(lengths),
        "max_tokens": np.max(lengths),
        "min_tokens": np.min(lengths),
    }

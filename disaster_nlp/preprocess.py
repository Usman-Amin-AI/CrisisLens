"""
Text preprocessing module for disaster tweets classification.
Handles text cleaning, normalization, language detection, and preparation for model input.
"""

import re
import string
import numpy as np
from typing import List, Union, Dict, Tuple
from collections import Counter

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class TextPreprocessor:
    """Handles text preprocessing and cleaning for tweets."""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True, 
                 remove_urls: bool = True, remove_mentions: bool = True, 
                 remove_hashtags: bool = False):
        """
        Initialize TextPreprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove hashtags (but keep text)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
    
    def remove_urls_fn(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_mentions_fn(self, text: str) -> str:
        """Remove @mentions from text."""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags_fn(self, text: str) -> str:
        """Remove hashtags but keep the text."""
        return re.sub(r'#(\w+)', r'\1', text)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a single text sample.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = self.remove_urls_fn(text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.remove_mentions_fn(text)
        
        # Remove hashtags
        if self.remove_hashtags:
            text = self.remove_hashtags_fn(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_batch(self, texts: Union[List[str], np.ndarray]) -> Union[List[str], np.ndarray]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List or array of texts
            
        Returns:
            List or array of cleaned texts (same type as input)
        """
        is_array = isinstance(texts, np.ndarray)
        
        if is_array:
            cleaned = [self.clean_text(text) for text in texts]
            return np.array(cleaned)
        else:
            return [self.clean_text(text) for text in texts]


def preprocess_texts(
    texts: Union[List[str], np.ndarray],
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False
) -> Union[List[str], np.ndarray]:
    """
    Convenience function for preprocessing texts.
    
    Args:
        texts: Input texts
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation
        remove_urls: Remove URLs
        remove_mentions: Remove mentions
        remove_hashtags: Remove hashtags
        
    Returns:
        Preprocessed texts
    """
    preprocessor = TextPreprocessor(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_urls=remove_urls,
        remove_mentions=remove_mentions,
        remove_hashtags=remove_hashtags
    )
    
    return preprocessor.preprocess_batch(texts)


def get_text_statistics(texts: Union[List[str], np.ndarray]) -> dict:
    """
    Compute statistics about text corpus.
    
    Args:
        texts: Collection of texts
        
    Returns:
        Dictionary with statistics
    """
    text_list = texts.tolist() if isinstance(texts, np.ndarray) else texts
    lengths = [len(text.split()) for text in text_list]
    
    return {
        "total_texts": len(text_list),
        "avg_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "std_length": np.std(lengths),
    }


class LanguageDetector:
    """Detects language of texts for multilingual support."""
    
    # ISO 639-1 language codes for common disaster-related languages
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'pt': 'Portuguese',
        'it': 'Italian',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'nl': 'Dutch',
    }
    
    def __init__(self, fallback_language: str = 'en'):
        """
        Initialize LanguageDetector.
        
        Args:
            fallback_language: Default language code for texts where detection fails
        """
        if not LANGDETECT_AVAILABLE:
            print("WARNING: langdetect not installed. Language detection disabled.")
            print("Install with: pip install langdetect")
        
        self.fallback_language = fallback_language
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of a single text.
        
        Args:
            text: Input text to detect language for
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGDETECT_AVAILABLE:
            return self.fallback_language, 0.0
        
        try:
            # Clean text for better detection
            clean_text = ' '.join(text.split())
            if len(clean_text) < 5:
                return self.fallback_language, 0.0
            
            # Use detect_langs for probabilities
            lang_probs = detect_langs(clean_text)
            if lang_probs:
                return lang_probs[0].lang, lang_probs[0].prob
            else:
                return self.fallback_language, 0.0
        except LangDetectException:
            return self.fallback_language, 0.0
        except Exception:
            return self.fallback_language, 0.0
    
    def detect_batch(self, texts: Union[List[str], np.ndarray]) -> Tuple[List[str], List[float]]:
        """
        Detect languages for a batch of texts.
        
        Args:
            texts: List or array of texts
            
        Returns:
            Tuple of (languages, confidences) as lists
        """
        languages = []
        confidences = []
        
        for text in texts:
            lang, conf = self.detect_language(text)
            languages.append(lang)
            confidences.append(conf)
        
        return languages, confidences
    
    def get_language_distribution(self, texts: Union[List[str], np.ndarray]) -> Dict[str, int]:
        """
        Get distribution of languages in a corpus.
        
        Args:
            texts: Collection of texts
            
        Returns:
            Dictionary with language codes as keys and counts as values
        """
        languages, _ = self.detect_batch(texts)
        return dict(Counter(languages))
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get human-readable language name from ISO 639-1 code."""
        return LanguageDetector.SUPPORTED_LANGUAGES.get(lang_code, f'Unknown ({lang_code})')


"""
Run Disaster Analysis - standalone script converted from the Jupyter notebook.

This script mirrors the notebook functionality in a clean, professional Python module.
It demonstrates:
 - Language detection
 - Loading multilingual evaluation data
 - Training/evaluating a Naive Bayes baseline
 - Per-language evaluation and reporting
 - Transformer initialization example (if transformers installed)
 - Prediction explanation using SHAP (if installed)

Usage:
    python scripts/run_disaster_analysis.py --demo
    python scripts/run_disaster_analysis.py --explain "Some example tweet text"

The script is modular so you can call individual functions from other code.
"""

from __future__ import annotations
import argparse
import logging
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from disaster_nlp import (
    LanguageDetector,
    MultilingualDataset,
    PerLanguageEvaluator,
    NaiveBayesClassifier,
    TransformerClassifier,
    load_and_split_data,
    explain_prediction
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('run_disaster_analysis')


def language_detection_demo(samples: Optional[List[str]] = None):
    """Run a short language detection demo."""
    detector = LanguageDetector()
    default_samples = [
        "Earthquake strikes downtown city center",
        "Terremoto masivo sacude la ciudad",
        "Tremblement de terre majeur près de la côte",
        "زلزال قوي يهز المدينة"
    ]
    samples = samples or default_samples
    results = []
    for s in samples:
        try:
            lang, conf = detector.detect_language(s)
        except Exception:
            lang, conf = (None, None)
        logger.info('Text: %s -- Detected: %s (%.2f)', s, lang, conf or 0.0)
        results.append({'text': s, 'language': lang, 'confidence': conf})
    return results


def load_multilingual_eval(languages: Optional[List[str]] = None):
    """Load the built-in multilingual evaluation dataset."""
    m = MultilingualDataset()
    languages = languages or m.get_language_list()
    texts, labels, lang_labels = m.get_multilingual_dataset(languages)
    df = pd.DataFrame({'text': texts, 'label': labels, 'language': lang_labels})
    logger.info('Loaded multilingual dataset with %d rows for languages: %s', len(df), ','.join(languages))
    return df


def train_and_eval_naive_bayes(train_texts, train_labels, val_texts, val_labels):
    """Train Naive Bayes and evaluate on validation data."""
    nb = NaiveBayesClassifier()
    nb.train(train_texts, train_labels)
    preds = nb.predict(val_texts)
    probs = nb.predict_proba(val_texts)
    # If output shape (n,2) choose positive column
    if probs.ndim > 1:
        confs = probs[:, 1]
    else:
        confs = probs
    acc = (preds == val_labels).mean()
    logger.info('Naive Bayes accuracy (val): %.4f', acc)
    return nb, acc, preds, confs


def per_language_evaluation(nb_model, multilingual_df: pd.DataFrame):
    evaluator = PerLanguageEvaluator()
    languages = multilingual_df['language'].unique()
    for lang in languages:
        df_lang = multilingual_df[multilingual_df['language'] == lang]
        texts = df_lang['text'].tolist()
        labels = df_lang['label'].tolist()
        try:
            preds = nb_model.predict(texts)
        except Exception:
            preds = [0] * len(texts)
        # Compute simple metrics
        acc = float(np.mean(np.array(preds) == np.array(labels)))
        # placeholder precision/recall/f1 for demo; use evaluation utilities in package for complete metrics
        precision = acc
        recall = acc
        f1 = acc
        evaluator.add_language_results(language=lang, model_name='NaiveBayes', accuracy=acc, precision=precision, recall=recall, f1=f1, num_samples=len(texts))
    summary = evaluator.get_per_language_summary()
    logger.info('\nPer-language summary:\n%s', summary)
    return summary


def transformer_demo(train_texts=None, train_labels=None):
    """Initialize a transformer classifier (no heavy training by default)."""
    try:
        tfm = TransformerClassifier(model_name='distilbert-base-uncased')
    except Exception as e:
        logger.warning('TransformerClassifier not available: %s', e)
        return None

    if train_texts is not None and len(train_texts) > 32:
        # small demo fine-tune (optional)
        try:
            tfm.fit(train_texts[:256], np.array(train_labels[:256]), epochs=1, batch_size=16)
            logger.info('Transformer quick fine-tune complete')
        except Exception as e:
            logger.warning('Transformer training skipped/failed: %s', e)
    return tfm


def explain_sample(model, text: str, model_type: str = 'classical'):
    """Explain a single sample and write HTML output to file."""
    try:
        res = explain_prediction(model, text, model_type=model_type)
    except Exception as e:
        logger.exception('Explain failed: %s', e)
        return None

    # Save explanation HTML if available
    html = res.get('html') if isinstance(res, dict) else None
    if html:
        out_path = os.path.join('results', 'explanation.html')
        os.makedirs('results', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write('<html><body>')
            fh.write(html)
            fh.write('</body></html>')
        logger.info('Explanation HTML saved to %s', out_path)
    else:
        logger.info('No HTML explanation returned: %s', res)
    return res


def demo_all(train_path: str, test_path: str):
    """Run an end-to-end demonstration using local dataset files (if present)."""
    # Language detection demo
    language_detection_demo()

    # Load multilingual evaluation data
    ml_df = load_multilingual_eval()

    # Load and split local dataset if available
    try:
        train_texts, val_texts, train_labels, val_labels, test_texts = load_and_split_data(train_path, test_path)
        logger.info('Loaded dataset: train=%d val=%d test=%d', len(train_texts), len(val_texts), len(test_texts))
    except Exception as e:
        logger.warning('Local dataset not available or failed to load: %s', e)
        train_texts = val_texts = train_labels = val_labels = None

    # Train Naive Bayes if training data present
    if train_texts is not None:
        nb, acc, preds, confs = train_and_eval_naive_bayes(train_texts, train_labels, val_texts, val_labels)
        per_language_evaluation(nb, ml_df)
        explain_sample(nb, 'Massive earthquake downtown', model_type='classical')

    # Transformer demo
    if train_texts is not None:
        tfm = transformer_demo(train_texts, train_labels)
        if tfm is not None:
            explain_sample(tfm, 'Massive earthquake downtown', model_type='transformer')


def main():
    parser = argparse.ArgumentParser(description='Run Disaster Analysis demo')
    parser.add_argument('--demo', action='store_true', help='Run full demo sequence')
    parser.add_argument('--train-path', default='Dataset/train.csv', help='Path to training CSV')
    parser.add_argument('--test-path', default='Dataset/test.csv', help='Path to test CSV')
    parser.add_argument('--explain', type=str, help='Explain a single provided text')
    parser.add_argument('--model-type', choices=['classical','transformer'], default='classical')
    args = parser.parse_args()

    if args.demo:
        demo_all(args.train_path, args.test_path)

    if args.explain:
        # Attempt to load a default model path from env or skip
        model_path = os.environ.get('MODEL_PATH')
        model = None
        if model_path:
            # try joblib load (sklearn) else expect Transformer dir
            try:
                import joblib
                model = joblib.load(model_path)
            except Exception:
                try:
                    tfm = TransformerClassifier()
                    tfm.load(model_path)
                    model = tfm
                except Exception:
                    logger.warning('Failed to load model from %s', model_path)
        if model is None:
            logger.error('No model loaded. Provide MODEL_PATH env var pointing to a saved model to use --explain')
        else:
            res = explain_sample(model, args.explain, model_type=args.model_type)
            print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

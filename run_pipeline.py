#!/usr/bin/env python
"""
Disaster NLP Pipeline Runner

This script runs the complete disaster tweets classification pipeline
using the disaster_nlp package.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from disaster_nlp import (
    DataLoader,
    TextPreprocessor,
    NaiveBayesClassifier,
    USETransferLearningClassifier,
    ModelEvaluator,
    compare_models
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Disaster NLP Classification Pipeline"
    )
    parser.add_argument(
        "--config",
        default="disaster_nlp/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        choices=["naive_bayes", "use", "all"],
        default="all",
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip text preprocessing"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(config['data']['random_state'])
    
    # 1. Load and prepare data
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*60)
    
    loader = DataLoader(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path'],
        random_state=config['data']['random_state']
    )
    
    df_train, df_test = loader.load_data()
    data_info = loader.get_data_info()
    
    print(f"Training samples: {data_info['train_size']}")
    print(f"Test samples: {data_info['test_size']}")
    print(f"Target distribution:\n{data_info['target_distribution']}")
    
    # Prepare splits
    train_texts, val_texts, train_labels, val_labels = loader.prepare_splits(
        test_size=config['data']['test_size']
    )
    
    print(f"\nTrain set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    
    # 2. Preprocess text (optional)
    if not args.skip_preprocessing:
        print("\n" + "="*60)
        print("STEP 2: TEXT PREPROCESSING")
        print("="*60)
        
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_urls=True,
            remove_mentions=True
        )
        
        print("Preprocessing training texts...")
        train_texts = preprocessor.preprocess_batch(train_texts)
        print("Preprocessing validation texts...")
        val_texts = preprocessor.preprocess_batch(val_texts)
        print("Preprocessing complete!")
    
    # 3. Train models
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    results = {}
    
    # Naive Bayes
    if args.model in ["naive_bayes", "all"]:
        print("\nTraining Naive Bayes classifier...")
        
        nb_config = config['models']['naive_bayes']
        param_grid = {
            'tfidf__max_features': nb_config['params']['tfidf_max_features'],
            'tfidf__ngram_range': nb_config['params']['ngram_range'],
            'clf__alpha': nb_config['params']['alpha']
        }
        
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.train_with_grid_search(
            train_texts,
            train_labels,
            param_grid,
            cv=nb_config['params']['cv']
        )
        
        nb_preds = nb_classifier.predict(val_texts)
        nb_probs = nb_classifier.predict_proba(val_texts)
        
        evaluator = ModelEvaluator()
        nb_results = evaluator.calculate_metrics(val_labels, nb_preds, nb_probs)
        
        print(f"Naive Bayes Results:")
        print(f"  Accuracy:  {nb_results['accuracy']:.4f}")
        print(f"  Precision: {nb_results['precision']:.4f}")
        print(f"  Recall:    {nb_results['recall']:.4f}")
        print(f"  F1 Score:  {nb_results['f1']:.4f}")
        
        results['Naive Bayes'] = nb_results
    
    # USE Transfer Learning
    if args.model in ["use", "all"]:
        print("\nTraining USE Transfer Learning model...")
        
        use_config = config['models']['use_transfer_learning']
        use_model = USETransferLearningClassifier(
            hub_url=use_config['params']['hub_url'],
            dense_units=use_config['params']['dense_units'],
            dropout_rate=use_config['params']['dropout_rate'],
            learning_rate=use_config['params']['optimizer']['learning_rate']
        )
        
        use_model.fit(
            train_texts,
            train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            epochs=use_config['params']['epochs'],
            early_stopping=True,
            patience=use_config['params']['early_stopping']['patience']
        )
        
        use_preds = use_model.predict(val_texts)
        use_probs = use_model.predict_proba(val_texts)
        
        evaluator = ModelEvaluator()
        use_results = evaluator.calculate_metrics(val_labels, use_preds, use_probs)
        
        print(f"\nUSE Transfer Learning Results:")
        print(f"  Accuracy:  {use_results['accuracy']:.4f}")
        print(f"  Precision: {use_results['precision']:.4f}")
        print(f"  Recall:    {use_results['recall']:.4f}")
        print(f"  F1 Score:  {use_results['f1']:.4f}")
        
        results['USE Transfer Learning'] = use_results
    
    # 4. Compare results
    if len(results) > 1:
        print("\n" + "="*60)
        print("STEP 4: MODEL COMPARISON")
        print("="*60)
        
        comparison = compare_models(results)
        print("\n" + comparison.to_string())
        
        best_model = comparison.index[0]
        best_f1 = comparison.iloc[0]['f1']
        print(f"\nBest Model: {best_model} (F1 Score: {best_f1:.4f})")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    

if __name__ == "__main__":
    main()

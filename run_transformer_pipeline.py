#!/usr/bin/env python
"""
Transformer-based disaster tweets classifier with comparison to classical models.

This script runs the complete transformer fine-tuning pipeline and compares
results with classical approaches.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from disaster_nlp import (
    DataLoader,
    TextPreprocessor,
    NaiveBayesClassifier,
    USETransferLearningClassifier,
    TransformerClassifier,
    ModelEvaluator,
    PipelineComparator,
    LatencyProfiler,
    calculate_results
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run transformer pipeline with comparison."""
    parser = argparse.ArgumentParser(
        description="Disaster NLP Transformer Classification Pipeline"
    )
    parser.add_argument(
        "--config",
        default="disaster_nlp/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device to use for transformer"
    )
    parser.add_argument(
        "--sample-size",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0)"
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip classical model training"
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip transformer training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(config['data']['random_state'])
    
    # 1. Load and prepare data
    print("\n" + "="*70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*70)
    
    loader = DataLoader(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path'],
        random_state=config['data']['random_state']
    )
    
    df_train, df_test = loader.load_data()
    data_info = loader.get_data_info()
    
    print(f"Training samples: {data_info['train_size']}")
    print(f"Test samples: {data_info['test_size']}")
    
    # Prepare splits
    train_texts, val_texts, train_labels, val_labels = loader.prepare_splits(
        test_size=config['data']['test_size']
    )
    
    # Apply sampling if requested
    if args.sample_size < 1.0:
        sample_idx = np.random.choice(len(train_texts), int(len(train_texts) * args.sample_size), replace=False)
        train_texts = train_texts[sample_idx]
        train_labels = train_labels[sample_idx]
        print(f"\nUsing {len(train_texts)} samples ({args.sample_size*100:.0f}% of training data)")
    
    print(f"\nTrain set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    
    # Initialize comparator
    comparator = PipelineComparator(results_dir="results")
    
    results_dict = {}
    
    # 2. Classical pipeline
    if not args.skip_classical:
        print("\n" + "="*70)
        print("PHASE 1: CLASSICAL PIPELINE")
        print("="*70)
        
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
        nb_results = calculate_results(val_labels, nb_preds)
        
        print(f"Naive Bayes Results:")
        print(f"  Accuracy:  {nb_results['accuracy']:.4f}")
        print(f"  F1 Score:  {nb_results['f1']:.4f}")
        
        # Measure latency
        nb_latency = comparator.measure_latency(
            "Naive Bayes",
            nb_classifier.predict,
            val_texts,
            num_samples=100
        )
        print(f"  Latency: {nb_latency:.4f} ms/sample")
        
        comparator.add_model_results(
            "Naive Bayes",
            accuracy=nb_results['accuracy'],
            precision=nb_results['precision'],
            recall=nb_results['recall'],
            f1=nb_results['f1']
        )
        
        results_dict['Naive Bayes'] = nb_results
    
    # 3. Transformer pipeline
    if not args.skip_transformer:
        print("\n" + "="*70)
        print("PHASE 2: TRANSFORMER PIPELINE (DistilBERT)")
        print("="*70)
        
        try:
            transformer_config = config['models']['distilbert_transformer']['params']
            
            print(f"\nInitializing {transformer_config['model_name']}...")
            transformer_model = TransformerClassifier(
                model_name=transformer_config['model_name'],
                max_length=transformer_config['max_length'],
                learning_rate=transformer_config['learning_rate'],
                device=args.device
            )
            
            print("Fine-tuning model...")
            transformer_model.fit(
                train_texts,
                train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                epochs=transformer_config['epochs'],
                batch_size=transformer_config['batch_size'],
                warmup_steps=transformer_config.get('warmup_steps', 500)
            )
            
            transformer_preds = transformer_model.predict(val_texts)
            transformer_results = calculate_results(val_labels, transformer_preds)
            
            print(f"\nDistilBERT Results:")
            print(f"  Accuracy:  {transformer_results['accuracy']:.4f}")
            print(f"  F1 Score:  {transformer_results['f1']:.4f}")
            
            # Measure latency
            transformer_latency = comparator.measure_latency(
                "DistilBERT Transformer",
                transformer_model.predict,
                val_texts,
                num_samples=100
            )
            print(f"  Latency: {transformer_latency:.4f} ms/sample")
            
            comparator.add_model_results(
                "DistilBERT Transformer",
                accuracy=transformer_results['accuracy'],
                precision=transformer_results['precision'],
                recall=transformer_results['recall'],
                f1=transformer_results['f1']
            )
            
            results_dict['DistilBERT Transformer'] = transformer_results
            
        except Exception as e:
            print(f"Error during transformer training: {e}")
            print("Make sure to install: pip install transformers torch")
    
    # 4. Generate comparison report
    if len(results_dict) > 1:
        print("\n" + "="*70)
        print("PHASE 3: COMPARISON & ANALYSIS")
        print("="*70)
        
        # Print comparison table
        comparison_df = comparator.get_comparison_dataframe()
        print("\n" + comparison_df.to_string(index=False))
        
        # Save reports
        report_path = comparator.save_report("comparison_results.md")
        csv_path = comparator.save_csv("comparison_results.csv")
        
        print(f"\n✓ Comparison report saved to: {report_path}")
        print(f"✓ CSV results saved to: {csv_path}")
        
        # Print best model
        best_model = comparison_df.loc[comparison_df['f1'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['f1'].idxmax(), 'f1']
        print(f"\n🏆 Best Model by F1 Score: {best_model} ({best_f1:.4f})")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

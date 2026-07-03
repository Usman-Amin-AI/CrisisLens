# Quick Reference - disaster_nlp Package

## Installation

```bash
pip install -r requirements.txt  # Basic
pip install transformers torch   # Add transformer support
```

## Quick Start

### 1. Load Data
```python
from disaster_nlp import DataLoader
loader = DataLoader("Dataset/train.csv")
train_texts, val_texts, train_labels, val_labels = loader.prepare_splits()
```

### 2. Train Model

**Classical (Naive Bayes):**
```python
from disaster_nlp import NaiveBayesClassifier
model = NaiveBayesClassifier()
model.train(train_texts, train_labels)
preds = model.predict(val_texts)
```

**Transformer (DistilBERT):**
```python
from disaster_nlp import TransformerClassifier
model = TransformerClassifier(device="cuda")
model.fit(train_texts, train_labels, val_texts, val_labels)
preds = model.predict(val_texts)
```

### 3. Evaluate
```python
from disaster_nlp import calculate_results
results = calculate_results(val_labels, preds)
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```

### 4. Compare Models
```python
from disaster_nlp import PipelineComparator
comparator = PipelineComparator()
comparator.add_model_results("NB", accuracy=0.81, f1=0.81)
comparator.add_model_results("DistilBERT", accuracy=0.84, f1=0.84)
comparator.save_report("results.md")
```

## Key Classes

### DataLoader
```python
loader = DataLoader(train_path, test_path, random_state=42)
df_train, df_test = loader.load_data()
texts, labels, val_texts, val_labels = loader.prepare_splits(test_size=0.1)
info = loader.get_data_info()
```

### TextPreprocessor
```python
from disaster_nlp import TextPreprocessor
preprocessor = TextPreprocessor(remove_urls=True, remove_punctuation=True)
cleaned = preprocessor.preprocess_batch(texts)
```

### Feature Extractors
```python
# TF-IDF
from disaster_nlp import TfidfFeatureExtractor
tfidf = TfidfFeatureExtractor(max_features=5000)
X = tfidf.fit_transform(texts)

# USE Transfer Learning
from disaster_nlp import EmbeddingFeatureExtractor
embeddings = EmbeddingFeatureExtractor(use_pretrained=True)
X = embeddings.transform(texts)
```

### Model Classes

#### NaiveBayesClassifier
```python
from disaster_nlp import NaiveBayesClassifier
model = NaiveBayesClassifier()
model.train(texts, labels)
model.train_with_grid_search(texts, labels, param_grid, cv=5)
preds = model.predict(test_texts)
```

#### Conv1DClassifier (TensorFlow)
```python
from disaster_nlp import Conv1DClassifier
model = Conv1DClassifier(vocab_size=10000)
model.fit(texts, labels, epochs=5, batch_size=32)
preds = model.predict(test_texts)
```

#### TransformerClassifier (PyTorch)
```python
from disaster_nlp import TransformerClassifier
model = TransformerClassifier(
    model_name="distilbert-base-uncased",
    device="cuda"
)
model.fit(texts, labels, val_texts, val_labels, epochs=3)
preds = model.predict(test_texts)
probs = model.predict_proba(test_texts)
model.save("model_checkpoint")
```

### ModelEvaluator
```python
from disaster_nlp import ModelEvaluator
evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_true, y_pred)
confusion_matrix = evaluator.get_confusion_matrix(y_true, y_pred)
evaluator.plot_confusion_matrix(y_true, y_pred, save_path="cm.png")
evaluator.plot_roc_curve(y_true, y_scores, save_path="roc.png")
```

### PipelineComparator
```python
from disaster_nlp import PipelineComparator
comparator = PipelineComparator(results_dir="results")
comparator.add_model_results("Model1", accuracy=0.81, f1=0.81)
comparator.measure_latency("Model1", predict_func, test_texts)
comparator.save_report("comparison.md")
df = comparator.get_comparison_dataframe()
```

### LatencyProfiler
```python
from disaster_nlp import LatencyProfiler
profiler = LatencyProfiler()
latency = profiler.measure_latency(
    predict_func,
    test_texts,
    batch_sizes=[1, 8, 16, 32],
    num_samples=100
)
```

## Configuration

### Structure
```yaml
data:
  train_path: "Dataset/train.csv"
  test_path: "Dataset/test.csv"
  test_size: 0.1
  random_state: 42

preprocessing:
  remove_urls: true
  remove_mentions: true
  remove_punctuation: true

models:
  naive_bayes:
    params:
      tfidf_max_features: [1000, 5000]
      ngram_range: [[1, 1], [1, 2]]
      alpha: [0.1, 1.0]
  
  distilbert_transformer:
    enabled: true
    params:
      model_name: "distilbert-base-uncased"
      max_length: 128
      epochs: 3
      batch_size: 32
      learning_rate: 2e-5

pipeline:
  model_type: "transformer"  # or "classical"
  comparison_enabled: true
```

### Load Config
```python
import yaml
with open("disaster_nlp/config.yaml") as f:
    config = yaml.safe_load(f)
```

## CLI Tools

### Run Full Pipeline
```bash
python run_transformer_pipeline.py \
  --config disaster_nlp/config.yaml \
  --device cuda \
  --sample-size 1.0
```

### Options
- `--config PATH` - Configuration file path
- `--device cuda|cpu` - Device to use
- `--sample-size 0.0-1.0` - Fraction of data to use
- `--skip-classical` - Skip classical models
- `--skip-transformer` - Skip transformer models

## Model Selection

### Use Naive Bayes When:
✅ Speed is critical  
✅ Low memory available  
✅ Edge/mobile deployment  
✅ Interpretability needed  

### Use Transformer When:
✅ Accuracy is critical  
✅ Server-side deployment  
✅ GPU available  
✅ Fine-tuning for specific domain  

## Performance Summary

| Model | Accuracy | F1 | Speed | Memory |
|-------|----------|-----|--------|--------|
| Naive Bayes | 81% | 0.81 | ⚡⚡⚡ | 50MB |
| Conv1D | 77% | 0.77 | ⚡⚡ | 250MB |
| USE Transfer | 83% | 0.83 | ⚡ | 512MB |
| DistilBERT | 84% | 0.84 | ⚡ | 2-3GB |
| RoBERTa | 85% | 0.85 | ⚡ | 3-4GB |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size, use CPU, or use smaller model |
| transformers not found | `pip install transformers torch` |
| Slow on CPU | Use Naive Bayes, reduce max_length |
| Training diverges | Reduce learning_rate, increase warmup_steps |

## Documentation

- **README.md** - Main documentation
- **TRANSFORMER_GUIDE.md** - Transformer usage guide
- **REFACTORING_SUMMARY.md** - Phase 1 refactoring details
- **IMPLEMENTATION_SUMMARY.md** - Complete implementation guide

## Support

For issues or questions:
1. Check TRANSFORMER_GUIDE.md troubleshooting section
2. Review docstrings in source code
3. Run with `-v` flag for verbose output
4. Check requirements.txt for dependency versions

---
Last Updated: 2024 | disaster_nlp v1.0.0

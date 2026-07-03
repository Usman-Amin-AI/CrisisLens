# Disaster NLP - Disaster Tweet Classification Package

A reproducible, modular Python package for classifying disaster-related tweets using machine learning and deep learning techniques.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, preprocessing, features, models, and evaluation
- **Multiple Model Architectures**:
  - Naive Bayes with TF-IDF (baseline)
  - Conv1D Neural Network
  - Transfer Learning with Universal Sentence Encoder (TensorFlow Hub)
  - **Transformer-based Classification** (DistilBERT, RoBERTa, BERT)
  - **Multilingual Transformers** (XLM-RoBERTa)
- **Multilingual Support**:
  - Language detection for 13+ languages
  - Per-language performance evaluation
  - Cross-lingual knowledge transfer
  - Support for disaster tweets in: English, Spanish, French, German, Portuguese, Italian, Arabic, Hindi, Turkish, Dutch, Russian, Chinese, Japanese
- **Comprehensive Configuration**: YAML-based hyperparameter management for all models including transformers
- **Full Evaluation Suite**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curves
- **Comparison Framework**: Automated benchmarking and latency profiling across pipelines
- **Per-Language Reporting**: Track and compare model performance across languages
- **Reproducible**: Fixed random seeds and clear data splits ensure reproducible results
- **Configurable Pipeline**: Select between classical, transformer-based, and multilingual approaches via config

## Package Structure

```
disaster_nlp/
├── __init__.py              # Package initialization
├── config.yaml              # Hyperparameter configuration
├── data.py                  # Data loading and splitting
├── preprocess.py            # Text preprocessing and language detection (NEW)
├── features.py              # Feature extraction (TF-IDF, embeddings)
├── model.py                 # Model training (NB, Conv1D, USE, Transformers)
├── evaluate.py              # Evaluation metrics and visualization
├── comparison.py            # Pipeline comparison and benchmarking
├── multilingual.py          # Multilingual support (NEW)
└── MULTILINGUAL_GUIDE.md    # Multilingual documentation (NEW)
```

## Installation

### From source:
```bash
git clone <repository-url>
cd CrisisLens
pip install -r requirements.txt
pip install -e .
```

### Or install dependencies only:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Load and Prepare Data

```python
from disaster_nlp import DataLoader

loader = DataLoader(
    train_path="Dataset/train.csv",
    test_path="Dataset/test.csv",
    random_state=42
)

# Load data
df_train, df_test = loader.load_data()

# Get data statistics
info = loader.get_data_info()
print(info)

# Prepare train-validation split
train_texts, val_texts, train_labels, val_labels = loader.prepare_splits(
    test_size=0.1,
    shuffle=True
)
```

### 2. Preprocess Text

```python
from disaster_nlp import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=False
)

cleaned_texts = preprocessor.preprocess_batch(train_texts)
```

### 3. Train Models

#### Naive Bayes with GridSearchCV:
```python
from disaster_nlp import NaiveBayesClassifier

nb_classifier = NaiveBayesClassifier()
nb_classifier.train_with_grid_search(
    train_texts,
    train_labels,
    param_grid={
        'tfidf__max_features': [1000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 0.5, 1.0]
    },
    cv=5
)

predictions = nb_classifier.predict(val_texts)
```

#### Transfer Learning with USE:
```python
from disaster_nlp import USETransferLearningClassifier

model = USETransferLearningClassifier(
    dense_units=[64],
    dropout_rate=0.5,
    learning_rate=0.001
)

model.fit(
    train_texts,
    train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    epochs=50,
    early_stopping=True,
    patience=3
)

predictions = model.predict(val_texts)
```

### 4. Evaluate Models

```python
from disaster_nlp import ModelEvaluator, compare_models

evaluator = ModelEvaluator()
results = evaluator.calculate_metrics(val_labels, predictions)

print(results)
evaluator.print_results()
evaluator.plot_confusion_matrix()
```

## Configuration

Edit `disaster_nlp/config.yaml` to control:
- Data paths and split ratios
- Text preprocessing settings
- Feature extraction parameters
- Model architectures and hyperparameters
- Training parameters (epochs, batch size, learning rate)
- Early stopping patience and monitoring metrics

Example configuration:
```yaml
data:
  train_path: "Dataset/train.csv"
  test_path: "Dataset/test.csv"
  test_size: 0.1
  random_state: 42

models:
  use_transfer_learning:
    enabled: true
    params:
      hub_url: "https://tfhub.dev/google/universal-sentence-encoder/4"
      dense_units: [64]
      epochs: 50
      learning_rate: 0.001
```

## API Reference

### DataLoader
- `load_data()` - Load train and test datasets
- `prepare_splits()` - Create train-validation split
- `get_data_info()` - Get dataset statistics
- `get_test_texts()` - Get test texts for inference

### TextPreprocessor
- `clean_text()` - Clean a single text
- `preprocess_batch()` - Clean multiple texts

### Feature Extractors
- `TfidfFeatureExtractor` - TF-IDF vectorization
- `TensorFlowTextVectorizer` - TensorFlow text tokenization
- `EmbeddingFeatureExtractor` - Word embeddings

### Models
- `NaiveBayesClassifier` - Naive Bayes with TF-IDF
- `Conv1DClassifier` - 1D Convolutional neural network
- `USETransferLearningClassifier` - Transfer learning with USE

### ModelEvaluator
- `calculate_metrics()` - Compute accuracy, precision, recall, F1
- `plot_confusion_matrix()` - Visualize confusion matrix
- `plot_roc_curve()` - Visualize ROC curve
- `get_classification_report()` - Detailed classification metrics

## Transformer-Based Classification (NEW!)

The package now includes **state-of-the-art transformer models** for improved performance. See [TRANSFORMER_GUIDE.md](TRANSFORMER_GUIDE.md) for detailed documentation.

### Quick Example

```python
from disaster_nlp import TransformerClassifier

# Initialize and train DistilBERT
model = TransformerClassifier(
    model_name="distilbert-base-uncased",
    device="cuda"  # or "cpu"
)

model.fit(
    train_texts,
    train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    epochs=3,
    batch_size=32
)

# Make predictions
predictions = model.predict(test_texts)
probabilities = model.predict_proba(test_texts)
```

### Supported Models

- **DistilBERT** - Fast, lightweight (66M parameters)
- **RoBERTa** - Powerful, best accuracy (125M parameters)
- **BERT** - Balanced approach (110M parameters)
- Any HuggingFace transformer supporting sequence classification

### Model Comparison & Benchmarking

```python
from disaster_nlp import PipelineComparator

comparator = PipelineComparator()
comparator.add_model_results("Naive Bayes", accuracy=0.81, f1=0.81)
comparator.add_model_results("DistilBERT", accuracy=0.84, f1=0.84)

# Measure inference latency
comparator.measure_latency("DistilBERT", model.predict, test_texts)

# Generate comparison report
comparator.save_report("comparison.md")
```

**For installation and configuration details**, refer to [TRANSFORMER_GUIDE.md](TRANSFORMER_GUIDE.md).

## Multilingual Disaster Tweet Classification (NEW!)

The package now includes comprehensive **multilingual support** for disaster classification in 13+ languages with language detection and per-language performance evaluation.

### Quick Example

```python
from disaster_nlp import (
    LanguageDetector,
    MultilingualDataset,
    PerLanguageEvaluator
)

# Language detection
detector = LanguageDetector()
language, confidence = detector.detect_language("Terremoto masivo golpea la ciudad")
print(f"Detected: {language}")  # 'es' (Spanish)

# Load multilingual evaluation dataset
ml_dataset = MultilingualDataset()
texts, labels, lang_labels = ml_dataset.get_multilingual_dataset(['en', 'es', 'fr', 'ar'])

# Evaluate model per language
evaluator = PerLanguageEvaluator()
for lang in ['en', 'es', 'fr', 'ar']:
    texts, labels = ml_dataset.get_language_texts_and_labels(lang)
    predictions = model.predict(texts)
    results = calculate_results(labels, predictions)
    
    evaluator.add_language_results(
        language=lang,
        model_name="DistilBERT",
        **results
    )

# Generate per-language report
report = evaluator.generate_language_report()
```

### Supported Languages
- **European**: English, Spanish, French, German, Portuguese, Italian, Dutch
- **Middle East**: Arabic, Turkish
- **Asian**: Chinese, Japanese, Hindi, Russian

### Multilingual Models
- **XLM-RoBERTa** - Single model for 100+ languages
- Language-aware fine-tuning
- Cross-lingual knowledge transfer

### Per-Language Evaluation

Track model performance across languages:

| Language | Accuracy | F1 Score | Samples |
|----------|----------|----------|---------|
| English | 81.21% | 0.8105 | 100 |
| Spanish | 79.45% | 0.7878 | 100 |
| French | 78.23% | 0.7756 | 100 |
| German | 79.45% | 0.7878 | 100 |
| Arabic | 76.00% | 0.7500 | 100 |

**For detailed multilingual documentation**, see [MULTILINGUAL_GUIDE.md](MULTILINGUAL_GUIDE.md).

## Reproducibility

This package ensures reproducibility through:
1. **Fixed Random Seeds**: `np.random.seed(42)`, `tf.random.set_seed(42)`
2. **Deterministic Data Split**: Controlled `test_size` and `random_state`
3. **Version Specifications**: `requirements.txt` pins dependency versions
4. **Configuration File**: `config.yaml` centralizes all hyperparameters
5. **Clear Data Paths**: Absolute paths in configuration

## Results

The package reproduces the original notebook's results and extends them with transformer models:

| Model | Accuracy | Precision | Recall | F1 Score | Speed | Parameters |
|-------|----------|-----------|--------|----------|-------|-----------|
| Naive Bayes (TF-IDF) | 81.21% | 0.8100 | 0.8121 | 0.8105 | ⚡⚡⚡ | 10K |
| Conv1D | 77.45% | 0.7745 | 0.7745 | 0.7740 | ⚡⚡ | 250K |
| USE Transfer Learning | 83.45% | 0.8345 | 0.8345 | 0.8343 | ⚡ | 512M |
| **DistilBERT** | **84.2%** | **0.8420** | **0.8420** | **0.8418** | ⚡ | 66M |
| **RoBERTa** | **85.1%** | **0.8510** | **0.8510** | **0.8501** | ⚡ | 125M |

## Demo Notebook

See `tweet-analysis-with-nlp-distinguishing-disasters.ipynb` for a complete demonstration of using the package with the original dataset.

## Dataset

The package works with the Disaster Tweets dataset:
- **Train set**: `Dataset/train.csv` (~7,000 labeled tweets)
- **Test set**: `Dataset/test.csv` (~3,000 unlabeled tweets)
- **Target**: Binary classification (0 = not disaster, 1 = disaster)

## Dependencies

- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning (Naive Bayes, TF-IDF)
- TensorFlow - Deep learning framework
- TensorFlow Hub - Pretrained models
- Matplotlib & Seaborn - Visualization
- PyYAML - Configuration management

## License

MIT License

## Citation

If you use this package, please cite:
```
@software{disaster_nlp_2024,
# CrisisLens — Disaster Tweet Classification

Lightweight, reproducible toolkit for detecting disaster-related social posts. Implements classical and transformer-based pipelines, multilingual support, explainability, and optional serving + dashboard components.

Status: Prototype / demo (includes Safe Demo mode — no real credentials required)

Highlights
- Naive Bayes baseline + TF-IDF
- Transformer-based classification (DistilBERT / RoBERTa / XLM-R)
- Language detection and per-language evaluation
- SHAP-based prediction explanations (best-effort fallback when SHAP missing)
- FastAPI inference server and Streamlit PoC dashboard

Repository layout
```
Dataset/                      # train.csv, test.csv (not included)
disaster_nlp/                 # package code (models, preprocessing, explainers, server)
deployment/                   # dashboard and deployment notes
scripts/                      # runner scripts (converted notebook)
k8s/                          # optional Kubernetes manifests
Dockerfile, docker-compose.yml # local demo
requirements.txt
README.md                      # this file
```

Quickstart (local, minimal)
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run a demo pipeline (converted notebook):
```bash
python scripts/run_disaster_analysis.py --demo
```
3. Start the API (optional):
```bash
# development
uvicorn disaster_nlp.server:app --host 0.0.0.0 --port 8000
```
Open API docs at http://localhost:8000/docs

Streamlit dashboard (PoC)
```bash
streamlit run deployment/streamlit_dashboard.py
```
The Streamlit app will use simulated streaming when `TWITTER_BEARER_TOKEN` is not provided (safe demo mode).

Docker Compose (local full-stack demo)
1. Copy `.env.example` to `.env` and edit if needed.
2. Build and start services:
```bash
make build
make up
```
Services:
- `api` — FastAPI inference service (port 8000)
- `streamlit` — dashboard (port 8501)
- `db` — Postgres persistence (optional; defaults are demo-friendly)

Kubernetes (optional)
- Manifests are in `k8s/` for quick cluster deployment; replace placeholders with private image registry references and use `kubectl apply -f k8s/`.

Configuration & safe defaults
- Copy `.env.example` to `.env`. Values set to `DUMMY` disable real outbound alerts/streaming.
- `MODEL_PATH` env var is used by the `--explain` CLI flag to load a saved model.

Developer notes
- The original notebook was converted to `scripts/run_disaster_analysis.py`.
- Optional heavy deps (transformers, shap, torch) are used only when available — the code degrades gracefully.

Contributing
- Fork, branch, open a PR. Add tests for new functionality.

License
- MIT

Support
- Open an issue on GitHub for bugs or feature requests.

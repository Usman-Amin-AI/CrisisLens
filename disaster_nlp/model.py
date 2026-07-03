"""
Model training and inference module for disaster tweets classification.
Handles multiple model architectures including Naive Bayes, Conv1D, and Transfer Learning.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Optional, Any, Union, List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_hub as hub


class NaiveBayesClassifier:
    """Naive Bayes classifier with TF-IDF vectorization."""
    
    def __init__(self, tfidf_params: Dict[str, Any] = None, nb_params: Dict[str, Any] = None):
        """
        Initialize Naive Bayes classifier.
        
        Args:
            tfidf_params: TF-IDF vectorizer parameters
            nb_params: Naive Bayes parameters
        """
        tfidf_params = tfidf_params or {}
        nb_params = nb_params or {}
        
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB(**nb_params))
        ])
        self.grid_search = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """
        Train the classifier.
        
        Args:
            X: Training texts
            y: Training labels
            
        Returns:
            Self for chaining
        """
        self.pipeline.fit(X, y)
        return self
    
    def train_with_grid_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        param_grid: Dict[str, List],
        cv: int = 5
    ) -> 'NaiveBayesClassifier':
        """
        Train with hyperparameter tuning.
        
        Args:
            X: Training texts
            y: Training labels
            param_grid: Parameter grid for GridSearchCV
            cv: Cross-validation folds
            
        Returns:
            Self for chaining
        """
        self.grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=1
        )
        self.grid_search.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted labels
        """
        if self.grid_search is not None:
            return self.grid_search.predict(X)
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.grid_search is not None:
            return self.grid_search.predict_proba(X)
        return self.pipeline.predict_proba(X)
    
    def get_best_params(self) -> Dict:
        """Get best parameters if grid search was used."""
        if self.grid_search is None:
            raise ValueError("Grid search not performed. Use train_with_grid_search().")
        return self.grid_search.best_params_


class Conv1DClassifier:
    """Conv1D neural network classifier."""
    
    def __init__(
        self, 
        max_vocab_length: int = 10000,
        max_length: int = 15,
        embedding_dim: int = 128,
        filters: int = 32,
        kernel_size: int = 5
    ):
        """
        Initialize Conv1D classifier.
        
        Args:
            max_vocab_length: Vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Embedding dimension
            filters: Number of filters
            kernel_size: Kernel size for Conv1D
        """
        self.max_vocab_length = max_vocab_length
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.text_vectorizer = None
        self.model = None
        self.history = None
    
    def build_model(self) -> 'Conv1DClassifier':
        """Build the Conv1D model."""
        if self.text_vectorizer is None:
            raise ValueError("Text vectorizer not initialized. Call fit() first.")
        
        tf.random.set_seed(42)
        
        # Create embedding layer
        embedding = layers.Embedding(
            input_dim=self.max_vocab_length,
            output_dim=self.embedding_dim,
            embeddings_initializer="uniform",
            input_length=self.max_length,
            name="embedding"
        )
        
        # Build model
        inputs = layers.Input(shape=(1,), dtype="string")
        x = self.text_vectorizer(inputs)
        x = embedding(x)
        x = layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation="relu")(x)
        x = layers.GlobalMaxPool1D()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        self.model = Model(inputs, outputs, name="Conv1DClassifier")
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"]
        )
        
        return self
    
    def fit(
        self, 
        train_texts: np.ndarray, 
        train_labels: np.ndarray,
        val_texts: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 5,
        batch_size: int = 32
    ) -> 'Conv1DClassifier':
        """
        Fit the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Self for chaining
        """
        from disaster_nlp.features import TensorFlowTextVectorizer
        
        # Initialize text vectorizer
        self.text_vectorizer = TensorFlowTextVectorizer(
            self.max_vocab_length,
            self.max_length
        )
        self.text_vectorizer.adapt(train_texts)
        
        # Build model
        self.build_model()
        
        # Train
        val_data = (val_texts, val_labels) if val_texts is not None else None
        self.history = self.model.fit(
            train_texts,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            verbose=1
        )
        
        return self
    
    def predict(self, texts: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        probs = self.model.predict(texts)
        return (probs > 0.5).astype(int).squeeze()
    
    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(texts).squeeze()


class USETransferLearningClassifier:
    """Transfer Learning classifier using Universal Sentence Encoder."""
    
    def __init__(self, 
                 hub_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4",
                 dense_units: List[int] = None,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.001):
        """
        Initialize USE Transfer Learning classifier.
        
        Args:
            hub_url: TensorFlow Hub URL for USE model
            dense_units: List of dense layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.hub_url = hub_url
        self.dense_units = dense_units or [64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self) -> 'USETransferLearningClassifier':
        """Build the model with USE layer."""
        tf.random.set_seed(42)
        
        # Load USE layer
        sentence_encoder_layer = hub.KerasLayer(
            self.hub_url,
            input_shape=[],
            dtype=tf.string,
            trainable=False,
            name="USE"
        )
        
        # Build model
        inputs = layers.Input(shape=[], dtype=tf.string)
        x = sentence_encoder_layer(inputs)
        
        # Add dense layers
        for i, units in enumerate(self.dense_units[:-1]):
            x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        if len(self.dense_units) > 0:
            x = layers.Dense(self.dense_units[-1], activation="relu")(x)
        
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        self.model = Model(inputs, outputs, name="USEClassifier")
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        
        return self
    
    def fit(
        self,
        train_texts: np.ndarray,
        train_labels: np.ndarray,
        val_texts: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping: bool = True,
        patience: int = 3
    ) -> 'USETransferLearningClassifier':
        """
        Fit the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Self for chaining
        """
        self.build_model()
        
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ))
        
        val_data = (val_texts, val_labels) if val_texts is not None else None
        
        self.history = self.model.fit(
            train_texts,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, texts: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        probs = self.model.predict(texts)
        return (probs > 0.5).astype(int).squeeze()
    
    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(texts).squeeze()
    
    def save(self, filepath: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.save(filepath)
    
    def load(self, filepath: str) -> 'USETransferLearningClassifier':
        """Load model."""
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        return self


class TransformerClassifier:
    """Transformer-based classifier using HuggingFace Transformers (DistilBERT, RoBERTa, etc.)."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        learning_rate: float = 2e-5,
        device: str = "cuda"
    ):
        """
        Initialize Transformer classifier.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            learning_rate: Learning rate for training
            device: "cuda" or "cpu"
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification
        )
        
        # Move model to device
        import torch
        self.model.to(device)
        
        self.history = None
        self.trained = False
    
    def fit(
        self,
        train_texts: Union[List[str], np.ndarray],
        train_labels: np.ndarray,
        val_texts: Optional[Union[List[str], np.ndarray]] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 3,
        batch_size: int = 32,
        warmup_steps: int = 500
    ) -> 'TransformerClassifier':
        """
        Fine-tune the transformer model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            warmup_steps: Warmup steps for learning rate
            
        Returns:
            Self for chaining
        """
        from transformers import get_linear_schedule_with_warmup
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        # Tokenize texts
        print("Tokenizing training texts...")
        train_encodings = self.tokenizer(
            list(train_texts) if isinstance(train_texts, np.ndarray) else train_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create dataset
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            torch.tensor(train_labels, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        training_losses = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if (step + 1) % 100 == 0:
                    print(f"  Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            training_losses.append(avg_loss)
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # Validation
            if val_texts is not None and val_labels is not None:
                val_loss = self._evaluate(val_texts, val_labels, batch_size)
                print(f"  Validation Loss: {val_loss:.4f}")
        
        self.trained = True
        self.history = {"training_loss": training_losses}
        
        return self
    
    def _evaluate(
        self,
        texts: Union[List[str], np.ndarray],
        labels: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """Evaluate model on validation set."""
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        # Tokenize texts
        encodings = self.tokenizer(
            list(texts) if isinstance(texts, np.ndarray) else texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long)
        )
        
        loader = DataLoader(dataset, batch_size=batch_size)
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, batch_labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            texts: Input texts
            
        Returns:
            Predicted labels
        """
        probs = self.predict_proba(texts)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Input texts
            
        Returns:
            Prediction probabilities for positive class
        """
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Tokenize texts
        encodings = self.tokenizer(
            list(texts) if isinstance(texts, np.ndarray) else texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"]
        )
        
        loader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        all_logits = []
        
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits.cpu().numpy()
                all_logits.append(logits)
        
        all_logits = np.vstack(all_logits)
        
        # Convert logits to probabilities using softmax
        from scipy.special import softmax
        probs = softmax(all_logits, axis=1)[:, 1]  # Probability of positive class
        
        return probs
    
    def save(self, filepath: str) -> None:
        """Save model and tokenizer."""
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
    
    def load(self, filepath: str) -> 'TransformerClassifier':
        """Load model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        self.model.to(self.device)
        self.trained = True
        
        return self

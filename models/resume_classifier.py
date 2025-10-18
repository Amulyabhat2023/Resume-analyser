"""
Resume Classification Models

This module provides machine learning resume category classification using traditional ML and can be extended to deep learning.
Supported models: SVM, Random Forest, Naive Bayes, Logistic Regression, Voting Ensemble
"""

import logging
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ResumeClassifier:
    def __init__(self, model_type='ensemble', model_path='trained_models/'):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        os.makedirs(model_path, exist_ok=True)
        self.model_file = os.path.join(model_path, f'{self.model_type}_resume_classifier.pkl')

    def prepare_traditional_ml_model(self):
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        label_encoder = LabelEncoder()
        if self.model_type == 'svm':
            base_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        elif self.model_type == 'rf':
            base_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        elif self.model_type == 'nb':
            base_model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'ensemble':
            svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            nb_model = MultinomialNB(alpha=1.0)
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            base_model = VotingClassifier(estimators=[
                ('svm', svm_model), ('rf', rf_model), ('nb', nb_model), ('lr', lr_model)
            ], voting='soft')
        else:
            base_model = RandomForestClassifier(n_estimators=200, random_state=42)
        # Sklearn Pipeline: TF-IDF + classifier
        model = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', base_model)
        ])
        return model, label_encoder

    def train_model(self, texts, labels, test_size=0.15, validation_size=0.15):
        model, label_encoder = self.prepare_traditional_ml_model()
        encoded_labels = label_encoder.fit_transform(labels)

        # 70/15/15 split for train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        logger.info(f"Validation Accuracy: {val_acc}")

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        logger.info(f"Test Accuracy: {test_acc}")

        # Store for inference
        self.model = model
        self.label_encoder = label_encoder

        # Save
        self.save_model()

    def predict_category(self, text):
        if self.model is None or self.label_encoder is None:
            raise ValueError("Model not trained or loaded")
        prediction = self.model.predict([text])[0]          # pipeline handles vectorization
        category = self.label_encoder.inverse_transform([prediction])[0]
        return {'category': category}

    def save_model(self):
        model_obj = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_obj, f)
        logger.info(f"Model saved to {self.model_file}")

    def load_model(self):
        if not os.path.exists(self.model_file):
            logger.warning(f"Model file {self.model_file} not found.")
            return False
        with open("trained_models/ensemble_resume_classifier.pkl", 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']          # pipeline: vectorizer + classifier
            self.label_encoder = data['label_encoder']
        logger.info("Model loaded successfully")
        return True

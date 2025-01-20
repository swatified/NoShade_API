import os
import json
import numpy as np
import scipy.sparse as sp
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from django.conf import settings
from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid


class ToxicityAnalyzer:
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'model_artifacts')

        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'logistic_model.joblib')
        self.vectorizer_path = os.path.join(
            self.model_dir, 'tfidf_vectorizer.joblib')
        self.sentiment_cache_path = os.path.join(
            self.model_dir, 'sentiment_cache.json')

        self.label_columns = ['toxic', 'severe_toxic',
                              'obscene', 'threat', 'insult', 'identity_hate']

        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

        self.sentiment_cache = self._load_sentiment_cache()

        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(
                "Model files not found. Please train the model first.")

    def _load_sentiment_cache(self):
        if os.path.exists(self.sentiment_cache_path):
            with open(self.sentiment_cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_sentiment_cache(self):
        with open(self.sentiment_cache_path, 'w') as f:
            json.dump(self.sentiment_cache, f)

    def clean_text(self, text):
        text = str(text).lower()
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def get_sentiment(self, text):
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]

        result = self.sentiment_analyzer(text, truncation=True, max_length=128)
        sentiment_map = {
            'LABEL_0': [1, 0, 0],  # Negative
            'LABEL_1': [0, 1, 0],  # Neutral
            'LABEL_2': [0, 0, 1]   # Positive
        }
        sentiment = sentiment_map[result[0]['label']]

        self.sentiment_cache[text] = sentiment
        self._save_sentiment_cache()
        return sentiment

    def analyze(self, text):
        cleaned_text = self.clean_text(text)

        # Get features
        X_tfidf = self.vectorizer.transform([cleaned_text])
        sentiment_features = np.array([self.get_sentiment(cleaned_text)])
        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(sentiment_features)])

        # Get predictions
        predictions = [clf.predict(X_combined)[0]
                       for clf in self.model.estimators_]

        # Format sentiment output
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        dominant_sentiment = sentiment_labels[np.argmax(sentiment_features[0])]

        # Get detected labels
        detected_labels = []
        if dominant_sentiment == 'Negative':
            for label, pred in zip(self.label_columns, predictions):
                if pred == 1:
                    detected_labels.append(label.replace('_', ' ').title())

        return {
            'sentiment': dominant_sentiment,
            'toxic_labels': '\n'.join(detected_labels) if detected_labels else "No toxic labels detected"
        }


class CustomUser(AbstractUser):
        email = models.CharField(max_length=100, unique=True)
        USERNAME_FIELD = 'email'
        api_key = models.CharField(max_length=100, null=True, blank=True)
        api_usage = models.IntegerField(default=0)
        subscribed = models.BooleanField(default=False)
        subscription_date = models.DateTimeField(null=True, blank=True)
        subscription_expiry = models.DateTimeField(null=True, blank=True)
        subscription_plan = models.CharField(
            max_length=100, null=True, blank=True)
        company = models.CharField(max_length=100, null=True, blank=True)
        REQUIRED_FIELDS = ['username', 'password']

class APIKey(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='api_keys')
    key = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def save(self, *args, **kwargs):
        if not self.key:
            self.key = f"sk_{'live' if self.name == 'Production' else 'test'}_{uuid.uuid4().hex[:16]}"
            super().save(*args, **kwargs)


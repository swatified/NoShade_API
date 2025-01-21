import os
import json
import re
from transformers import pipeline
import joblib
from django.conf import settings
from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid

class ToxicityAnalyzer:
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'model_artifacts')
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'logistic_model.joblib')
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        self.sentiment_cache_path = os.path.join(self.model_dir, 'sentiment_cache.json')

        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Initialize sentiment analyzer with CPU device
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=-1  # Force CPU
            )
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {str(e)}")
            self.sentiment_analyzer = None

        self.sentiment_cache = self._load_sentiment_cache()

        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")

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
        try:
            if text in self.sentiment_cache:
                return self.sentiment_cache[text]

            if not self.sentiment_analyzer:
                return [0, 1, 0]  # Default to neutral if analyzer isn't available

            result = self.sentiment_analyzer(text, truncation=True, max_length=128)
            
            # Convert torch tensor to Python list
            label = result[0]['label']
            score = float(result[0]['score'])  # Convert to Python float

            sentiment_map = {
                'LABEL_0': [1, 0, 0],  # Negative
                'LABEL_1': [0, 1, 0],  # Neutral
                'LABEL_2': [0, 0, 1]   # Positive
            }
            sentiment = sentiment_map.get(label, [0, 1, 0])  # Default to neutral if unknown label

            self.sentiment_cache[text] = sentiment
            self._save_sentiment_cache()
            return sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return [0, 1, 0]  # Default to neutral on error

    def analyze(self, text):
        try:
            cleaned_text = self.clean_text(text)
            
            # Get TFIDF features
            X_tfidf = self.vectorizer.transform([cleaned_text])
            
            # Get sentiment features without using numpy
            sentiment_features = self.get_sentiment(cleaned_text)
            
            # Make predictions
            predictions = []
            for clf in self.model.estimators_:
                # Combine features
                combined_features = []
                for i in range(X_tfidf.shape[1]):
                    combined_features.append(X_tfidf[0, i])
                combined_features.extend(sentiment_features)
                
                pred = clf.predict([combined_features])[0]
                predictions.append(pred)

            # Determine sentiment
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            max_sentiment_idx = sentiment_features.index(max(sentiment_features))
            dominant_sentiment = sentiment_labels[max_sentiment_idx]

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
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {
                'sentiment': 'Error',
                'toxic_labels': f"Analysis failed: {str(e)}"
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


import os
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import joblib
from django.conf import settings
from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
import scipy.sparse as sp
from azure.storage.blob import BlobServiceClient
import tempfile

class ToxicityAnalyzer:
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'model_artifacts')
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'logistic_model.joblib')
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        self.sentiment_cache_path = os.path.join(self.model_dir, 'sentiment_cache.json')
        
        # Azure Storage settings
        self.connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = "model-artifacts"
        
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Download and initialize sentiment model
        try:
            self._download_sentiment_model()
            print("Loading sentiment model...")
            self.sentiment_analyzer = pipeline(
                task="sentiment-analysis",
                model=os.path.join(self.model_dir, 'sentiment_model'),
                device="cpu",
                local_files_only=True
            )
        except Exception as e:
            print(f"Error loading sentiment model: {str(e)}")
            self.sentiment_analyzer = None

        self.sentiment_cache = self._load_sentiment_cache()

        # Load toxicity models
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")

    def _download_sentiment_model(self):
        """Download sentiment model files from Azure Blob Storage"""
        model_path = os.path.join(self.model_dir, 'sentiment_model')
        os.makedirs(model_path, exist_ok=True)
        
        # List all blobs in the sentiment_model folder
        blobs = self.container_client.list_blobs(name_starts_with='sentiment_model/')
        
        for blob in blobs:
            # Get local file path
            local_path = os.path.join(self.model_dir, blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download if not exists
            if not os.path.exists(local_path):
                print(f"Downloading {blob.name}...")
                blob_client = self.container_client.get_blob_client(blob.name)
                with open(local_path, "wb") as file:
                    data = blob_client.download_blob()
                    file.write(data.readall())

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
                print("Sentiment analyzer not available, returning neutral")
                return [0, 1, 0]  # Default to neutral if analyzer isn't available

            result = self.sentiment_analyzer(text, truncation=True, max_length=128)
            
            # Convert torch tensor to Python list
            label = result[0]['label']
            score = float(result[0]['score'])

            sentiment_map = {
                'LABEL_0': [1, 0, 0],  # Negative
                'LABEL_1': [0, 1, 0],  # Neutral
                'LABEL_2': [0, 0, 1]   # Positive
            }
            sentiment = sentiment_map.get(label, [0, 1, 0])

            self.sentiment_cache[text] = sentiment
            self._save_sentiment_cache()
            return sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return [0, 1, 0]

    def analyze(self, text):
        try:
            cleaned_text = self.clean_text(text)
            print(f"Analyzing text: {cleaned_text}")
            
            X_tfidf = self.vectorizer.transform([cleaned_text])
            print(f"TFIDF shape: {X_tfidf.shape}")
            
            sentiment_features = self.get_sentiment(cleaned_text)
            print(f"Sentiment features: {sentiment_features}")
            
            sentiment_matrix = sp.csr_matrix([sentiment_features])
            
            X_combined = sp.hstack([X_tfidf, sentiment_matrix])
            print(f"Combined features shape: {X_combined.shape}")
            
            predictions = []
            for clf in self.model.estimators_:
                pred = clf.predict(X_combined)[0]
                predictions.append(pred)
            print(f"Raw predictions: {predictions}")
                
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            max_sentiment_value = max(sentiment_features)
            max_sentiment_idx = sentiment_features.index(max_sentiment_value)
            dominant_sentiment = sentiment_labels[max_sentiment_idx]
            
            detected_labels = []
            for label, pred in zip(self.label_columns, predictions):
                if pred == 1:
                    detected_labels.append(label.replace('_', ' ').title())
            
            print(f"Final sentiment: {dominant_sentiment}")
            print(f"Final labels: {detected_labels}")
            
            return {
                'sentiment': dominant_sentiment,
                'toxic_labels': ', '.join(detected_labels) if detected_labels else "No toxic labels detected"
            }
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {
                'sentiment': 'Error',
                'toxic_labels': f"Analysis failed: {str(e)}"
            }

# These classes remain unchanged
class CustomUser(AbstractUser):
    email = models.CharField(max_length=100, unique=True)
    USERNAME_FIELD = 'email'
    api_key = models.CharField(max_length=100, null=True, blank=True)
    api_usage = models.IntegerField(default=0)
    subscribed = models.BooleanField(default=False)
    subscription_date = models.DateTimeField(null=True, blank=True)
    subscription_expiry = models.DateTimeField(null=True, blank=True)
    subscription_plan = models.CharField(max_length=100, null=True, blank=True)
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
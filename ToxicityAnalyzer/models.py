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
import torch

class ToxicityAnalyzer:
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'model_artifacts')
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, 'logistic_model.joblib')
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        self.sentiment_cache_path = os.path.join(self.model_dir, 'sentiment_cache.json')
        self.sentiment_model_path = os.path.join(self.model_dir, 'sentiment_model')
        os.makedirs(self.sentiment_model_path, exist_ok=True)

        # Azure Storage settings
        try:
            if 'AZURE_STORAGE_CONNECTION_STRING' not in os.environ:
                print("Azure connection string not found in environment variables, checking for local models")
                self.blob_service_client = None
                self.container_client = None
            else:
                self.connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
                self.container_name = "model-artifacts"
                
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                
                # Check for model files
                required_files = [
                    'config.json', 
                    'pytorch_model.bin', 
                    'tokenizer.json', 
                    'tokenizer_config.json',
                    'vocab.json',
                    'merges.txt'
                ]
                files_missing = not all(os.path.exists(os.path.join(self.sentiment_model_path, f)) 
                                     for f in required_files)
                
                if files_missing:
                    print("Some model files missing, attempting to download from Azure...")
                    self._download_sentiment_model()

        except Exception as e:
            print(f"Error initializing Azure storage: {str(e)}")
            self.blob_service_client = None
            self.container_client = None
        
        # Initialize sentiment analyzer
        try:
            print(f"Loading sentiment model from {self.sentiment_model_path}")
            if os.path.exists(os.path.join(self.sentiment_model_path, 'pytorch_model.bin')):
                print("Found model files, loading sentiment analyzer...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.sentiment_model_path,
                    local_files_only=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.sentiment_model_path,
                    local_files_only=True
                )
                self.sentiment_analyzer = pipeline(
                    task="sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device="cpu"
                )
            else:
                print("No model files found, initializing without sentiment analyzer")
                self.sentiment_analyzer = None
        except Exception as e:
            print(f"Error loading sentiment model: {str(e)}")
            self.sentiment_analyzer = None

        self.sentiment_cache = self._load_sentiment_cache()
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        # Load toxicity models
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")

    def _download_sentiment_model(self):
        if not self.container_client:
            print("Azure Storage not initialized, skipping download")
            return
            
        try:
            print(f"Sentiment model path: {self.sentiment_model_path}")
            print("Listing blobs in sentiment_model folder...")
            blobs = self.container_client.list_blobs(name_starts_with='sentiment_model/')
            
            for blob in blobs:
                local_path = os.path.join(self.model_dir, blob.name)
                print(f"Downloading {blob.name} to {local_path}")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if not os.path.exists(local_path):
                    blob_client = self.container_client.get_blob_client(blob.name)
                    with open(local_path, "wb") as file:
                        data = blob_client.download_blob()
                        file.write(data.readall())
                    print(f"Downloaded: {os.path.getsize(local_path)} bytes")
            
            print("\nVerifying downloaded files:")
            for file in os.listdir(self.sentiment_model_path):
                print(f"- {file}")

        except Exception as e:
            print(f"Error downloading from Azure: {str(e)}")
            raise

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
                print("Using cached sentiment")
                return self.sentiment_cache[text]

            if not self.sentiment_analyzer:
                print("Sentiment analyzer not available, returning neutral")
                return [0, 1, 0]

            print("Performing sentiment analysis...")
            result = self.sentiment_analyzer(text, truncation=True, max_length=128)
            print(f"Raw sentiment result: {result}")
            
            label = result[0]['label']
            score = float(result[0]['score'])
            print(f"Label: {label}, Score: {score}")

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
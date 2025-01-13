import os
import json
import numpy as np
import pandas as pd
from transformers import pipeline
import gradio as gr
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib

# Constants
TEST_SAMPLE_SIZE = 500

class ToxicityAnalyzer:
    def __init__(self, model_dir='model_artifacts'):
        print("Initializing ToxicityAnalyzer...")
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_path = os.path.join(model_dir, 'logistic_model.joblib')
        self.vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        self.sentiment_cache_path = os.path.join(model_dir, 'sentiment_cache.json')
        
        print("Loading sentiment analyzer...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        
        # Initialize components
        self.sentiment_cache = self._load_sentiment_cache()
        self.vectorizer = None
        self.model = None
        
        # Load or train the model
        self._initialize_model()
    
    def _load_sentiment_cache(self):
        if os.path.exists(self.sentiment_cache_path):
            print("Loading sentiment cache...")
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

    def prepare_data(self, df, training=False):
        print(f"Preparing dataset with {len(df)} samples...")
        
        X = df.comment_text.apply(self.clean_text)
        print("Texts cleaned...")
        
        if training:
            print("Creating TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                max_features=10000,  # Reduced for test dataset
                ngram_range=(1, 2),
                strip_accents='unicode',
                min_df=2
            )
            X_tfidf = self.vectorizer.fit_transform(X)
            print("TF-IDF features created...")
            
            print("Analyzing sentiments...")
            sentiment_features = []
            for i, text in enumerate(X):
                sentiment = self.get_sentiment(text)
                sentiment_features.append(sentiment)
                if (i + 1) % 50 == 0:  # Progress every 50 samples
                    print(f"Processed {i + 1}/{len(X)} texts...")
            
            sentiment_features = np.array(sentiment_features)
            print("Sentiment analysis complete.")
            
            # Combine TF-IDF and sentiment features
            X_combined = np.hstack([
                X_tfidf.toarray(),
                sentiment_features
            ])
            
            # Save vectorizer
            print("Saving vectorizer...")
            joblib.dump(self.vectorizer, self.vectorizer_path)
            
            return X_combined
        else:
            X_tfidf = self.vectorizer.transform(X)
            sentiment_features = np.array([self.get_sentiment(text) for text in X])
            return np.hstack([X_tfidf.toarray(), sentiment_features])

    def _initialize_model(self):
        print("Loading data...")
        df = pd.read_csv('data/train.csv').head(TEST_SAMPLE_SIZE)  # Load only 500 samples
        df = df.drop('id', axis=1)  # Drop ID column if it exists
        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            print("Loading existing model and vectorizer...")
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        else:
            print("Training new model...")
            # Prepare features
            y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult']].values
            X_combined = self.prepare_data(df, training=True)
            
            # Train test split
            print("Splitting dataset...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42
            )
            
            # Create and train model
            print("Training logistic regression model...")
            base_classifier = LogisticRegression(
                C=1.0,
                max_iter=100,
                class_weight='balanced',
                n_jobs=-1,
                verbose=1
            )
            
            self.model = MultiOutputClassifier(base_classifier)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            print("Evaluating model...")
            score = self.model.score(X_test, y_test)
            print(f"Model accuracy: {score:.4f}")
            
            # Save model
            print("Saving model...")
            joblib.dump(self.model, self.model_path)
            print("Model saved successfully.")

    def score_comment(self, comment):
        cleaned_text = self.clean_text(comment)
        
        # Get features
        X_tfidf = self.vectorizer.transform([cleaned_text])
        sentiment_features = np.array([self.get_sentiment(cleaned_text)])
        X_combined = np.hstack([X_tfidf.toarray(), sentiment_features])
        
        # Get predictions
        predictions = [clf.predict_proba(X_combined) for clf in self.model.estimators_]
        
        # Format sentiment output
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        dominant_sentiment = sentiment_labels[np.argmax(sentiment_features[0])]
        
        # Get toxicity labels
        label_names = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult']
        detected_labels = []
        
        # Check each classifier's probability
        threshold = 0.5
        for label_name, pred_proba in zip(label_names, predictions):
            if pred_proba[0][1] > threshold:  # Probability of positive class > threshold
                detected_labels.append(label_name)
        
        return dominant_sentiment, "\n".join(detected_labels) if detected_labels else "No toxic labels detected"

def create_interface():
    analyzer = ToxicityAnalyzer()
    
    interface = gr.Interface(
        fn=analyzer.score_comment,
        inputs=gr.Textbox(
            lines=3, 
            placeholder='Type your text here...',
            label="Input Text"
        ),
        outputs=[
            gr.Textbox(label="Sentiment", lines=1),
            gr.Textbox(label="Detected Labels", lines=5)
        ],
        title="Toxicity & Sentiment Analyzer (Test Version - 500 Samples)",
        description="Analysis of text for toxicity and sentiment (Trained on 500 samples)",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css="""
            .gradio-container {background-color: #1f1f1f}
        """
    )
    return interface

if __name__ == "__main__":
    print("Starting Toxicity Analyzer...")
    interface = create_interface()
    interface.launch()
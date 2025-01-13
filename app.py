import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from transformers import pipeline
import gradio as gr
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

class ToxicityAnalyzer:
    def __init__(self, model_dir='model_artifacts'):
        print("Initializing ToxicityAnalyzer...")
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_path = os.path.join(model_dir, 'logistic_model.joblib')
        self.vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        self.sentiment_cache_path = os.path.join(model_dir, 'sentiment_cache.json')
        
        # Label columns in order
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
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
        return sentiment

    def prepare_data(self, df, training=False):
        print(f"\nPreparing dataset with {len(df)} samples...")
        
        # Extract features and labels
        X = df['comment_text'].apply(self.clean_text)
        if training:
            y = df[self.label_columns].values
            print(f"Label distribution:")
            for col in self.label_columns:
                positive_count = df[col].sum()
                print(f"{col}: {positive_count} positive samples ({positive_count/len(df)*100:.2f}%)")
        
        print("\nCreating TF-IDF features...")
        if training:
            self.vectorizer = TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                strip_accents='unicode',
                min_df=5
            )
            X_tfidf = self.vectorizer.fit_transform(X)
        else:
            X_tfidf = self.vectorizer.transform(X)
        
        print("Getting sentiment features...")
        sentiment_features = []
        for text in tqdm(X, desc="Processing sentiments"):
            sentiment = self.get_sentiment(text)
            sentiment_features.append(sentiment)
            
        sentiment_features = np.array(sentiment_features)
        
        # Combine features
        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(sentiment_features)])
        
        if training:
            return X_combined, y
        return X_combined

    def _initialize_model(self):
        print("\nLoading data...")
        df = pd.read_csv('data/train.csv')
        print(f"Dataset size: {len(df)} samples")
        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            print("Loading existing model and vectorizer...")
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        else:
            print("Training new model...")
            X_combined, y = self.prepare_data(df, training=True)
            
            print("\nSplitting dataset...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42, 
                stratify=y[:, 0]  # Stratify on toxic label
            )
            
            print("\nTraining models for each label...")
            estimators = []
            for i, label in enumerate(tqdm(self.label_columns)):
                print(f"\nTraining classifier for {label}...")
                clf = LogisticRegression(
                    C=1.0,
                    max_iter=200,
                    class_weight='balanced',
                    verbose=1
                )
                clf.fit(X_train, y_train[:, i])
                
                # Evaluate on test set
                score = clf.score(X_test, y_test[:, i])
                print(f"{label} classifier accuracy: {score:.4f}")
                estimators.append(clf)
            
            self.model = MultiOutputClassifier(estimators)
            self.model.estimators_ = estimators
            
            print("\nSaving models...")
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)

    def score_comment(self, comment):
        cleaned_text = self.clean_text(comment)
        
        # Get features
        X_tfidf = self.vectorizer.transform([cleaned_text])
        sentiment_features = np.array([self.get_sentiment(cleaned_text)])
        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(sentiment_features)])
        
        # Get predictions
        predictions = [clf.predict(X_combined)[0] for clf in self.model.estimators_]
        
        # Format sentiment output
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        dominant_sentiment = sentiment_labels[np.argmax(sentiment_features[0])]
        
        # Get detected labels
        if dominant_sentiment == 'Negative':
            predictions = [clf.predict(X_combined)[0] for clf in self.model.estimators_]
            detected_labels = []
            for label, pred in zip(self.label_columns, predictions):
                if pred == 1:
                    detected_labels.append(label.replace('_', ' ').title())
            return dominant_sentiment, "\n".join(detected_labels) if detected_labels else "No toxic labels detected"
        else:
            return dominant_sentiment, "No toxic labels detected"

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
            gr.Textbox(label="Detected Labels", lines=6)
        ],
        title="Toxicity & Sentiment Analyzer",
        description="Analysis of text for toxicity and sentiment",
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
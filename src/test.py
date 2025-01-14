import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from tqdm import tqdm
import re
import scipy.sparse as sp

def clean_text(text):
    """Clean text by removing special characters and converting to lowercase"""
    text = str(text).lower()
    return re.sub(r'[^a-zA-Z\s]', '', text)

def main():
    print("Loading model and vectorizer...")
    model = joblib.load('model_artifacts/logistic_model.joblib')
    vectorizer = joblib.load('model_artifacts/tfidf_vectorizer.joblib')
    
    print("\nLoading test data...")
    test_df = pd.read_csv('data/test.csv')
    print(f"Test dataset size: {len(test_df)} samples")
    
    # Prepare text data
    print("Preparing test data...")
    X_test = pd.Series(test_df['comment_text'].apply(clean_text))
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Add three zero columns for sentiment features
    sentiment_features = np.zeros((X_test_tfidf.shape[0], 3))
    X_combined = sp.hstack([X_test_tfidf, sentiment_features])
    
    print(f"Final feature matrix shape: {X_combined.shape}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_combined)
    
    # Label columns
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create results DataFrame
    results_df = pd.DataFrame(predictions, columns=label_columns)
    results_df['comment_text'] = test_df['comment_text']
    results_df['id'] = test_df['id']
    
    # Save predictions
    print("\nSaving predictions...")
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Save full results
    results_df.to_csv('evaluation_results/test_predictions.csv', index=False)
    
    # Save submission format (only required columns)
    submission_df = results_df[['id'] + label_columns]
    submission_df.to_csv('evaluation_results/submission.csv', index=False)
    
    # Print some statistics
    print("\nPrediction Statistics:")
    for column in label_columns:
        positive_count = results_df[column].sum()
        percentage = (positive_count / len(results_df)) * 100
        print(f"{column}: {positive_count} texts ({percentage:.2f}%)")
    
    # Show some example predictions
    print("\nExample Predictions:")
    examples = results_df.sample(n=10)
    for _, row in examples.iterrows():
        print("\nText:", row['comment_text'][:100] + "..." if len(row['comment_text']) > 100 else row['comment_text'])
        print("Detected Labels:")
        detected = False
        for label in label_columns:
            if row[label] == 1:
                print(f"- {label.replace('_', ' ').title()}")
                detected = True
        if not detected:
            print("- No toxic labels detected")
    
    print("\nEvaluation complete! Check evaluation_results directory for:")
    print("1. test_predictions.csv - Full results with text")
    print("2. submission.csv - Submission format with just predictions")

if __name__ == "__main__":
    main()
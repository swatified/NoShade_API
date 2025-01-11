import os
# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
import gradio as gr
import pickle

# Constants
MAX_FEATURES = 50000
SEQUENCE_LENGTH = 256
BATCH_SIZE = 16
EMBEDDING_DIM = 16

def load_data(data_path='data/train.csv'):
    print("Loading data...")
    df = pd.read_csv(data_path)
    df.drop('id', inplace=True, axis=1)
    return df

def prepare_dataset(df):
    print("Preparing datasets...")
    X = df.comment_text
    y = df[df.columns[2:]].values
    
    # Create and adapt vectorizer
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=SEQUENCE_LENGTH,
        output_mode='int'
    )
    
    text_ds = tf.data.Dataset.from_tensor_slices(X.values).batch(BATCH_SIZE)
    vectorizer.adapt(text_ds)
    
    def vectorize_batch(texts, labels):
        return vectorizer(texts), labels
    
    dataset = tf.data.Dataset.from_tensor_slices((X.values, y))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(vectorize_batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)
    
    train = dataset.take(train_size // BATCH_SIZE)
    val = dataset.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
    test = dataset.skip((train_size + val_size) // BATCH_SIZE)
    
    return train, val, test, vectorizer

def create_model():
    print("Creating model...")
    model = Sequential([
        Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
        Bidirectional(LSTM(16, activation='tanh')),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(5, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def format_results(results, labels):
    """Format prediction results in a readable way"""
    formatted = []
    for label, value in zip(labels, results[0]):
        label_name = label.replace('_', ' ').title()
        result = "Yes" if value > 0.5 else "No"
        formatted.append(f"{label_name}: {result}")
    return "\n".join(formatted)

def main():
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is available")
    else:
        print("No GPU found. Running on CPU.")

    df = load_data()
    model_path = 'toxicity_model'
    vectorizer_path = 'vectorizer.pkl'

    # Check if saved model exists
    if os.path.exists(model_path):
        print("Loading saved model...")
        model = tf.keras.models.load_model(model_path)
        # Always create a fresh vectorizer
        _, _, _, vectorizer = prepare_dataset(df)
        print("Vectorizer initialized...")
    else:
        print("Training new model...")
        train, val, test, vectorizer = prepare_dataset(df)
        model = create_model()
        
        history = model.fit(
            train,
            validation_data=val,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True
                )
            ]
        )
        
        print("Saving model and vectorizer...")
        model.save(model_path)
        
        # Save vectorizer configuration
        config = vectorizer.get_config()
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(config, f)

    # Create Gradio interface
    def score_comment(comment):
        vectorized_comment = vectorizer([comment])
        results = model.predict(vectorized_comment)
        return format_results(results, df.columns[2:])

    interface = gr.Interface(
        fn=score_comment,
        inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
        outputs=gr.Textbox(label="Toxicity Analysis"),
        title="Toxicity Detector",
        description="Enter a comment to check for toxic content"
    )
    interface.launch()

if __name__ == "__main__":
    main()
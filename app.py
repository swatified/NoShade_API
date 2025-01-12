import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured")
        print(f"Found GPU: {gpus[0].device_type}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

print("Tensorflow devices:", tf.config.list_physical_devices())

import pandas as pd
import numpy as np
from transformers import pipeline
from tensorflow.keras.layers import TextVectorization, Concatenate, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
import gradio as gr
import re

# Constants
MAX_FEATURES = 50000
SEQUENCE_LENGTH = 256
BATCH_SIZE = 32  # Increased for GPU
EMBEDDING_DIM = 64  # Increased for better representation

# Load sentiment analysis pipeline
print("Loading RoBERTa model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0 if tf.test.gpu_device_name() else -1  # Use GPU if available
)

def get_roberta_sentiment(text):
    """Get sentiment features using RoBERTa"""
    try:
        result = sentiment_analyzer(text, truncation=True, max_length=128)
        sentiment_map = {
            'LABEL_0': [1, 0, 0],  # Negative
            'LABEL_1': [0, 1, 0],  # Neutral
            'LABEL_2': [0, 0, 1]   # Positive
        }
        return sentiment_map[result[0]['label']]
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return [0.33, 0.33, 0.34]

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_data(data_path='data/train.csv'):
    print("Loading data...")
    df = pd.read_csv(data_path)
    df.drop('id', inplace=True, axis=1)
    return df

def prepare_dataset(df):
    print("Preparing datasets...")
    X = df.comment_text
    y = df[df.columns[2:]].values
    
    X_clean = X.apply(clean_text)
    
    print("Analyzing sentiments...")
    sentiment_features = []
    for i in range(0, len(X_clean), BATCH_SIZE):
        batch = X_clean[i:i + BATCH_SIZE]
        batch_sentiments = sentiment_analyzer(batch.tolist(), truncation=True, max_length=128, batch_size=BATCH_SIZE)
        numerical_sentiments = []
        for sent in batch_sentiments:
            sentiment_map = {
                'LABEL_0': [1, 0, 0],
                'LABEL_1': [0, 1, 0],
                'LABEL_2': [0, 0, 1]
            }
            numerical_sentiments.append(sentiment_map[sent['label']])
        sentiment_features.extend(numerical_sentiments)
        if i % 1000 == 0:
            print(f"Processed {i}/{len(X_clean)} texts...")
    
    sentiment_features = np.array(sentiment_features)
    
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=SEQUENCE_LENGTH,
        output_mode='int'
    )
    
    text_ds = tf.data.Dataset.from_tensor_slices(X_clean.values).batch(BATCH_SIZE)
    vectorizer.adapt(text_ds)
    
    def prepare_features(texts, sentiments, labels):
        return {
            'text_input': vectorizer(texts), 
            'sentiment_input': sentiments
        }, labels
    
    dataset = tf.data.Dataset.from_tensor_slices((X_clean.values, sentiment_features, y))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(prepare_features)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)
    
    train = dataset.take(train_size // BATCH_SIZE)
    val = dataset.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
    test = dataset.skip((train_size + val_size) // BATCH_SIZE)
    
    return train, val, test, vectorizer

def create_model():
    print("Creating enhanced model...")
    
    # Text input branch
    text_input = Input(shape=(SEQUENCE_LENGTH,), name='text_input')
    embedding = Embedding(MAX_FEATURES + 1, EMBEDDING_DIM)(text_input)
    
    # Multiple LSTM layers
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    lstm1 = BatchNormalization()(lstm1)
    
    lstm2 = Bidirectional(LSTM(32))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    # Sentiment input branch
    sentiment_input = Input(shape=(3,), name='sentiment_input')
    sentiment_dense = Dense(32, activation='relu')(sentiment_input)
    sentiment_dense = BatchNormalization()(sentiment_dense)
    
    # Combine features
    combined = Concatenate()([lstm2, sentiment_dense])
    
    # Dense layers
    dense = Dense(256, activation='relu')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    
    dense = Dense(64, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    
    outputs = Dense(5, activation='sigmoid')(dense)
    
    model = Model(inputs=[text_input, sentiment_input], outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def main():
    # Print GPU information
    print("\nGPU Information:")
    print("GPU Available:", bool(tf.config.list_physical_devices('GPU')))
    print("GPU Device:", tf.test.gpu_device_name())

    df = load_data()
    model_path = 'toxicity_model'

    if os.path.exists(model_path):
        print("Loading saved model...")
        model = tf.keras.models.load_model(model_path)
        _, _, _, vectorizer = prepare_dataset(df)
        print("Vectorizer initialized...")
    else:
        print("Training new model...")
        train, val, test, vectorizer = prepare_dataset(df)
        model = create_model()
        
        history = model.fit(
            train,
            validation_data=val,
            epochs=25,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=4,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        )
        
        print("Saving model...")
        model.save(model_path)

    def score_comment(comment):
        cleaned_text = clean_text(comment)
        sentiment_features = get_roberta_sentiment(cleaned_text)
        vectorized_text = vectorizer([cleaned_text])
        
        results = model.predict({
            'text_input': vectorized_text,
            'sentiment_input': np.array([sentiment_features])
        })
        
        # Format sentiment output
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment_dict = {label: score for label, score in zip(sentiment_labels, sentiment_features)}
        dominant_sentiment = max(sentiment_dict.items(), key=lambda x: x[1])
        sentiment_output = f"{dominant_sentiment[0]}: {dominant_sentiment[1]*100:.1f}%"
        
        # Only show toxicity labels for negative sentiment
        toxicity_labels = {}
        if sentiment_features[0] > 0.5:  # If negative sentiment
            for label, value in zip(df.columns[2:], results[0]):
                if value > 0.2:  # Only include if confidence exceeds threshold
                    label_name = label.replace('_', ' ').title()
                    confidence = value * sentiment_features[0] * 100  # Weight by negative sentiment
                    toxicity_labels[label_name] = confidence
        
        return sentiment_output, toxicity_labels

    interface = gr.Interface(
        fn=score_comment,
        inputs=gr.Textbox(
            lines=3, 
            placeholder='Type your text here...',
            label="Input Text"
        ),
        outputs=[
            gr.Textbox(label="Sentiment Score", lines=1),
            gr.Label(label="Toxicity Tags", show_label=False, num_top_classes=3)
        ],
        title="Toxicity & Sentiment Analyzer",
        description="Analysis of text for toxicity and sentiment",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css="""
            .gradio-container {background-color: #1f1f1f}
            .label-confidence {display: none}
        """
    )
    interface.launch()

if __name__ == "__main__":
    main()
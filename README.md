# NoShade API: Toxic Comments Analyzer

A machine learning model that analyzes text for sentiment and toxicity. The model first determines if the text is negative, neutral, or positive, and then identifies specific types of toxic content only in negative text.

## Features
- Sentiment Analysis (Positive, Neutral, Negative)
- Toxicity Detection Categories:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
<br/><br/>

## Installation Steps

1. Clone the repository
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages
```bash
pip install transformers torch scikit-learn pandas numpy gradio tqdm joblib
```

### Running the Application
```bash
python app.py
```
The application will start on http://127.0.0.1:7861

### Initial Load
![Startup](https://github.com/user-attachments/assets/5448d79d-464b-4d08-af8f-979e58d06b14)

When you start the application, it loads:
- The pre-trained sentiment analyzer
- Dataset of 159,571 samples
- Trained model and vectorizer
<br/><br/>


# Usage Examples ~

## 1. **Neutral Text**
![Neutral Example](https://github.com/user-attachments/assets/a40c515b-8d64-4751-8c64-d2f309c01812)
Simple greeting "Hey" is classified as neutral sentiment with no toxic labels.
<br/><br/>

## 2. **Positive Text**
![Positive Example](https://github.com/user-attachments/assets/c9b00506-f195-4062-82c3-cb9584345441)
"I have never seen something so funny" is classified as positive sentiment with no toxic labels.
<br/><br/>

## 3. **Negative Text with Toxicity**
![Negative Example 1](https://github.com/user-attachments/assets/f62daeb1-81fa-432d-994b-9c48201661d3)
"He is so stupid" triggers:
- Negative sentiment
- Multiple toxicity labels: Toxic, Obscene, Insult, Identity Hate
<br/>

## 4. **Another Negative Example**
![Negative Example 2](https://github.com/user-attachments/assets/85ab63a9-da7b-47be-a347-6d0de3ce7a89)
"Her mom cant cook" triggers:
- Negative sentiment
- Multiple toxicity labels: Toxic, Obscene, Insult
<br/>

## How the Model Works

1. **Sentiment Analysis**
   - Uses RoBERTa-based model for sentiment classification
   - Categorizes text as Positive, Neutral, or Negative

2. **Toxicity Detection**
   - Only activates for negative sentiment
   - Uses logistic regression classifiers
   - Trained on 159,571 labeled examples
   - Multiple independent classifiers for each toxicity category

3. **Feature Processing**
   - Text cleaning and normalization
   - TF-IDF vectorization
   - Combines text features with sentiment analysis
<br/><br/>

## Technical Details
- Backend:Django
- Frontend: Django html
- Model: Multi-label logistic regression with TF-IDF features
- Sentiment Analysis: RoBERTa-based transformer model
<br/><br/>

![Screenshot 2025-01-11 134424](https://github.com/user-attachments/assets/950167a3-dcdd-419f-94cc-301be950a1c8)

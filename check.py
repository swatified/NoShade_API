import os

def check_files():
    model_exists = os.path.exists('toxicity_model')
    vectorizer_exists = os.path.exists('vectorizer.pkl')
    
    print(f"Model directory exists: {model_exists}")
    print(f"Vectorizer file exists: {vectorizer_exists}")
    
    if model_exists:
        print("\nContents of toxicity_model directory:")
        print(os.listdir('toxicity_model'))

if __name__ == "__main__":
    check_files()
import nltk
import os

def setup_nltk():
    try:
        # Download required datasets
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data setup complete!")
        return True
    except Exception as e:
        print(f"Error setting up NLTK: {e}")
        return False

if __name__ == "__main__":
    setup_nltk()
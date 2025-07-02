import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os

# Improved NLTK data download function
def download_nltk_data():
    required_data = {
        'tokenizers': ['punkt'],
        'corpora': ['wordnet'],
        'taggers': ['averaged_perceptron_tagger']  # Often needed for tokenization
    }
    
    print("Checking NLTK data...")
    for category, datasets in required_data.items():
        for dataset in datasets:
            try:
                nltk.data.find(f'{category}/{dataset}')
                print(f"✓ {dataset} already downloaded")
            except LookupError:
                print(f"Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
                print(f"✓ {dataset} downloaded")

# Download required NLTK data first
download_nltk_data()

# Rest of your existing train.py code continues...
lemmatizer = WordNetLemmatizer()
# ... [keep all the rest of your existing code]

# Load and process the intents file
intents_file = os.path.join('training_data', 'intents.json')
intents = json.load(open(intents_file, encoding='utf-8'))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes to pickle files
os.makedirs('models', exist_ok=True)
pickle.dump(words, open(os.path.join('models', 'words_v2.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join('models', 'classes_v2.pkl'), 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize bag of words
    bag = []
    # List of tokenized words for the pattern
    word_patterns = doc[0]
    # Lemmatize each word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words array
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Output is '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle features and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model with SGD optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
print("Starting model training...")
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save(os.path.join('models', 'chatbot_model_v2.h5'))
print(f"Model training complete. Files saved to {os.path.abspath('models')} directory.")

# Print training summary
print("\nTraining Summary:")
print(f"- Total words: {len(words)}")
print(f"- Total classes: {len(classes)}")
print(f"- Training samples: {len(train_x)}")
print(f"- Model accuracy: {hist.history['accuracy'][-1]:.2f}")

# Add RAG index building after model training
from rag_utils import RAGSystem


print("\nBuilding RAG index...")
rag = RAGSystem()

# Sample documents - in a real application, you would load your knowledge base here
documents = [
    "Our company was founded in 2010.",
    "We specialize in AI and machine learning solutions.",
    "Our office is located at 123 Main Street, Tech City.",
    "Customer support hours are 9am to 5pm Monday through Friday.",
    "Our products include chatbots, recommendation systems, and computer vision solutions."
]

rag.build_index(documents)
rag.save_index()
print("RAG index built successfully.")

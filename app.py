from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os
from rag_utils import RAGSystem

# Add this near the top after other imports
def initialize_rag_system():
    rag = RAGSystem()
    try:
        rag.load_index()
        print("RAG index loaded successfully")
        return rag
    except FileNotFoundError:
        print("No RAG index found - some functionality will be limited")
        return rag
    except Exception as e:
        print(f"Error loading RAG index: {e}")
        return None

# Then modify your existing RAG initialization:
rag = initialize_rag_system()

app = Flask(__name__)
app.static_folder = 'static'

# Load the model and data files
model = load_model(os.path.join('models', 'chatbot_model_v2.h5'))
words = pickle.load(open(os.path.join('models', 'words_v2.pkl'), 'rb'))
classes = pickle.load(open(os.path.join('models', 'classes_v2.pkl'), 'rb'))
intents = json.load(open(os.path.join('training_data', 'intents.json'), encoding='utf-8'))

rag = RAGSystem()
try:
    rag.load_index()
except:
    print("No RAG index found. Please build the index first.")

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I didn't understand that. Could you rephrase?"
    
    return result

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    try:
        user_text = request.args.get('msg')
        if not user_text:
            return "Please enter a message"
            
        print(f"Received message: {user_text}")  # Debug logging
        
        ints = predict_class(user_text, model)
        res = get_response(ints, intents)
        
        print(f"Returning response: {res}")  # Debug logging
        return str(res)
        
    except Exception as e:
        print(f"Error in get_bot_response: {e}")
        return "Sorry, I encountered an error processing your request."

# Add a new endpoint to rebuild the RAG index
@app.route('/build_rag', methods=['POST'])
def build_rag():
    if not rag:
        return jsonify({"status": "error", "message": "RAG system not initialized"}), 500
    
    try:
        # In a real app, you would load documents from a database or files
        documents = [
            "Our company was founded in 2010.",
            "We specialize in AI and machine learning solutions.",
            "Our office is located at 123 Main Street, Tech City.",
            "Customer support hours are 9am to 5pm Monday through Friday.",
            "Our products include chatbots, recommendation systems, and computer vision solutions."
        ]
        rag.build_index(documents)
        rag.save_index()
        return jsonify({"status": "success", "message": "RAG index rebuilt"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
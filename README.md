# AI Chatbot with TensorFlow and RAG

This project is an intelligent AI-powered chatbot built with TensorFlow. It uses a custom-trained model on intent-based data and integrates a simple Retrieval-Augmented Generation (RAG) system to enhance its contextual understanding and responses.

---

## 🧠 Features

- Trained on intents defined in `intents.json`.
- Uses a neural network (`chatbot_model_v2.h5`) for classification.
- Retrieval-Augmented Generation (RAG) mechanism using precomputed index (`rag_index.pkl`).
- Interactive web interface (HTML + CSS + JS).
- Simple Flask backend (`app.py`) to handle chat logic.

---

## 📁 Project Structure
```bash
AI-Chatbot-with-Tensorflow-master - RAG/
│
├── app.py # Main Flask app
├── rag_utils.py # Functions for RAG logic
├── set_up.py # Preprocessing and setup utilities
├── train.py # Training script for the chatbot model
├── requirements.txt # Python dependencies
│
├── training_data/
│ └── intents.json # Training intents for chatbot
│
├── models/
│ ├── chatbot_model_v2.h5 # Trained neural network model
│ ├── words_v2.pkl # Vocabulary from training
│ ├── classes_v2.pkl # Intent classes
│ └── rag_index.pkl # Precomputed RAG vector index
│
├── templates/
│ └── home.html # Frontend interface (HTML)
│
├── static/
│ ├── css/
│ │ └── style.css # Styling for the interface
│ ├── js/
│ │ └── script.js # Frontend logic
│ └── images/
│ └── favicon.ico # Website icon
│
├── venv/ # Virtual environment (not tracked by Git)
└── pycache/ # Compiled Python cache
```


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
Copier
Modifier
git clone https://github.com/your-username/AI-Chatbot-with-Tensorflow-RAG.git
cd AI-Chatbot-with-Tensorflow-RAG
```
### 2. Set Up Virtual Environment
```bash
Copier
Modifier

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
### 3. Install Requirements
```bash
Copier
Modifier
pip install -r requirements.txt
```
### 4. Train the Model (Optional)
```bash
Copier
Modifier
python train.py
```
### 5. Run the Application
```bash
Copier
Modifier
python app.py
# Then go to http://127.0.0.1:5000 in your browser.
```

📊 Data & Model
The chatbot is trained on intent patterns and responses defined in intents.json.

The training script (train.py) builds the model and saves it in the models/ directory.

RAG uses rag_utils.py to enhance response accuracy by retrieving relevant context.

🛠 Technologies Used
Python 3.x

TensorFlow / Keras

Flask

Scikit-learn / NumPy / Pickle

HTML / CSS / JavaScript

📌 Notes
rag_index.pkl must be pre-generated with relevant documents or embeddings.

Ensure that words_v2.pkl, classes_v2.pkl, and chatbot_model_v2.h5 are up to date if retraining.

The venv/ folder is ignored in version control and should be created locally.

📧 Contact
Created by Ismail Jirari
📧 Email: ismailjirari5@email.com
🌐 GitHub: ismailjirari

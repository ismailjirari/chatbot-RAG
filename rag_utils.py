import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List

class RAGSystem:
    def __init__(self, max_features: int = 5000):
        """
        Initialize RAG system with TF-IDF vectorizer
        Args:
            max_features: Maximum number of vocabulary features to keep
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.embeddings = None
        self.documents = []
        self.idf = None
        
    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents"""
        if not documents:
            raise ValueError("Documents list cannot be empty")
            
        self.documents = documents
        self.embeddings = self.vectorizer.fit_transform(documents)
        self.idf = self.vectorizer.idf_
            
    def save_index(self, path: str = 'models/rag_index') -> None:
        """Save index to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'vocab': self.vectorizer.vocabulary_,
                'idf': self.idf,
                'max_features': self.vectorizer.max_features
            }, f)
            
    def load_index(self, path: str = 'models/rag_index') -> None:
        """Load index from disk"""
        with open(path + '.pkl', 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.vectorizer = TfidfVectorizer(
                vocabulary=data['vocab'],
                max_features=data.get('max_features', 5000),
                stop_words='english'
            )
            # Manually set IDF values after initialization
            self.vectorizer.idf_ = data['idf']
            self.embeddings = self.vectorizer.transform(self.documents)
            
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top k most relevant documents"""
        if not hasattr(self, 'embeddings') or self.embeddings is None:
            raise RuntimeError("Index not built or loaded")
            
        query_embedding = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]
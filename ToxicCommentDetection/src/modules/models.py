from .llm_embedder import LLMEmbedder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

class ToxicityModel:
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        
        if self.use_llm:
            self.embedder = LLMEmbedder()
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=15000)
        
        # Cambiar a RandomForest con ajuste de clase
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',  # Penalizar más los falsos positivos
            n_jobs=-1
        )
    
    def train(self, texts, labels):
        if self.use_llm:
            if isinstance(texts, pd.Series):
                texts = texts.tolist()
            print("🔄 Generando embeddings con LLM...")
            X = self.embedder.embed(texts)
            print(f"📊 Embeddings generados: {X.shape[0]} muestras, {X.shape[1]} dimensiones")
        else:
            X = self.vectorizer.fit_transform(texts)
        
        # Balancear clases adicionalmente
        print("🔁 Balanceando clases con SMOTE...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, labels)
        
        print("🧠 Entrenando clasificador...")
        self.clf.fit(X_res, y_res)
        print("✅ Modelo entrenado")
    
    def predict(self, text):
        """Devuelve la probabilidad de toxicidad (sin umbral ajustado)"""
        if isinstance(text, str):
            text = [text]
        
        if self.use_llm:
            X = self.embedder.embed(text)
        else:
            X = self.vectorizer.transform(text)
        
        # Devuelve probabilidad (sin aplicar umbral)
        return self.clf.predict_proba(X)[0][1]
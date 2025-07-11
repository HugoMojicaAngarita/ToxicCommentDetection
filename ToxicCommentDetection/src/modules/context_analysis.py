from textblob import TextBlob
import re

class ContextAnalyzer:
    def __init__(self):
        # Patrones para sarcasmo e irnoia
        self.sarcasm_patterns = [
            re.compile(r"\b(qué|vaya)\s+(solución|idea|trabajo)\s+(tan|más)\s+(brillante|genial|fantástica)\b", re.IGNORECASE),
            re.compile(r"\b(claro\s+que\s+sí|como\s+no)\b", re.IGNORECASE),
            re.compile(r"\b(genial|fantástico|maravilloso)\s+(como\s+siempre)\b", re.IGNORECASE)
        ]
        
        # Patrones de negación
        self.negation_pattern = re.compile(r"\b(no|nunca|jamás|tampoco)\s+(\w+)", re.IGNORECASE)
        
        # Patrones de exclamación e interrogación
        self.exclamation_pattern = re.compile(r"!+")
        self.question_pattern = re.compile(r"\?+")
        
        # Lista de palabras de contexto positivo
        self.positive_words = {
            'bueno', 'excelente', 'gracias', 'aprecio', 'respeto',
            'orgulloso', 'amor', 'apoyo', 'feliz', 'positivo'
        }
    
    def analyze(self, text):
        if not isinstance(text, str):
            return {
                'sentiment': 0,
                'sarcasm_score': 0,
                'negation_score': 0,
                'exclamation_score': 0,
                'question_score': 0,
                'positive_word_count': 0
            }
        
        # Análisis de sentimiento
        sentiment = TextBlob(text).sentiment.polarity
        
        # Detección de sarcasmo
        sarcasm_score = any(pattern.search(text) for pattern in self.sarcasm_patterns)
        
        # Detección de negaciones
        negation_score = len(self.negation_pattern.findall(text))
        
        # Puntuación de exclamación e interrogación
        exclamation_score = len(self.exclamation_pattern.findall(text))
        question_score = len(self.question_pattern.findall(text))
        
        # Conteo de palabras positivas
        text_lower = text.lower()
        positive_word_count = sum(1 for word in self.positive_words if word in text_lower)
        
        return {
            'sentiment': float(sentiment),
            'sarcasm_score': int(sarcasm_score),
            'negation_score': negation_score,
            'exclamation_score': exclamation_score,
            'question_score': question_score,
            'positive_word_count': positive_word_count
        }
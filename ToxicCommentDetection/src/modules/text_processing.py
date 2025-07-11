import re
from nltk.stem import SnowballStemmer

# Clase para limpiar y procesar texto
class TextCleaner:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.toxic_patterns = re.compile(r"\b(idiota|estúpido|incompetente|imbécil|mierda|matar|jodido)\b", re.IGNORECASE)
    
    # Método para limpiar el texto
    # Conserva palabras tóxicas y realiza limpieza básica
    def clean(self, text):
        if not isinstance(text, str):
            return ""
            
        # Conservar palabras tóxicas
        text = self.toxic_patterns.sub(lambda x: f"TOXIC_{x.group().upper()}", text.lower())
        
        # Limpieza básica
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenización y stemming
        tokens = text.split()
        tokens = [self.stemmer.stem(token) for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
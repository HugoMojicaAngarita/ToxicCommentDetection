class IdentityDetector:
    def __init__(self, identity_terms):
        self.identity_terms = identity_terms
    
    # Detecta la presencia de términos de identidad en el texto
    def detect(self, text):
        if not isinstance(text, str):
            return {category: False for category in self.identity_terms}
        
        text_lower = text.lower()
        return {
            category: any(term.lower() in text_lower for term in terms)
            for category, terms in self.identity_terms.items()
        }
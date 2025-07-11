from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos y modelos
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Configuración de procesamiento de texto
TFIDF_MAX_FEATURES = 15000
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_SPACY = False

# Terminos de identidad
IDENTITY_TERMS = {
    'gender': ['mujer', 'hombre', 'transgénero', 'feminista', 'machista'],
    'religion': ['cristiano', 'musulmán', 'judío', 'ateo', 'religión'],
    'race': ['negro', 'blanco', 'asiático', 'latino'],
    'sexual_orientation': ['gay', 'lesbiana', 'homosexual', 'bisexual']
}

# Frases de sarcasmo
SARCASM_PHRASES = [
    "qué solución tan brillante",
    "genial idea",
    "fantástico trabajo",
    "vaya manera de hacerlo"
]

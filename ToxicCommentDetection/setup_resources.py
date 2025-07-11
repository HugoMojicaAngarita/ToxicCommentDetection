import nltk
from textblob.download_corpora import download_all
from transformers import AutoTokenizer, AutoModel
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_essential_resources():
    """Descarga todos los recursos necesarios para NLP y LLMs"""
    logger.info("="*60)
    logger.info("CONFIGURANDO RECURSOS PARA EL PROYECTO FINAL")
    logger.info("="*60)
    
    # 1. Recursos de NLTK
    nltk_resources = [
        'punkt', 'stopwords', 'wordnet',
        'averaged_perceptron_tagger', 'omw-1.4'
    ]
    
    logger.info("\nDescargando recursos de NLTK...")
    for resource in nltk_resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"✅ {resource}")
        except Exception as e:
            logger.error(f"⚠️ Error con {resource}: {str(e)}")
    
    # 2. Recursos de TextBlob
    logger.info("\nDescargando recursos de TextBlob...")
    try:
        download_all()
        logger.info("✅ TextBlob Corpora")
    except Exception as e:
        logger.error(f"⚠️ TextBlob: {str(e)}")
    
    # 3. Modelo LLM (DistilRoBERTa - versión ligera)
    logger.info("\n⏳ Descargando modelo de lenguaje (distilroberta-base)...")
    try:
        # Intento de carga para verificar si ya existe
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
            model = AutoModel.from_pretrained("distilroberta-base")
            logger.info("✅ Modelo ya existe en caché")
        except:
            logger.info("Descargando modelo (≈300MB)...")
            tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
            model = AutoModel.from_pretrained("distilroberta-base")
            logger.info("✅ Modelo descargado")
        
        # Test de funcionamiento rápido
        inputs = tokenizer("Prueba de integración LLM exitosa", return_tensors="pt")
        outputs = model(**inputs)
        logger.info(f"🧪 Test de embeddings: Vector de {outputs.last_hidden_state.size()}")
        
    except Exception as e:
        logger.error(f"❌ Error crítico: {str(e)}")
        logger.error("El sistema no funcionará correctamente sin el modelo")
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("¡TODOS LOS RECURSOS ESTÁN LISTOS PARA EL PROYECTO FINAL!")
    logger.info("="*60)

if __name__ == "__main__":
    download_essential_resources()
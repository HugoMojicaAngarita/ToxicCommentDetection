import sys
import os
import pandas as pd
import joblib
from pathlib import Path
from config import DATA_PROCESSED, MODELS_DIR
from modules.models import ToxicityModel
from preprocessing import load_and_preprocess
import logging
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Descargar recursos primero (desde la raíz)
try:
    # Añadir raíz al path para importar setup_resources
    sys.path.append(str(Path(__file__).parent.parent))
    from setup_resources import download_essential_resources
    logger.info("Descargando recursos esenciales...")
    download_essential_resources()
    logger.info("Recursos descargados exitosamente")
except ImportError as e:
    logger.error(f"No se pudo importar setup_resources: {str(e)}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error crítico al descargar recursos: {str(e)}")
    sys.exit(1)

def run_test_cases(model):
    """Ejecuta casos de prueba críticos y devuelve resultados"""
    test_cases = [
        ("Neutro", "Hoy es un día soleado", 0),
        ("Neutro con emoción", "¡Qué hermoso día hace hoy!", 0),
        ("Insulto directo", "Eres un completo idiota", 1),
        ("Ataque identidad", "Los musulmanes son el problema", 1),
        ("Amenaza", "Voy a matarte si hablas", 1),
        ("Lenguaje obsceno", "Eres un maldito pedazo de mierda", 1),
        ("Falso positivo", "Orgullosamente musulmán", 0),
        ("Positivo con identidad", "Las mujeres son excelentes líderes", 0),
        ("Amenaza física", "Te voy a golpear", 1),
        ("Sarcasmo tóxico", "¡Vaya solución tan brillante! Claramente no pensaste en nada", 1),
        ("Identidad compleja", "Como persona negra en tecnología, enfrento desafíos únicos", 0),
        ("Emoción positiva", "Me encanta este lugar", 0)
    ]
    
    logger.info("\nEvaluando casos de prueba críticos:")
    results = []
    
    for name, text, expected in test_cases:
        try:
            proba = model.predict(text)
            prediction = 1 if proba >= 0.5 else 0
            correct = prediction == expected
            results.append((name, text, proba, prediction, expected, correct))
            
            status = "✅" if correct else "❌"
            color = "\033[92m" if correct else "\033[91m"
            reset = "\033[0m"
            logger.info(f"{color}{status} {name:<25}: {proba:.4f} ({'TÓXICO' if prediction else 'No tóxico'}) | Texto: '{text[:30]}...'{reset}")
        except Exception as e:
            logger.error(f"Error procesando caso '{name}': {str(e)}")
            results.append((name, text, -1, -1, expected, False))
    
    return results

def generate_performance_report(results, model):
    """Genera un reporte de desempeño detallado"""
    # Calcular métricas
    accuracy = sum(1 for r in results if r[5]) / len(results)
    
    # Casos problemáticos
    false_negatives = [r for r in results if r[4] == 1 and not r[5]]
    false_positives = [r for r in results if r[4] == 0 and not r[5]]
    
    # Crear reporte
    report = [
        "\n=== REPORTE DE DESEMPEÑO ===",
        f"- Total casos: {len(results)}",
        f"- Precisión: {accuracy:.2%}",
        f"- Falsos negativos: {len(false_negatives)}",
        f"- Falsos positivos: {len(false_positives)}"
    ]
    
    # Detalles de casos problemáticos
    if false_negatives:
        report.append("\nFALSOS NEGATIVOS (tóxico no detectado):")
        for case in false_negatives:
            report.append(f"  - {case[0]}: '{case[1][:50]}...' | Prob: {case[2]:.4f}")
    
    if false_positives:
        report.append("\nFALSOS POSITIVOS (no tóxico marcado como tóxico):")
        for case in false_positives:
            report.append(f"  - {case[0]}: '{case[1][:50]}...' | Prob: {case[2]:.4f}")
    
    # Información del modelo
    report.append("\nCONFIGURACIÓN DEL MODELO:")
    report.append(f"- Usando embeddings LLM: {type(model.embedder).__name__}")
    report.append(f"- Clasificador: {type(model.clf).__name__}")
    
    return "\n".join(report)

def main():
    print("\n" + "="*60)
    print("SISTEMA DE DETECCIÓN DE TOXICIDAD CON LLMs")
    print("="*60)
    
    try:
        # 1. Cargar datos
        logger.info("\nCargando y preprocesando datos...")
        train, test = load_and_preprocess()
        
        # Estadísticas de datos
        logger.info(f"\nDatos cargados: {len(train)} ejemplos de entrenamiento")
        toxic_perc = train['target_binary'].mean() * 100
        logger.info(f"- Porcentaje de tóxicos: {toxic_perc:.1f}%")
        logger.info(f"- Ejemplo no tóxico: '{train[train['target_binary'] == 0]['comment_text'].iloc[0][:50]}...'")
        logger.info(f"- Ejemplo tóxico: '{train[train['target_binary'] == 1]['comment_text'].iloc[0][:50]}...'")
        
        # 2. Entrenar modelo con LLM
        logger.info("\nInicializando modelo con embeddings de LLM...")
        model = ToxicityModel(use_llm=True)
        
        logger.info("\nEntrenando modelo...")
        model.train(train['comment_text'], train['target_binary'])
        
        # 3. Evaluar con casos de prueba
        logger.info("\nEvaluando modelo con casos críticos...")
        results = run_test_cases(model)
        
        # 4. Generar reporte de desempeño
        report = generate_performance_report(results, model)
        logger.info(report)
        
        # 5. Guardar modelo
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / 'llm_toxicity_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"\nModelo guardado en: {model_path}")
        
        # 6. Preparar submission para Kaggle (con procesamiento en chunks)
        if not test.empty:
            logger.info(f"\nGenerando predicciones para {len(test)} textos...")
            
            # Procesar en chunks para evitar sobrecarga de memoria
            chunk_size = 100
            test_predictions = []
            
            for i in range(0, len(test), chunk_size):
                chunk = test.iloc[i:i+chunk_size]
                chunk_texts = chunk['comment_text'].tolist()
                test_predictions.extend([model.predict(text) for text in chunk_texts])
                processed = min(i+chunk_size, len(test))
                logger.info(f"📦 Procesados {processed}/{len(test)} textos ({processed/len(test):.1%})")
            
            submission = pd.DataFrame({
                'id': test['id'],
                'prediction': test_predictions
            })
            submission_path = MODELS_DIR / 'kaggle_submission.csv'
            submission.to_csv(submission_path, index=False)
            logger.info(f"✅ Submission generada: {submission_path}")
        
        logger.info("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        logger.exception(f"❌ Error crítico en el sistema: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
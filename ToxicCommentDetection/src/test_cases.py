from pathlib import Path
import joblib
from config import MODELS_DIR, IDENTITY_TERMS
from modules.text_processing import TextCleaner
from modules.context_analysis import ContextAnalyzer
from modules.identity_detection import IdentityDetector

def run_custom_tests():
    """Ejecuta casos de prueba personalizados con análisis detallado"""
    print("\n" + "="*60)
    print("CASOS DE PRUEBA PERSONALIZADOS - ANÁLISIS DETALLADO")
    print("="*60)
    
    # 1. Cargar modelos y componentes
    print("...Cargando modelos y componentes...")
    model = joblib.load(MODELS_DIR / 'toxicity_model.joblib')
    vectorizer = joblib.load(MODELS_DIR / 'tfidf_vectorizer.joblib')
    cleaner = TextCleaner()
    context_analyzer = ContextAnalyzer()
    identity_detector = IdentityDetector(IDENTITY_TERMS)
    
    # 2. Casos de prueba detallados por frase, categoria y lo esperado
    test_cases = [
        {
            "name": "Neutro",
            "text": "Aprobado por unanimidad en la sesión de hoy",
            "expected": "No tóxico"
        },
        {
            "name": "Insulto directo",
            "text": "Eres un incompetente sin remedio y deberías renunciar",
            "expected": "TÓXICO"
        },
        {
            "name": "Sarcasmo obvio",
            "text": "¡Vaya solución tan brillante que propone! Claramente no pensaste en nada",
            "expected": "TÓXICO"
        },
        {
            "name": "Ataque de identidad",
            "text": "Los negros no son inteligentes y deberían ser excluidos",
            "expected": "TÓXICO"
        },
        {
            "name": "Falso positivo (identidad positiva)",
            "text": "Soy musulmán y orgulloso de mis creencias",
            "expected": "No tóxico"
        },
        {
            "name": "Amenaza violenta",
            "text": "Voy a matarte si no lo haces, te juro que lo haré",
            "expected": "TÓXICO"
        },
        {
            "name": "Lenguaje obsceno",
            "text": "Eres un maldito idiota y no sirves para nada",
            "expected": "TÓXICO"
        },
        {
            "name": "Comentario positivo sobre identidad",
            "text": "Las mujeres son excelentes líderes y aportan mucho",
            "expected": "No tóxico"
        },
        {
            "name": "Ironía compleja",
            "text": "Claro, como no se te ocurrió antes esta idea tan maravillosa",
            "expected": "TÓXICO"
        },
        {
            "name": "Identidad sin ataque",
            "text": "Los cristianos celebran la navidad en diciembre",
            "expected": "No tóxico"
        }
    ]
    
    # 3. Procesar y evaluar cada caso
    print(f"\n...Evaluando {len(test_cases)} casos de prueba...")
    
    # Iterar sobre los casos de prueba
    for case in test_cases:
        # Preprocesamiento
        clean_text = cleaner.clean(case["text"])
        
        # Análisis de contexto e identidades
        context = context_analyzer.analyze(clean_text)
        identities = identity_detector.detect(clean_text)
        
        # Predicción base
        text_vector = vectorizer.transform([clean_text])
        base_proba = model.predict(text_vector)[0]
        
        # Ajustes contextuales
        adjusted_proba = base_proba
        if context['sarcasm_score'] > 0:
            adjusted_proba = min(base_proba + 0.25, 1.0)
        elif any(identities.values()):
            if context['sentiment'] > 0.3:  # Declaración positiva
                adjusted_proba = max(base_proba - 0.2, 0.0)
            elif context['sentiment'] < -0.3:  # Declaración negativa
                adjusted_proba = min(base_proba + 0.1, 1.0)
        
        # Resultado final
        prediction = "TÓXICO" if adjusted_proba >= 0.5 else "No tóxico"
        correct = "✅" if prediction == case["expected"] else "❌"
        
        # Mostrar resultados
        print(f"\n🔹 {case['name'].upper()} {correct}")
        print(f"📜 Texto original: '{case['text']}'")
        print(f"🧹 Texto limpio: '{clean_text[:80]}...'")
        print(f"🎭 Contexto: Sarcasmo={context['sarcasm_score']}, Sentimiento={context['sentiment']:.2f}")
        print(f"👥 Identidades detectadas: {', '.join(k for k, v in identities.items() if v) or 'Ninguna'}")
        print(f"📊 Probabilidades: Base={base_proba:.4f}, Ajustada={adjusted_proba:.4f}")
        print(f"🔮 Predicción: {prediction} (Esperado: {case['expected']})")
        
        # Mostrar palabras clave influyentes si es incorrecto
        if correct == "❌":
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.pipeline.named_steps['clf'].calibrated_classifiers_[0].estimator.feature_importances_
            important_words = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"🔍 Palabras clave influyentes: {', '.join(f'{w}({c:.2f})' for w, c in important_words)}")

if __name__ == "__main__":
    run_custom_tests()
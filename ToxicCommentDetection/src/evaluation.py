from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo con métricas estándar
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba vectorizados
        y_test: Etiquetas verdaderas
    Returns:
        Diccionario con métricas de evaluación
    """
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    
    return {
        'auc': roc_auc_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'f1': f1_score(y_test, y_pred_binary),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fn_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    }

def bias_analysis(model, vectorizer, test_df, identity_terms):
    """
    Analiza el sesgo del modelo por grupos de identidad
    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF ajustado
        test_df: DataFrame con datos de prueba
        identity_terms: Diccionario con términos de identidad
    Returns:
        Diccionario con métricas de sesgo por categoría
    """
    results = {}
    
    # Se asegura que el texto esté limpio
    if 'clean_text' not in test_df.columns:
        from modules.text_processing import TextCleaner
        cleaner = TextCleaner()
        test_df['clean_text'] = test_df['comment_text'].apply(cleaner.clean)
    
    # Vectorizar el texto
    X_test = vectorizer.transform(test_df['clean_text'])
    
    # Obtener predicciones
    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    
    # Solo proceder si tenemos labels verdaderos
    if 'target_binary' in test_df.columns:
        y_test = test_df['target_binary']
        
        for category, terms in identity_terms.items():
            # Filtrar comentarios que mencionan esta categoría
            mask = test_df['clean_text'].str.contains('|'.join(terms), case=False, na=False)
            
            if sum(mask) == 0:
                continue
                
            # Calcular métricas para este grupo
            tn, fp, fn, tp = confusion_matrix(y_test[mask], y_pred[mask]).ravel()
            
            results[category] = {
                'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'fn_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'support': sum(mask),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            }
    
    return results
import pandas as pd
from pathlib import Path
from config import DATA_RAW, DATA_PROCESSED
from sklearn.utils import shuffle

# Función para cargar y preprocesar datos automáticamente detectando columnas de toxicidad
# Modifica la función load_and_preprocess() así:

def load_and_preprocess():
    """Carga y preprocesa datos automáticamente detectando columnas de toxicidad"""
    try:
        train = pd.read_csv(DATA_RAW / 'train.csv').copy()
        test = pd.read_csv(DATA_RAW / 'test.csv').copy()
    except Exception as e:
        raise FileNotFoundError(f"Error al cargar archivos: {e}")

    # Limpieza básica
    for df in [train, test]:
        df['comment_text'] = df['comment_text'].fillna('').astype(str)
        df = df[df['comment_text'].str.strip().astype(bool)].copy()

    # Detección automática de columna de toxicidad
    toxic_cols = [col for col in train.columns if 'toxic' in col.lower()]
    
    if not toxic_cols:
        raise ValueError("No se encontró ninguna columna relacionada con toxicidad")
    
    # Usar la primera columna de toxicidad encontrada
    toxic_col = toxic_cols[0]
    
    # Crear target binario según el tipo de columna encontrada
    if train[toxic_col].dtype in ['float64', 'int64']:
        train['target_binary'] = (train[toxic_col] >= 0.5).astype(int)
    else:
        train['target_binary'] = train[toxic_col].astype(int)

    # Balanceo de clases SOLO SI hay suficientes datos
    if 'target_binary' in train.columns:
        toxic = train[train['target_binary'] == 1].copy()
        non_toxic = train[train['target_binary'] == 0].copy()
        
        # Solo balancear si ambas clases tienen al menos 1 muestra
        if len(toxic) > 0 and len(non_toxic) > 0:
            min_samples = min(len(toxic), len(non_toxic))
            balanced_train = pd.concat([
                toxic.sample(min_samples, random_state=42, replace=True),  # Permitir muestreo con reemplazo
                non_toxic.sample(min_samples, random_state=42, replace=True)
            ])
            train = shuffle(balanced_train, random_state=42)
        else:
            print("⚠️ Advertencia: No se pudo balancear - una clase no tiene muestras")

    # Guardar datos
    DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
    train.to_csv(DATA_PROCESSED / 'train_processed.csv', index=False)
    test.to_csv(DATA_PROCESSED / 'test_processed.csv', index=False)
    
    # Limitar tamaño para pruebas (solo si es necesario)
    # train = train.sample(min(500, len(train)))  # Usar máximo 500 muestras
    
    return train, test
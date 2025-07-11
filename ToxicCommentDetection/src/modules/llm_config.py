# src/modules/llm_config.py
import torch

# Configuración automática de dispositivo
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo eficiente
LLM_MODEL_NAME = "distilroberta-base"

# Parámetros de rendimiento
LLM_BATCH_SIZE = 16 if LLM_DEVICE == "cuda" else 8
LLM_MAX_LENGTH = 128

# Información para el reporte
LLM_TECH_STACK = {
    "framework": "PyTorch",
    "model": LLM_MODEL_NAME,
    "device": LLM_DEVICE,
    "max_length": LLM_MAX_LENGTH
}
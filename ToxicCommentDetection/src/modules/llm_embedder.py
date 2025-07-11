from .llm_config import LLM_MODEL_NAME, LLM_DEVICE, LLM_BATCH_SIZE, LLM_MAX_LENGTH
from transformers import AutoTokenizer, AutoModel
import torch

class LLMEmbedder:
    def __init__(self):
        self.device = torch.device(LLM_DEVICE)
        print(f"⚙️ Cargando modelo {LLM_MODEL_NAME} en {self.device}...")
        
        # Optimización para CPU
        if self.device.type == "cpu":
            torch.set_num_threads(4)  # Limitar hilos para evitar sobrecarga
        
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModel.from_pretrained(LLM_MODEL_NAME).to(self.device)
        self.model.eval()
        print("✅ Modelo LLM cargado exitosamente")
    
    def embed(self, texts):
        """Genera embeddings optimizados para CPU"""
        if not texts:
            return []
            
        if isinstance(texts, str):
            texts = [texts]
        
        # Procesamiento por lotes reducido para CPU
        batch_size = 4 if self.device.type == "cpu" else LLM_BATCH_SIZE
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=LLM_MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Pooling eficiente para CPU
                batch_embeds = outputs.last_hidden_state[:, 0, :].cpu()
                embeddings.append(batch_embeds)
        
        return torch.cat(embeddings, dim=0).numpy()
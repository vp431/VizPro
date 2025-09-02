"""
Model manager for loading and caching transformer models.
Copied and adapted from OldApp for single-page application.
"""
import os
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
import logging
from config import MODEL_CONFIG, LOCAL_MODELS_DIR

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages loading and caching of transformer models for the visualization tool.
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_model(self, model_path, model_type="bert", force_reload=False):
        """
        Load a model and tokenizer from local path or HuggingFace.
        
        Args:
            model_path: Path to model (local or HuggingFace model name)
            model_type: Type of model ("bert", "sentiment", "ner")
            force_reload: Whether to force reload if already cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = f"{model_path}_{model_type}"
        
        if not force_reload and cache_key in self.models:
            logger.info(f"Using cached model: {cache_key}")
            return self.models[cache_key], self.tokenizers[cache_key]
        
        try:
            logger.info(f"Loading model: {model_path} (type: {model_type})")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            
            # Load model based on type with proper meta tensor handling
            if model_type == "sentiment":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float32,
                    device_map=None,
                    local_files_only=True
                )
            elif model_type == "ner":
                model = AutoModelForTokenClassification.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    local_files_only=True
                )
            elif model_type == "qa":
                from transformers import AutoModelForQuestionAnswering
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    local_files_only=True
                )
            else:  # bert or general
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    local_files_only=True
                )
            
            # Move to device with proper handling for meta tensors
            if hasattr(model, '_modules'):
                # Check if model has meta tensors and handle appropriately
                try:
                    model = model.to(self.device)
                except RuntimeError as e:
                    if "meta tensor" in str(e):
                        # Use to_empty() for meta tensors
                        model = model.to_empty(device=self.device)
                        # Reinitialize parameters if needed
                        for param in model.parameters():
                            if param.is_meta:
                                param.data = torch.empty_like(param, device=self.device)
                    else:
                        raise e
            else:
                model = model.to(self.device)
            
            model.eval()
            
            # Cache the model and tokenizer
            self.models[cache_key] = model
            self.tokenizers[cache_key] = tokenizer
            
            logger.info(f"Successfully loaded model: {cache_key}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            raise
    
    def get_pipeline(self, model_path, task_type, force_reload=False):
        """
        Get a HuggingFace pipeline for the model.
        
        Args:
            model_path: Path to model
            task_type: Type of task ("sentiment-analysis", "ner", etc.)
            force_reload: Whether to force reload
            
        Returns:
            HuggingFace pipeline
        """
        cache_key = f"{model_path}_{task_type}"
        
        if not force_reload and cache_key in self.pipelines:
            return self.pipelines[cache_key]
        
        try:
            pipe = pipeline(task_type, model=model_path, device=0 if torch.cuda.is_available() else -1)
            self.pipelines[cache_key] = pipe
            return pipe
        except Exception as e:
            logger.error(f"Error creating pipeline for {model_path}: {str(e)}")
            raise
    
    def unload_model(self, model_path, model_type="bert"):
        """
        Unload a model from cache to free memory.
        """
        cache_key = f"{model_path}_{model_type}"
        if cache_key in self.models:
            del self.models[cache_key]
            del self.tokenizers[cache_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {cache_key}")
    
    def clear_cache(self):
        """
        Clear all cached models to free memory.
        """
        self.models.clear()
        self.tokenizers.clear()
        self.pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared all model cache")
    
    def get_model_info(self, model_path, model_type="bert"):
        """
        Get information about a loaded model.
        """
        cache_key = f"{model_path}_{model_type}"
        if cache_key not in self.models:
            return None
        
        model = self.models[cache_key]
        tokenizer = self.tokenizers[cache_key]
        
        return {
            "model_path": model_path,
            "model_type": model_type,
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.model_max_length,
            "device": str(model.device),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "cache_key": cache_key
        }

# Global model manager instance
model_manager = ModelManager()
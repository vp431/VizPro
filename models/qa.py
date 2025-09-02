"""
Question Answering (QA) functionality for the single-page app.
Base & native model: distilbert-base-uncased-distilled-squad
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import logging
from models.model_manager import model_manager

logger = logging.getLogger(__name__)

class QAModel:
    """
    Simple QA wrapper that answers a question given a context.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading QA model from: {self.model_path}")
            self.model, self.tokenizer = model_manager.load_model(
                self.model_path, model_type="qa"
            )
            self.pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            # Ensure model outputs attentions for analysis
            if hasattr(self.model, 'config'):
                self.model.config.output_attentions = True
            
            logger.info("QA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading QA model: {str(e)}")
            raise

    def answer(self, context: str, question: str):
        """
        Answer a question given the context.
        Returns: dict with 'answer', 'score', 'start', 'end'
        """
        if not context or not question:
            return {"answer": "", "score": 0.0, "start": -1, "end": -1}
        try:
            result = self.pipeline({"context": context, "question": question})
            # Normalize keys
            return {
                "answer": result.get("answer", ""),
                "score": float(result.get("score", 0.0)),
                "start": int(result.get("start", -1)),
                "end": int(result.get("end", -1))
            }
        except Exception as e:
            logger.error(f"Error in QA inference: {str(e)}")
            return {"error": str(e)}
    
    def get_attention_weights(self, text, layer_idx=None, head_idx=None):
        """
        Get attention weights for the input text using the QA model.
        
        Args:
            text: Input text string
            layer_idx: Specific layer index (None for all layers)
            head_idx: Specific head index (None for all heads)
            
        Returns:
            dict: Attention weights and metadata
        """
        try:
            logger.info(f"Computing attention weights for QA model: {text[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # Extract attention weights
            attentions = outputs.attentions  # Tuple of attention weights for each layer
            
            if not attentions:
                logger.warning("No attention weights returned from QA model")
                return None
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Convert attention tensors to numpy arrays
            attention_weights = []
            for layer_attention in attentions:
                # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                layer_attention_np = layer_attention[0].cpu().numpy()  # Remove batch dimension
                attention_weights.append(layer_attention_np)
            
            result = {
                "tokens": tokens,
                "attention_weights": attention_weights,
                "num_layers": len(attention_weights),
                "num_heads": attention_weights[0].shape[0] if attention_weights else 0,
                "sequence_length": len(tokens)
            }
            
            logger.info(f"Extracted attention weights: {result['num_layers']} layers, {result['num_heads']} heads")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting attention weights from QA model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

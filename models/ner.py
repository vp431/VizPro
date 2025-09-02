"""
Named Entity Recognition functionality for the single-page app.
Basic implementation for NER analysis.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging
from models.model_manager import model_manager

logger = logging.getLogger(__name__)

class NERModel:
    """
    Named Entity Recognition with attention visualization.
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the NER model and tokenizer."""
        try:
            logger.info(f"Loading NER model from: {self.model_path}")
            self.model, self.tokenizer = model_manager.load_model(
                self.model_path, model_type="ner"
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )
            
            logger.info("NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NER model: {str(e)}")
            raise
    
    def predict(self, text):
        """
        Predict named entities in text.
        
        Args:
            text: Input text string
            
        Returns:
            dict: NER prediction results
        """
        try:
            # Get predictions from pipeline
            entities = self.pipeline(text)
            
            # Process entities
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    "label": entity["entity_group"],  # Use "label" instead of "entity"
                    "word": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "confidence": entity["score"]  # Use "confidence" instead of "score"
                })
            
            return {
                "text": text,
                "entities": processed_entities,
                "num_entities": len(processed_entities)
            }
            
        except Exception as e:
            logger.error(f"Error in NER prediction: {str(e)}")
            raise
    
    def get_attention_weights(self, text, layer_idx=None, head_idx=None):
        """
        Get attention weights for NER model.
        
        Args:
            text: Input text string
            layer_idx: Specific layer index
            head_idx: Specific head index
            
        Returns:
            dict: Attention weights data
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # Extract attention weights
            attentions = outputs.attentions  # Tuple of attention tensors
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Convert attention tensors to numpy arrays
            attention_arrays = []
            for layer_attention in attentions:
                # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                attention_arrays.append(layer_attention.cpu().numpy())
            
            return {
                "tokens": tokens,
                "attention_weights": attention_arrays,
                "text": text,
                "num_layers": len(attention_arrays),
                "num_heads": attention_arrays[0].shape[1] if attention_arrays else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting NER attention weights: {str(e)}")
            return None
    
    def get_logit_matrix(self, text):
        """
        Get logit matrix for token-level NER predictions.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Logit matrix data for visualization
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with logits
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract logits and tokens
            logits = outputs.logits  # Shape: (batch_size, seq_len, num_classes)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Convert logits to numpy
            logits_np = logits.cpu().numpy()[0]  # Remove batch dimension: (seq_len, num_classes)
            
            # Get class names (entity labels)
            if hasattr(self.model.config, 'id2label'):
                class_names = [self.model.config.id2label[i] for i in range(logits_np.shape[1])]
            else:
                class_names = [f"Label_{i}" for i in range(logits_np.shape[1])]
            
            # Get probabilities for comparison
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs_np = probabilities.cpu().numpy()[0]  # (seq_len, num_classes)
            
            # Get predictions for each token
            predicted_classes = np.argmax(logits_np, axis=1)
            predicted_labels = [class_names[idx] for idx in predicted_classes]
            
            # Calculate confidence for each token
            token_confidences = [float(probs_np[i, predicted_classes[i]]) for i in range(len(predicted_classes))]
            
            return {
                "logits": logits_np.tolist(),  # (seq_len, num_classes)
                "probabilities": probs_np.tolist(),
                "class_names": class_names,
                "tokens": tokens,
                "text": text,
                "predicted_labels": predicted_labels,
                "predicted_classes": predicted_classes.tolist(),
                "token_confidences": token_confidences,
                "num_classes": len(class_names),
                "sequence_length": len(tokens)
            }
            
        except Exception as e:
            logger.error(f"Error getting NER logit matrix: {str(e)}")
            return None
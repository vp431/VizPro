"""
BERT attention visualization functionality for the single-page app.
Adapted from OldApp for attention analysis.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from models.model_manager import model_manager

logger = logging.getLogger(__name__)

class BertVisualizer:
    """
    BERT model attention visualization and analysis.
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the BERT model and tokenizer."""
        try:
            logger.info(f"Loading BERT model from: {self.model_path}")
            self.model, self.tokenizer = model_manager.load_model(
                self.model_path, model_type="bert"
            )
            
            # Ensure model outputs attentions
            self.model.config.output_attentions = True
            
            logger.info("BERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            raise
    
    def get_attention_weights(self, text, layer_idx=None, head_idx=None):
        """
        Get attention weights for the input text.
        
        Args:
            text: Input text string
            layer_idx: Specific layer index (None for all layers)
            head_idx: Specific head index (None for all heads)
            
        Returns:
            dict: Attention weights and metadata
        """
        try:
            logger.info(f"Computing attention weights for: {text[:50]}...")
            
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
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Process attention weights
            attention_data = []
            
            for layer_i, layer_attention in enumerate(attentions):
                # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                layer_attention = layer_attention[0]  # Remove batch dimension
                
                layer_data = {
                    "layer": layer_i,
                    "heads": []
                }
                
                for head_i in range(layer_attention.shape[0]):
                    head_attention = layer_attention[head_i].cpu().numpy()
                    
                    layer_data["heads"].append({
                        "head": head_i,
                        "attention_matrix": head_attention.tolist(),
                        "attention_weights": head_attention
                    })
                
                attention_data.append(layer_data)
            
            # Filter by specific layer/head if requested
            if layer_idx is not None:
                if layer_idx < len(attention_data):
                    layer_data = attention_data[layer_idx]
                    if head_idx is not None and head_idx < len(layer_data["heads"]):
                        head_data = layer_data["heads"][head_idx]
                        return {
                            "tokens": tokens,
                            "attention_matrix": head_data["attention_matrix"],
                            "layer": layer_idx,
                            "head": head_idx,
                            "text": text,
                            "num_layers": len(attentions),
                            "num_heads": attentions[0].shape[1]
                        }
                    else:
                        return {
                            "tokens": tokens,
                            "layer_data": layer_data,
                            "layer": layer_idx,
                            "text": text,
                            "num_layers": len(attentions),
                            "num_heads": attentions[0].shape[1]
                        }
            
            return {
                "tokens": tokens,
                "attention_data": attention_data,
                "text": text,
                "num_layers": len(attentions),
                "num_heads": attentions[0].shape[1],
                "sequence_length": len(tokens)
            }
            
        except Exception as e:
            logger.error(f"Error computing attention weights: {str(e)}")
            raise
    
    def get_attention_entropy(self, text):
        """
        Compute attention entropy for each layer and head.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Attention entropy data
        """
        try:
            logger.info(f"Computing attention entropy for: {text[:50]}...")
            
            attention_result = self.get_attention_weights(text)
            attention_data = attention_result["attention_data"]
            
            entropy_data = []
            
            for layer_data in attention_data:
                layer_entropy = {
                    "layer": layer_data["layer"],
                    "heads": []
                }
                
                for head_data in layer_data["heads"]:
                    attention_matrix = np.array(head_data["attention_weights"])
                    
                    # Compute entropy for each token's attention distribution
                    token_entropies = []
                    for i in range(attention_matrix.shape[0]):
                        attention_dist = attention_matrix[i]
                        # Add small epsilon to avoid log(0)
                        attention_dist = attention_dist + 1e-10
                        entropy = -np.sum(attention_dist * np.log(attention_dist))
                        token_entropies.append(float(entropy))
                    
                    # Compute average entropy for this head
                    avg_entropy = np.mean(token_entropies)
                    
                    layer_entropy["heads"].append({
                        "head": head_data["head"],
                        "token_entropies": token_entropies,
                        "average_entropy": float(avg_entropy)
                    })
                
                entropy_data.append(layer_entropy)
            
            return {
                "tokens": attention_result["tokens"],
                "entropy_data": entropy_data,
                "text": text,
                "num_layers": attention_result["num_layers"],
                "num_heads": attention_result["num_heads"]
            }
            
        except Exception as e:
            logger.error(f"Error computing attention entropy: {str(e)}")
            raise
    
    def get_token_embeddings(self, text, layer_idx=-1):
        """
        Get token embeddings from a specific layer.
        
        Args:
            text: Input text string
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            dict: Token embeddings and metadata
        """
        try:
            logger.info(f"Computing token embeddings for: {text[:50]}...")
            
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
            
            # Get model outputs with hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract hidden states
            hidden_states = outputs.hidden_states  # Tuple of hidden states for each layer
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Select specific layer
            if layer_idx == -1:
                layer_idx = len(hidden_states) - 1
            
            selected_layer = hidden_states[layer_idx][0]  # Remove batch dimension
            embeddings = selected_layer.cpu().numpy()
            
            return {
                "tokens": tokens,
                "embeddings": embeddings.tolist(),
                "layer": layer_idx,
                "text": text,
                "embedding_dim": embeddings.shape[-1],
                "sequence_length": len(tokens),
                "num_layers": len(hidden_states)
            }
            
        except Exception as e:
            logger.error(f"Error computing token embeddings: {str(e)}")
            raise
    
    def get_attention_summary(self, text):
        """
        Get a summary of attention patterns across all layers and heads.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Attention summary statistics
        """
        try:
            attention_result = self.get_attention_weights(text)
            attention_data = attention_result["attention_data"]
            
            summary = {
                "text": text,
                "tokens": attention_result["tokens"],
                "num_layers": attention_result["num_layers"],
                "num_heads": attention_result["num_heads"],
                "layer_summaries": []
            }
            
            for layer_data in attention_data:
                layer_summary = {
                    "layer": layer_data["layer"],
                    "head_summaries": []
                }
                
                for head_data in layer_data["heads"]:
                    attention_matrix = np.array(head_data["attention_weights"])
                    
                    # Compute statistics
                    max_attention = float(np.max(attention_matrix))
                    min_attention = float(np.min(attention_matrix))
                    mean_attention = float(np.mean(attention_matrix))
                    std_attention = float(np.std(attention_matrix))
                    
                    # Find most attended tokens
                    attention_sums = np.sum(attention_matrix, axis=0)
                    most_attended_idx = int(np.argmax(attention_sums))
                    
                    layer_summary["head_summaries"].append({
                        "head": head_data["head"],
                        "max_attention": max_attention,
                        "min_attention": min_attention,
                        "mean_attention": mean_attention,
                        "std_attention": std_attention,
                        "most_attended_token_idx": most_attended_idx,
                        "most_attended_token": attention_result["tokens"][most_attended_idx] if most_attended_idx < len(attention_result["tokens"]) else None
                    })
                
                summary["layer_summaries"].append(layer_summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error computing attention summary: {str(e)}")
            raise
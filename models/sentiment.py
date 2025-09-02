"""
Sentiment analysis functionality for the single-page app.
Adapted from OldApp with LIME integration.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
import logging
from models.model_manager import model_manager

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis with LIME explanations and token attributions.
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the sentiment model and tokenizer."""
        try:
            logger.info(f"Loading sentiment model from: {self.model_path}")
            self.model, self.tokenizer = model_manager.load_model(
                self.model_path, model_type="sentiment"
            )
            
            # Create pipeline for easier inference with proper device handling
            device_id = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                torch_dtype=torch.float32
            )
            
            logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise
    
    def predict(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Prediction results with label, score, and probabilities
        """
        try:
            # Get prediction from pipeline
            result = self.pipeline(text)[0]
            
            # Get detailed probabilities
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to numpy for easier handling
            probs = probabilities.cpu().numpy()[0]
            
            return {
                "label": result["label"],
                "score": result["score"],
                "probabilities": {
                    "NEGATIVE": float(probs[0]) if len(probs) > 1 else 1 - result["score"],
                    "POSITIVE": float(probs[1]) if len(probs) > 1 else result["score"]
                },
                "prediction": result["label"],
                "confidence": result["score"]
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {str(e)}")
            raise
    
    def explain_with_lime(self, text, num_features=10, num_samples=1000):
        """
        Generate LIME explanation for sentiment prediction.
        
        Args:
            text: Input text to explain
            num_features: Number of features to show in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            dict: LIME explanation with feature importances and visualization data
        """
        try:
            logger.info(f"Generating LIME explanation for: {text[:50]}...")
            
            # Create LIME explainer
            explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])
            
            # Define prediction function for LIME
            def predict_proba(texts):
                results = []
                for text in texts:
                    try:
                        prediction = self.predict(text)
                        probs = [
                            prediction["probabilities"]["NEGATIVE"],
                            prediction["probabilities"]["POSITIVE"]
                        ]
                        results.append(probs)
                    except:
                        # Fallback for failed predictions
                        results.append([0.5, 0.5])
                return np.array(results)
            
            # Generate explanation
            explanation = explainer.explain_instance(
                text,
                predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract feature importances
            feature_importance = explanation.as_list()
            
            # Get original prediction
            original_prediction = self.predict(text)
            
            # Create visualization data
            words = text.split()
            word_importances = {}
            
            for feature, importance in feature_importance:
                # Find the word in the original text
                for i, word in enumerate(words):
                    if feature.lower() in word.lower() or word.lower() in feature.lower():
                        word_importances[i] = {
                            "word": word,
                            "importance": importance,
                            "feature": feature
                        }
                        break
            
            return {
                "explanation": explanation,
                "feature_importance": feature_importance,
                "word_importances": word_importances,
                "original_prediction": original_prediction,
                "text": text,
                "words": words,
                "lime_score": explanation.score,
                "intercept": explanation.intercept[1] if hasattr(explanation, 'intercept') else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise
    
    def get_attention_weights(self, text, layer_idx=None, head_idx=None):
        """
        Get attention weights from the sentiment model.
        
        Args:
            text: Input text string
            layer_idx: Specific layer index (if None, returns all layers)
            head_idx: Specific head index (if None, returns all heads)
            
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
            logger.error(f"Error extracting attention weights: {str(e)}")
            return None
    
    def get_token_attributions(self, text):
        """
        Get token-level attributions using gradient-based methods.
        
        Args:
            text: Input text
            
        Returns:
            dict: Token attributions and visualization data
        """
        try:
            logger.info(f"Computing token attributions for: {text[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                return_offsets_mapping=True
            )
            
            # Move to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Enable gradients for input embeddings
            embeddings = self.model.get_input_embeddings()
            input_embeddings = embeddings(input_ids)
            input_embeddings.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
            
            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # Compute gradients
            target_score = probabilities[0, predicted_class]
            target_score.backward()
            
            # Get gradients and compute attributions
            gradients = input_embeddings.grad
            attributions = torch.norm(gradients, dim=-1).squeeze()
            
            # Convert to numpy
            attributions = attributions.cpu().detach().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Create token attribution data
            token_data = []
            for i, (token, attribution) in enumerate(zip(tokens, attributions)):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    token_data.append({
                        "token": token,
                        "attribution": float(attribution),
                        "position": i,
                        "normalized_attribution": float(attribution / np.max(attributions)) if np.max(attributions) > 0 else 0
                    })
            
            # Get original prediction
            original_prediction = self.predict(text)
            
            return {
                "tokens": token_data,
                "attributions": attributions.tolist(),
                "original_prediction": original_prediction,
                "text": text,
                "max_attribution": float(np.max(attributions)),
                "min_attribution": float(np.min(attributions))
            }
            
        except Exception as e:
            logger.error(f"Error computing token attributions: {str(e)}")
            raise
    
    def batch_predict(self, texts):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            list: List of prediction results
        """
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def get_attention_entropy(self, text):
        """
        Calculate attention entropy for all layers and heads.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Entropy data for visualization
        """
        try:
            # Get attention weights
            attention_data = self.get_attention_weights(text)
            if not attention_data:
                return None
            
            attention_weights = attention_data["attention_weights"]
            tokens = attention_data["tokens"]
            
            # Calculate entropy for each layer and head
            entropy_matrix = []
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_entropy = []
                # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                for head_idx in range(layer_attention.shape[1]):
                    head_attention = layer_attention[0, head_idx]  # Remove batch dimension
                    
                    # Calculate entropy for each token's attention distribution
                    entropies = []
                    for i in range(head_attention.shape[0]):
                        attention_dist = head_attention[i]
                        # Add small epsilon to avoid log(0)
                        attention_dist = attention_dist + 1e-10
                        entropy = -np.sum(attention_dist * np.log(attention_dist))
                        entropies.append(entropy)
                    
                    # Average entropy for this head
                    avg_entropy = np.mean(entropies)
                    layer_entropy.append(avg_entropy)
                
                entropy_matrix.append(layer_entropy)
            
            return {
                "entropy": np.array(entropy_matrix),
                "tokens": tokens,
                "text": text,
                "num_layers": len(entropy_matrix),
                "num_heads": len(entropy_matrix[0]) if entropy_matrix else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating attention entropy: {str(e)}")
            return None
    
    def get_sentiment_with_attention(self, text):
        """
        Get sentiment prediction along with attention weights.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Combined sentiment and attention data
        """
        try:
            # Get sentiment prediction
            sentiment_result = self.predict(text)
            
            # Get attention weights
            attention_data = self.get_attention_weights(text)
            
            if attention_data:
                return {
                    "sentiment": sentiment_result,
                    "tokens": attention_data["tokens"],
                    "attentions": attention_data["attention_weights"],
                    "text": text
                }
            else:
                return {
                    "sentiment": sentiment_result,
                    "tokens": text.split(),
                    "attentions": [],
                    "text": text
                }
                
        except Exception as e:
            logger.error(f"Error getting sentiment with attention: {str(e)}")
            return None
    
    def get_logit_matrix(self, text):
        """
        Get logit matrix for token-level and sequence-level predictions.
        
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
            logits = outputs.logits  # Shape: (batch_size, num_classes)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Convert logits to numpy
            logits_np = logits.cpu().numpy()[0]  # Remove batch dimension
            
            # Get class names
            if hasattr(self.model.config, 'id2label'):
                class_names = [self.model.config.id2label[i] for i in range(len(logits_np))]
            else:
                class_names = [f"Class_{i}" for i in range(len(logits_np))]
            
            # Get probabilities for comparison
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs_np = probabilities.cpu().numpy()[0]
            
            # Get prediction details
            predicted_class_idx = np.argmax(logits_np)
            predicted_class = class_names[predicted_class_idx]
            confidence = float(probs_np[predicted_class_idx])
            
            return {
                "logits": logits_np.tolist(),
                "probabilities": probs_np.tolist(),
                "class_names": class_names,
                "tokens": tokens,
                "text": text,
                "predicted_class": predicted_class,
                "predicted_class_idx": predicted_class_idx,
                "confidence": confidence,
                "num_classes": len(class_names)
            }
            
        except Exception as e:
            logger.error(f"Error getting logit matrix: {str(e)}")
            return None

    def get_sentence_embedding(self, text):
        """
        Get sentence and token embeddings using the model.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Embedding data for visualization
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract hidden states (token embeddings)
            hidden_states = outputs.hidden_states  # Tuple of hidden state tensors
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Use the last layer's hidden states as token embeddings
            last_hidden_state = hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
            token_embeddings = last_hidden_state[0].cpu().numpy()  # Remove batch dimension
            
            # Get sentence embedding (mean of token embeddings, excluding special tokens)
            # Find indices of special tokens to exclude
            special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
            valid_indices = [i for i, token in enumerate(tokens) if token not in special_tokens]
            
            if valid_indices:
                sentence_embedding = np.mean(token_embeddings[valid_indices], axis=0)
            else:
                sentence_embedding = np.mean(token_embeddings, axis=0)
            
            return {
                "sentence_embedding": sentence_embedding,
                "token_embeddings": token_embeddings,
                "tokens": tokens,
                "text": text,
                "embedding_dim": token_embeddings.shape[-1]
            }
            
        except Exception as e:
            logger.error(f"Error getting sentence embedding: {str(e)}")
            return None
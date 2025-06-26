import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
import lime
import lime.lime_text
import sklearn

class BERTAttentionVisualizer:
    """Utility class for visualizing attention in BERT models"""
    
    def __init__(self, model_name="huawei-noah/TinyBERT_General_4L_312D"):
        """Initialize with a BERT model
        
        Args:
            model_name: Name of the pretrained model from HuggingFace
        """
        print(f"Loading model: {model_name}")
        
        # Use local models if available, or download with a timeout and error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True, local_files_only=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to download model with increased timeout...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False, use_auth_token=False)
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True, local_files_only=False, use_auth_token=False)
        
        # Check if CUDA is available and move model to GPU if it is
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Get model architecture info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        
        # Cache for attention results to avoid recalculating
        self._attention_cache = {}
        
        # Task-specific models (loaded on demand)
        self._sentiment_model = None
        self._ner_model = None
        self._lime_explainer = None
    
    def get_attention(self, text):
        """Get attention maps for input text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing:
                - tokens: List of tokens
                - attentions: Tensor of attention weights [layers, heads, tokens, tokens]
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")
            
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        # Check cache first
        if text in self._attention_cache:
            return self._attention_cache[text]
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Limit sequence length to avoid memory issues
            if len(tokens) > 128:
                truncated_text = ' '.join(text.split()[:100])  # Rough truncation
                inputs = self.tokenizer(truncated_text, return_tensors="pt").to(self.device)
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert attention weights to numpy arrays
            # Shape: [layers, batch_size, heads, seq_length, seq_length]
            attentions = [layer.cpu().numpy() for layer in outputs.attentions]
            
            result = {
                "tokens": tokens,
                "attentions": attentions,
            }
            
            # Cache the result
            self._attention_cache[text] = result
            
            return result
        except Exception as e:
            raise RuntimeError(f"Error generating attention: {str(e)}")
    
    def get_attention_map(self, text, layer_idx=0, head_idx=0):
        """Get specific attention map for visualization
        
        Args:
            text: Input text string
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize
            
        Returns:
            Dictionary containing:
                - tokens: List of tokens
                - attention_map: 2D numpy array of attention weights
        """
        # Validate indices
        if not isinstance(layer_idx, (int, np.integer)) or layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index must be between 0 and {self.num_layers-1}, got {layer_idx}")
            
        if not isinstance(head_idx, (int, np.integer)) or head_idx < 0 or head_idx >= self.num_heads:
            raise ValueError(f"Head index must be between 0 and {self.num_heads-1}, got {head_idx}")
            
        try:
            # Get full attention
            result = self.get_attention(text)
            tokens = result["tokens"]
            
            # Extract attention map for the specified layer and head
            # Shape: [seq_length, seq_length]
            attention_map = result["attentions"][layer_idx][0, head_idx]
            
            return {
                "tokens": tokens,
                "attention_map": attention_map
            }
        except Exception as e:
            raise RuntimeError(f"Error getting attention map: {str(e)}")
    
    def get_model_info(self):
        """Get information about the model architecture
        
        Returns:
            Dictionary containing model architecture information
        """
        return {
            "model_name": self.model.config.name_or_path,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.model.config.hidden_size,
            "device": self.device
        }
    
    def load_sentiment_model(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Load a sentiment analysis model
        
        Args:
            model_name: Name of the pretrained model from HuggingFace
        """
        if self._sentiment_model is None:
            print(f"Loading sentiment model: {model_name}")
            self._sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
            self._sentiment_model.to(self.device)
        return self._sentiment_model
    
    def predict_sentiment(self, text):
        """Predict sentiment for input text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing:
                - label: Predicted label (0 for negative, 1 for positive)
                - score: Confidence score
                - tokens: List of tokens
                - attentions: Tensor of attention weights
        """
        model = self.load_sentiment_model()
        
        # Tokenize input
        inputs = self._sentiment_tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self._sentiment_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get prediction
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
        
        # Get attention weights
        attentions = [layer.cpu().numpy() for layer in outputs.attentions]
        
        return {
            "label": prediction,
            "score": confidence,
            "tokens": tokens,
            "attentions": attentions
        }
    
    def load_ner_model(self, model_name="dslim/bert-base-NER"):
        """Load a named entity recognition model
        
        Args:
            model_name: Name of the pretrained model from HuggingFace
        """
        if self._ner_model is None:
            print(f"Loading NER model: {model_name}")
            self._ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._ner_model = AutoModelForTokenClassification.from_pretrained(model_name, output_attentions=True)
            self._ner_model.to(self.device)
        return self._ner_model
    
    def predict_ner(self, text):
        """Predict named entities for input text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing:
                - entities: List of predicted entities
                - tokens: List of tokens
                - attentions: Tensor of attention weights
        """
        model = self.load_ner_model()
        
        # Tokenize input
        inputs = self._ner_tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self._ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        
        # Convert predictions to entity labels
        # This is a simplified version - in a real implementation you would use the model's label mapping
        entity_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
        
        # Get attention weights
        attentions = [layer.cpu().numpy() for layer in outputs.attentions]
        
        return {
            "entity_labels": entity_labels,
            "tokens": tokens,
            "attentions": attentions
        } 

    def _sentiment_predict_proba(self, texts):
        """Prediction function for LIME - takes multiple texts and returns probabilities
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of probabilities for each class (negative, positive)
        """
        model = self.load_sentiment_model()
        model.eval()
        
        all_probs = []
        batch_size = 16  # Process in batches to avoid memory issues
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize input
            inputs = self._sentiment_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probabilities)
        
        return np.vstack(all_probs)

    def _init_lime_explainer(self):
        """Initialize LIME text explainer"""
        if self._lime_explainer is None:
            self._lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=["Negative", "Positive"],
                kernel_width=25,
                verbose=False
            )
        return self._lime_explainer

    def explain_sentiment(self, text, num_features=10, num_samples=1000):
        """Explain sentiment prediction using LIME
        
        Args:
            text: Input text string
            num_features: Number of features to include in explanation
            num_samples: Number of samples to use for LIME
            
        Returns:
            Dictionary containing:
                - explanation: LIME explanation object
                - words: List of words in the explanation
                - weights: Weights for each word
                - prediction: Model prediction (0 for negative, 1 for positive)
                - probability: Probability of the prediction
        """
        # First get the prediction to confirm we have a valid sample
        sentiment_result = self.predict_sentiment(text)
        label = sentiment_result["label"]
        score = sentiment_result["score"]
        
        # Initialize the explainer
        explainer = self._init_lime_explainer()
        
        # Get the explanation
        explanation = explainer.explain_instance(
            text, 
            self._sentiment_predict_proba, 
            num_features=num_features, 
            num_samples=num_samples
        )
        
        # Extract the most important features
        words_with_weights = explanation.as_list()
        words = [item[0] for item in words_with_weights]
        weights = [item[1] for item in words_with_weights]
        
        return {
            "explanation": explanation,
            "words": words,
            "weights": weights,
            "prediction": label,
            "probability": score,
            "text": text
        } 
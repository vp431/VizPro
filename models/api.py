"""
Unified API for all model functionalities in the single-page app.
Adapted from OldApp for centralized model management.
"""
import torch
import numpy as np
from models.model_manager import model_manager
from models.sentiment import SentimentAnalyzer
from models.visualizer import BertVisualizer
from models.ner import NERModel
from models.qa import QAModel
from models.knowledge_competition import knowledge_competition_analyzer
import logging

logger = logging.getLogger(__name__)

class ModelAPI:
    """
    Unified API for accessing all model functionalities.
    Manages model loading and provides consistent interface for analysis.
    """
    
    def __init__(self):
        self.selected_model_path = None
        self.selected_model_type = None
        self._sentiment_analyzer = None
        self._bert_visualizer = None
        self._ner_model = None
        self._qa_model = None
        
    def set_selected_model(self, model_path, model_type):
        """Set the currently selected model."""
        self.selected_model_path = model_path
        self.selected_model_type = model_type
        logger.info(f"Selected model: {model_path} (type: {model_type})")
        
        # Clear cached analyzers if model changed
        self._sentiment_analyzer = None
        self._bert_visualizer = None
        self._ner_model = None
        self._qa_model = None
    
    def get_sentiment_analyzer(self, model_path=None):
        """Get or create sentiment analyzer for the selected model."""
        if model_path is None:
            model_path = self.selected_model_path
            
        if self._sentiment_analyzer is None or self._sentiment_analyzer.model_path != model_path:
            logger.info(f"Creating sentiment analyzer for: {model_path}")
            self._sentiment_analyzer = SentimentAnalyzer(model_path)
            
        return self._sentiment_analyzer
    
    def get_bert_visualizer(self, model_path=None):
        """Get or create BERT visualizer for the selected model."""
        if model_path is None:
            model_path = self.selected_model_path
            
        if self._bert_visualizer is None or self._bert_visualizer.model_path != model_path:
            logger.info(f"Creating BERT visualizer for: {model_path}")
            self._bert_visualizer = BertVisualizer(model_path)
            
        return self._bert_visualizer
    
    def get_ner_model(self, model_path=None):
        """Get or create NER model for the selected model."""
        if model_path is None:
            model_path = self.selected_model_path
            
        if self._ner_model is None or self._ner_model.model_path != model_path:
            logger.info(f"Creating NER model for: {model_path}")
            self._ner_model = NERModel(model_path)
            
        return self._ner_model
    
    # Sentiment Analysis Methods
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text."""
        analyzer = self.get_sentiment_analyzer()
        return analyzer.predict(text)
    
    def get_sentiment(self, text):
        """Get sentiment prediction (alias for analyze_sentiment for compatibility)."""
        return self.analyze_sentiment(text)
    
    def get_lime_explanation(self, text, num_features=10, num_samples=1000):
        """Get LIME explanation for sentiment prediction."""
        analyzer = self.get_sentiment_analyzer()
        return analyzer.explain_with_lime(text, num_features, num_samples)
    
    def explain_sentiment(self, text, num_features=10, num_samples=1000):
        """Get LIME explanation for sentiment prediction (alias for compatibility)."""
        return self.get_lime_explanation(text, num_features, num_samples)
    
    def get_attention_entropy(self, text):
        """Get attention entropy for all layers and heads."""
        analyzer = self.get_sentiment_analyzer()
        return analyzer.get_attention_entropy(text)
    
    def get_sentiment_with_attention(self, text):
        """Get sentiment prediction with attention weights."""
        analyzer = self.get_sentiment_analyzer()
        return analyzer.get_sentiment_with_attention(text)
    
    def get_sentence_embedding(self, text):
        """Get sentence embeddings and token embeddings."""
        analyzer = self.get_sentiment_analyzer()
        return analyzer.get_sentence_embedding(text)
    
    def get_attention_weights(self, text, layer_idx=None, head_idx=None):
        """Get attention weights for text."""
        if not self.selected_model_path:
            logger.error("No model selected for attention extraction")
            return None
            
        try:
            if self.selected_model_type == "sentiment":
                analyzer = self.get_sentiment_analyzer()
                return analyzer.get_attention_weights(text, layer_idx, head_idx)
            elif self.selected_model_type == "ner":
                ner_model = self.get_ner_model()
                return ner_model.get_attention_weights(text, layer_idx, head_idx)
            elif self.selected_model_type == "qa":
                # Use QA model's native attention extraction
                qa_model = self.get_qa_model()
                return qa_model.get_attention_weights(text, layer_idx, head_idx)
            else:
                visualizer = self.get_bert_visualizer()
                return visualizer.get_attention_weights(text, layer_idx, head_idx)
        except Exception as e:
            logger.error(f"Error getting attention weights: {str(e)}")
            return None
    
    def get_token_attributions(self, text):
        """Get token attributions for sentiment prediction."""
        analyzer = self.get_sentiment_analyzer()
        return analyzer.get_token_attributions(text)
    
    def get_logit_matrix(self, text):
        """Get logit matrix for the current model type."""
        if not self.selected_model_path:
            logger.error("No model selected for logit matrix extraction")
            return None
            
        try:
            if self.selected_model_type == "sentiment":
                analyzer = self.get_sentiment_analyzer()
                return analyzer.get_logit_matrix(text)
            elif self.selected_model_type == "ner":
                ner_model = self.get_ner_model()
                return ner_model.get_logit_matrix(text)
            else:
                logger.error(f"Logit matrix not supported for model type: {self.selected_model_type}")
                return None
        except Exception as e:
            logger.error(f"Error getting logit matrix: {str(e)}")
            return None
    
    # NER Methods
    def analyze_entities(self, text):
        """Analyze named entities in text."""
        ner_model = self.get_ner_model()
        return ner_model.predict(text)
    
    def get_ner_prediction(self, text):
        """Get NER prediction (alias for analyze_entities for compatibility)."""
        return self.analyze_entities(text)
    
    def get_entities_with_attention(self, text):
        """Get entities with attention weights for NER analysis."""
        ner_model = self.get_ner_model()
        # Get entities
        entities_result = ner_model.predict(text)
        # Get attention weights
        attention_data = ner_model.get_attention_weights(text)
        
        # Combine results
        result = {
            "entities": entities_result.get("entities", []) if entities_result else [],
            "tokens": attention_data.get("tokens", []) if attention_data else [],
            "attentions": attention_data.get("attention_weights", []) if attention_data else []
        }
        return result
    
    def get_ner_attention(self, text, layer_idx=None, head_idx=None):
        """Get attention weights for NER model."""
        ner_model = self.get_ner_model()
        return ner_model.get_attention_weights(text, layer_idx, head_idx)
    
    # QA Methods
    def get_qa_model(self, model_path=None):
        if model_path is None:
            model_path = self.selected_model_path
        if self._qa_model is None or self._qa_model.model_path != model_path:
            logger.info(f"Creating QA model for: {model_path}")
            self._qa_model = QAModel(model_path)
        return self._qa_model

    def answer_question(self, context: str, question: str):
        qa_model = self.get_qa_model()
        return qa_model.answer(context, question)
    
    # Knowledge Competition Methods
    def analyze_knowledge_competition(self, fact_text: str, counterfact_text: str):
        """Analyze competition between factual and counterfactual statements."""
        return knowledge_competition_analyzer.analyze_fact_counterfact_competition(fact_text, counterfact_text, self)
    
    def generate_fact_counterfact_pairs(self, template_type: str = "capital", num_pairs: int = 5):
        """Generate fact-counterfact pairs for analysis."""
        return knowledge_competition_analyzer.generate_fact_counterfact_pairs(template_type, num_pairs)

    # Model Information
    def get_model_info(self):
        """Get information about the currently selected model."""
        if not self.selected_model_path:
            return None
            
        return {
            "path": self.selected_model_path,
            "type": self.selected_model_type,
            "loaded_analyzers": {
                "sentiment": self._sentiment_analyzer is not None,
                "bert_visualizer": self._bert_visualizer is not None,
                "ner": self._ner_model is not None
            }
        }
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._sentiment_analyzer = None
        self._bert_visualizer = None
        self._ner_model = None
        model_manager.clear_cache()
        logger.info("Cleared all model cache")

# Global model API instance
model_api = ModelAPI()
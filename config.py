"""
Configuration settings for the Single-Page Transformers Visualization Tool.
"""
import os
import pathlib

# Base paths
BASE_DIR = pathlib.Path(__file__).parent.absolute()
LOCAL_MODELS_DIR = os.path.join(BASE_DIR, "LocalModels")
LOCAL_DATASETS_DIR = os.path.join(BASE_DIR, "LocalDatasets")

# Ensure directories exist
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)

# Default models
DEFAULT_BERT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_NER_MODEL = "dslim/bert-base-NER"

# Model configuration
MODEL_CONFIG = {
    "bert": {
        "name": DEFAULT_BERT_MODEL,
        "local_dir": os.path.join(LOCAL_MODELS_DIR, "TinyBERT"),
    },
    "sentiment": {
        "name": DEFAULT_SENTIMENT_MODEL,
        "local_dir": os.path.join(LOCAL_MODELS_DIR, "SentimentModel"),
    },
    "ner": {
        "name": DEFAULT_NER_MODEL,
        "local_dir": os.path.join(LOCAL_MODELS_DIR, "NERModel"),
    },
    "qa": {
        "name": "distilbert-base-uncased-distilled-squad",
        "local_dir": os.path.join(LOCAL_MODELS_DIR, "QAModel"),
    }
}

# Dataset configuration
DATASET_CONFIG = {
    "sentiment": {
        "SST2": {
            "name": "Stanford Sentiment Treebank",
            "local_dir": os.path.join(LOCAL_DATASETS_DIR, "SST2"),
            "default_samples": 200,
            "default_threshold": 0.7
        },
        "IMDb": {
            "name": "IMDb Movie Reviews",
            "local_dir": os.path.join(LOCAL_DATASETS_DIR, "IMDb"),
            "default_samples": 200,
            "default_threshold": 0.7
        }
    },
    "ner": {
        "CoNLL2003": {
            "name": "CoNLL-2003 Named Entity Recognition",
            "local_dir": os.path.join(LOCAL_DATASETS_DIR, "CoNLL2003"),
            "default_samples": 100,
            "default_threshold": 0.7
        }
    }
}

# UI Configuration
UI_CONFIG = {
    "theme": "bootstrap",
    "bootstrap_theme": "FLATLY",
    "brand_name": "Single-Page Transformers Visualization Tool",
    "navbar_color": "primary",
}

# App settings
DEBUG_MODE = True
PORT = 8050
HOST = "0.0.0.0"

# Visualization settings
VISUALIZATION_CONFIG = {
    "colorscale": [[0, "rgb(0,0,80)"], [1, "rgb(0,220,220)"]],
    "max_sequence_length": 128,
    "default_layer_idx": 0,
    "default_head_idx": 0,
}

# Feature configuration for different task+level combinations
FEATURE_CONFIG = {
    "sentiment": {
        "sentence": [
            {"id": "lime", "label": "LIME", "color": "primary"},
            {"id": "attention_entropy", "label": "Attention Entropy", "color": "info"},
            {"id": "token_embeddings", "label": "Token Embeddings", "color": "success"},
            {"id": "logit_matrix", "label": "Logit Matrix Heatmap", "color": "warning"}
        ],
        "model": [
            {"id": "error_analysis", "label": "Error Analysis", "color": "danger"},
            {"id": "error_patterns", "label": "Error Pattern Analysis", "color": "warning"},
            {"id": "similarity_analysis", "label": "Similarity Analysis", "color": "info"}
        ]
    },
    "ner": {
        "sentence": [
            {"id": "entity_viz", "label": "Entity Visualization", "color": "primary"},
            {"id": "attention_entropy", "label": "Attention Entropy", "color": "info"},
            {"id": "logit_matrix", "label": "Logit Matrix Heatmap", "color": "warning"}
        ],
        "model": [
            {"id": "in_development", "label": "In Development", "color": "secondary"}
        ]
    },
    "qa": {
        "sentence": [
            {"id": "knowledge_assessment", "label": "Knowledge Assessment", "color": "info"},
            {"id": "knowledge_competition", "label": "Knowledge Competition", "color": "primary"},
            {"id": "model_viz", "label": "Model Visualization", "color": "secondary"},
            {"id": "counterfactual_flow", "label": "Counterfactual Data Flow", "color": "warning"}
        ],
        "model": [
            {"id": "in_development", "label": "In Development", "color": "secondary"}
        ]
    }
}

# Help content configuration
HELP_CONFIG = {
    "attention": {
        "title": "What is Attention?",
        "content": """
        Attention mechanisms allow models to focus on different parts of the input when making predictions.
        
        **Key Concepts:**
        - **Attention Weights**: Show how much the model focuses on each token
        - **Multi-Head Attention**: Multiple attention patterns working in parallel
        - **Layer-wise Attention**: Different layers capture different types of patterns
        
        **Visualization Types:**
        - **Heatmaps**: Show attention weights as color intensity
        - **Line Plots**: Show attention flow between tokens
        - **Head Grids**: Compare multiple attention heads simultaneously
        """
    },
    "model": {
        "title": "How does this work?",
        "content": """
        This tool provides comprehensive analysis of transformer models through multiple lenses:
        
        **Sentence Level Analysis:**
        - Analyze individual sentences in detail
        - See token-level attributions and attention patterns
        - Understand specific predictions
        
        **Model Level Analysis:**
        - Analyze model behavior across datasets
        - Identify systematic errors and patterns
        - Compare performance across different examples
        
        **Model Types:**
        - **Native Models**: Trained specifically for the task
        - **Adaptable Models**: Can be fine-tuned for the task
        - **Incompatible Models**: Cannot be used for the selected task
        """
    },
    "metrics": {
        "title": "Understanding Metrics",
        "content": """
        Key metrics and visualizations explained:
        
        **Performance Metrics:**
        - **Accuracy**: Percentage of correct predictions
        - **Confidence**: Model's certainty in predictions
        - **F1 Score**: Balanced measure of precision and recall
        
        **Error Analysis:**
        - **High Confidence Errors**: Wrong predictions with high confidence
        - **Low Confidence Correct**: Right predictions with low confidence
        - **Error Patterns**: Systematic mistakes the model makes
        
        **Attention Metrics:**
        - **Attention Entropy**: How focused or diffuse attention is
        - **Token Importance**: Which tokens matter most for predictions
        """
    }
}
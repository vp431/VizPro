"""
Utility functions for scanning and managing local models.
"""
import os
import json
from pathlib import Path
from config import LOCAL_MODELS_DIR

def scan_local_models():
    """
    Scan the LocalModels directory for available models.
    
    Returns:
        Dictionary containing available models with unique keys
    """
    models = {}
    
    if not os.path.exists(LOCAL_MODELS_DIR):
        print(f"LocalModels directory does not exist: {LOCAL_MODELS_DIR}")
        return models
    
    print(f"Scanning models in: {LOCAL_MODELS_DIR}")
    
    # Recursively scan for config.json files
    for root, dirs, files in os.walk(LOCAL_MODELS_DIR):
        if "config.json" in files:
            try:
                config_path = os.path.join(root, "config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Get the model name from the path structure
                # Extract from path like "LocalModels/ModelType/models--name--model/snapshots/hash"
                path_parts = Path(root).parts
                model_name = None
                
                # Try to extract model name from path
                for part in path_parts:
                    if part.startswith("models--"):
                        model_name = part.replace("models--", "").replace("--", "/")
                        break
                
                # If no model name found in path, use directory name
                if not model_name:
                    model_name = os.path.basename(root)
                
                # Find the model type directory (direct child of LOCAL_MODELS_DIR)
                model_type_dir = None
                current_path = Path(root)
                while current_path != Path(LOCAL_MODELS_DIR) and current_path.parent != Path(LOCAL_MODELS_DIR):
                    current_path = current_path.parent
                
                if current_path.parent == Path(LOCAL_MODELS_DIR):
                    model_type_dir = current_path.name
                
                # Categorize model based on architecture, config or parent directory
                model_type = categorize_model(config, model_type_dir)
                
                # Create unique key for this model, prioritizing native models over duplicates
                model_key = f"{model_type_dir}_{model_name}" if model_type_dir else model_name
                
                # Check if we already have this model (prefer native over adaptable)
                existing_key = None
                for existing_model_key, existing_model in models.items():
                    if existing_model["name"] == model_name:
                        existing_key = existing_model_key
                        break
                
                # If we found an existing model, decide which one to keep
                if existing_key:
                    existing_model = models[existing_key]
                    # Prefer native models over BERT adaptable models
                    if model_type in ["sentiment", "ner"] and existing_model["type"] == "bert":
                        # Replace the BERT model with the native one
                        print(f"Replacing BERT model {existing_key} with native {model_key}")
                        del models[existing_key]
                    else:
                        # Keep the existing model, skip this one
                        print(f"Skipping duplicate model: {model_key} (keeping {existing_key})")
                        continue
                
                models[model_key] = {
                    "name": model_name,
                    "path": root,
                    "type": model_type,
                    "config": config,
                    "display_name": config.get("_name_or_path", model_name),
                    "type_dir": model_type_dir
                }
                
                print(f"Added model: {model_key} -> {model_name} (type: {model_type})")
                
            except Exception as e:
                print(f"Error reading config at {root}: {e}")
    
    print(f"Total models found: {len(models)}")
    return models

def categorize_model(config, model_type_dir=None):
    """
    Categorize a model based on its configuration and directory.
    
    Args:
        config: Model configuration dictionary
        model_type_dir: Name of the parent directory (NERModel, SentimentModel, etc.)
        
    Returns:
        String indicating model type ('bert', 'sentiment', 'ner', 'other')
    """
    if not config:
        return "other"
    
    # First try to categorize based on the parent directory name
    if model_type_dir:
        if "NER" in model_type_dir:
            return "ner"
        elif "Sentiment" in model_type_dir:
            return "sentiment"
        elif "QA" in model_type_dir or "QAModel" in model_type_dir or "Question" in model_type_dir:
            return "qa"
        elif "BERT" in model_type_dir or "Bert" in model_type_dir:
            return "bert"
    
    # Then check architecture type
    architectures = config.get("architectures", [])
    
    if any("ForSequenceClassification" in arch for arch in architectures):
        # Check if it's specifically for sentiment analysis
        num_labels = config.get("num_labels", 0)
        if num_labels == 2:  # Binary classification, likely sentiment
            return "sentiment"
        else:
            return "sentiment"  # Assume sequence classification is sentiment for now
    elif any("ForTokenClassification" in arch for arch in architectures):
        return "ner"
    elif any("QuestionAnswering" in arch for arch in architectures):
        return "qa"
    elif any("BertModel" in arch or "DistilBertModel" in arch for arch in architectures):
        return "bert"
    else:
        # Check model name for TinyBERT or other general models
        model_name = config.get("_name_or_path", "").lower()
        if "squad" in model_name or "question" in model_name:
            return "qa"
        if "tinybert" in model_name or "general" in model_name or "bert" in model_name:
            return "bert"
        return "other"

def get_model_display_name(model_info):
    """
    Get a user-friendly display name for a model.
    
    Args:
        model_info: Model information dictionary
        
    Returns:
        String display name
    """
    if model_info.get("config"):
        # Try to get name from config
        name_or_path = model_info["config"].get("_name_or_path", model_info["name"])
        if name_or_path and name_or_path != model_info["name"]:
            return f"{model_info['name']} ({name_or_path})"
    
    return model_info["name"]

def validate_model_compatibility(model_path, task_type):
    """
    Validate if a model is compatible with a specific task.
    
    Args:
        model_path: Path to the model directory
        task_type: Type of task ('sentiment', 'ner', 'bert')
        
    Returns:
        Tuple of (compatibility_status, message) where status is 'compatible', 'adaptable', or 'incompatible'
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return 'incompatible', 'No configuration file found'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get the model type directory from path
        model_type_dir = None
        current_path = Path(model_path)
        while current_path != Path(LOCAL_MODELS_DIR) and current_path.parent != Path(LOCAL_MODELS_DIR):
            current_path = current_path.parent
        
        if current_path.parent == Path(LOCAL_MODELS_DIR):
            model_type_dir = current_path.name
            
        model_type = categorize_model(config, model_type_dir)
        
        # Check compatibility rules
        if model_type == task_type:
            return 'compatible', 'Fully compatible'
        elif model_type == 'bert' and task_type in ['sentiment', 'ner']:
            return 'adaptable', 'Base BERT model - will be adapted'
        elif (model_type == 'sentiment' and task_type == 'ner') or \
             (model_type == 'ner' and task_type == 'sentiment'):
            return 'incompatible', f'{model_type.upper()} model not compatible with {task_type.upper()} tasks'
        else:
            # Check if it's TinyBERT or similar general models
            model_name = config.get("_name_or_path", "").lower()
            if "tinybert" in model_name or "general" in model_name:
                return 'adaptable', 'General BERT model - will be adapted'
            return 'adaptable', 'Will attempt to adapt for this task'
        
    except Exception as e:
        return 'incompatible', f'Error reading model configuration: {str(e)}'
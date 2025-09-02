import os
from transformers import AutoModel, AutoTokenizer

# List of recommended models and their Hugging Face names
MODELS = {
    'NERModel': {
        'name': 'dslim/bert-base-NER',
        'local_dir': 'LocalModels/NERModel/models--dslim--bert-base-NER'
    },
    'SentimentModel': {
        'name': 'distilbert-base-uncased-finetuned-sst-2-english',
        'local_dir': 'LocalModels/SentimentModel/models--distilbert-base-uncased-finetuned-sst-2-english'
    },
    'TinyBERT': {
        'name': 'huawei-noah/TinyBERT_General_4L_312D',
        'local_dir': 'LocalModels/TinyBERT/models--huawei-noah--TinyBERT_General_4L_312D'
    },
    'QAModel': {
        'name': 'distilbert-base-uncased-distilled-squad',
        'local_dir': 'LocalModels/QAModel/models--distilbert-base-uncased-distilled-squad'
    }
}

def download_model(model_name, local_dir):
    print(f"\nDownloading {model_name} to {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    AutoModel.from_pretrained(model_name, cache_dir=local_dir)
    AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    print(f"Downloaded {model_name} successfully!\n")

def check_model_exists(local_dir):
    """Check if a model already exists locally"""
    if not os.path.exists(local_dir):
        return False
    
    # Check if directory has content
    if len(os.listdir(local_dir)) > 0:
        return True
    
    return False

def main():
    print("Checking for locally available models:")
    available_models = []
    unavailable_models = []
    
    for key, info in MODELS.items():
        if check_model_exists(info['local_dir']):
            print(f"  ✓ {key}: {info['name']} (available locally)")
            available_models.append(key)
        else:
            print(f"  ✗ {key}: {info['name']} (not available)")
            unavailable_models.append(key)
    
    if not unavailable_models:
        print("\nAll models are already available locally.")
        return
    
    print("\nModels available for download:")
    for i, key in enumerate(unavailable_models, 1):
        info = MODELS[key]
        print(f"  {i}. {key}: {info['name']}")
    
    choice = input("\nDownload (a)ll missing models or (s)elect specific models? [a/s]: ").strip().lower()
    to_download = []
    
    if choice == 'a':
        to_download = unavailable_models
    else:
        for key in unavailable_models:
            info = MODELS[key]
            yn = input(f"Download {key} ({info['name']})? [y/n]: ").strip().lower()
            if yn == 'y':
                to_download.append(key)
    
    if not to_download:
        print("No models selected for download. Exiting.")
        return
    
    for key in to_download:
        info = MODELS[key]
        download_model(info['name'], info['local_dir'])
    
    print("All selected models downloaded.")

if __name__ == "__main__":
    main() 
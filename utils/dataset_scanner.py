"""
Utility functions for scanning and managing local datasets.
"""
import os
import json
from pathlib import Path
from config import LOCAL_DATASETS_DIR, DATASET_CONFIG

def scan_local_datasets():
    """
    Scan the LocalDatasets directory for available datasets.
    
    Returns:
        Dictionary containing available datasets organized by type
    """
    datasets = {
        "sentiment": [],
        "ner": []
    }
    
    if not os.path.exists(LOCAL_DATASETS_DIR):
        return datasets
    
    # Scan for dataset directories
    for dataset_type, type_datasets in DATASET_CONFIG.items():
        for dataset_key, dataset_info in type_datasets.items():
            local_dir = dataset_info['local_dir']
            if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
                datasets[dataset_type].append({
                    "key": dataset_key,
                    "name": dataset_info['name'],
                    "path": local_dir,
                    "default_samples": dataset_info.get('default_samples', 200),
                    "default_threshold": dataset_info.get('default_threshold', 0.7)
                })
    
    return datasets

def get_dataset_display_name(dataset_info):
    """
    Get a user-friendly display name for a dataset.
    
    Args:
        dataset_info: Dataset information dictionary
        
    Returns:
        String display name
    """
    return f"{dataset_info['key']}: {dataset_info['name']}"

def get_dataset_files(dataset_key, dataset_type):
    """
    Get the file paths for a specific dataset.
    
    Args:
        dataset_key: Key identifying the dataset (e.g., 'SST2')
        dataset_type: Type of dataset ('sentiment', 'ner')
        
    Returns:
        Dictionary with file paths for train, dev, and test sets
    """
    if dataset_type not in DATASET_CONFIG or dataset_key not in DATASET_CONFIG[dataset_type]:
        return {}
    
    dataset_info = DATASET_CONFIG[dataset_type][dataset_key]
    local_dir = dataset_info['local_dir']
    
    files = {}
    
    # Handle different dataset formats
    if dataset_key == 'SST2':
        # First check the direct path
        direct_files = {
            'train': os.path.join(local_dir, 'train.tsv'),
            'dev': os.path.join(local_dir, 'dev.tsv'),
            'test': os.path.join(local_dir, 'test.tsv')
        }
        
        # Then check the SST-2 subdirectory (which is how the GLUE dataset is structured)
        sst2_subdir_files = {
            'train': os.path.join(local_dir, 'SST-2', 'train.tsv'),
            'dev': os.path.join(local_dir, 'SST-2', 'dev.tsv'),
            'test': os.path.join(local_dir, 'SST-2', 'test.tsv')
        }
        
        # Combine and prioritize files that exist
        for key in direct_files:
            if os.path.exists(direct_files[key]):
                files[key] = direct_files[key]
            elif os.path.exists(sst2_subdir_files[key]):
                files[key] = sst2_subdir_files[key]
    elif dataset_key == 'IMDb':
        # IMDb has a different directory structure
        files = {
            'train_pos': os.path.join(local_dir, 'aclImdb', 'train', 'pos'),
            'train_neg': os.path.join(local_dir, 'aclImdb', 'train', 'neg'),
            'test_pos': os.path.join(local_dir, 'aclImdb', 'test', 'pos'),
            'test_neg': os.path.join(local_dir, 'aclImdb', 'test', 'neg')
        }
    elif dataset_key == 'CoNLL2003':
        # CoNLL-2003 has specific files
        files = {
            'train': os.path.join(local_dir, 'train.txt'),
            'valid': os.path.join(local_dir, 'valid.txt'),
            'test': os.path.join(local_dir, 'test.txt')
        }
    
    # Verify files exist
    return {k: v for k, v in files.items() if os.path.exists(v)}

def load_dataset_samples(dataset_key, dataset_type, split='dev', max_samples=200):
    """
    Load samples from a dataset.
    
    Args:
        dataset_key: Key identifying the dataset (e.g., 'SST2')
        dataset_type: Type of dataset ('sentiment', 'ner')
        split: Data split to load ('train', 'dev', 'test')
        max_samples: Maximum number of samples to load
        
    Returns:
        List of samples with text and labels
    """
    import pandas as pd
    import random
    
    files = get_dataset_files(dataset_key, dataset_type)
    samples = []
    
    print(f"Available files for {dataset_key}: {list(files.keys())}")
    
    try:
        if dataset_key == 'SST2':
            if split in files:
                print(f"Loading SST2 {split} data from {files[split]}")
                try:
                    # Try reading with header first
                    try:
                        df = pd.read_csv(files[split], sep='\t')
                        has_header = True
                    except Exception:
                        # If that fails, try reading without header
                        df = pd.read_csv(files[split], sep='\t', header=None)
                        has_header = False
                    
                    print(f"SST2 file read successfully. Has header: {has_header}")
                    print(f"SST2 columns: {df.columns.tolist()}")
                    
                    # Handle different column names
                    text_col = None
                    label_col = None
                    
                    if has_header:
                        # Try to find the text column
                        for possible_text_col in ['sentence', 'text', 'Sentence']:
                            if possible_text_col in df.columns:
                                text_col = possible_text_col
                                break
                        
                        # Try to find the label column
                        for possible_label_col in ['label', 'Label', 'sentiment', 'class']:
                            if possible_label_col in df.columns:
                                label_col = possible_label_col
                                break
                    else:
                        # If no header, assume first column is text, second is label
                        if len(df.columns) >= 2:
                            text_col = df.columns[0]
                            label_col = df.columns[1]
                    
                    if text_col is not None and label_col is not None:
                        print(f"Using columns: {text_col} for text and {label_col} for labels")
                        # Check if the dataframe has data
                        if len(df) > 0:
                            samples = df[[text_col, label_col]].values.tolist()
                            # Convert to list of dictionaries
                            samples = [{'text': str(text), 'label': str(label)} for text, label in samples]
                            
                            # Limit samples
                            if len(samples) > max_samples:
                                samples = random.sample(samples, max_samples)
                        else:
                            print(f"Error: No data found in {files[split]}")
                    else:
                        print(f"Error: Could not find text or label columns in {files[split]}")
                        print(f"Available columns: {df.columns.tolist()}")
                except Exception as e:
                    print(f"Error parsing SST2 file {files[split]}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Error: Split '{split}' not found for SST2. Available splits: {list(files.keys())}")
        
        elif dataset_key == 'IMDb':
            pos_key = f"{split}_pos"
            neg_key = f"{split}_neg"
            
            if pos_key in files and neg_key in files:
                print(f"Loading IMDb {split} data from {files[pos_key]} and {files[neg_key]}")
                
                # Check if directories exist and contain files
                if not os.path.isdir(files[pos_key]) or not os.path.isdir(files[neg_key]):
                    print(f"Error: IMDb directories not found: {files[pos_key]}, {files[neg_key]}")
                    return samples
                
                # Read positive reviews
                try:
                    pos_files = [os.path.join(files[pos_key], f) for f in os.listdir(files[pos_key]) 
                               if f.endswith('.txt')]
                    print(f"Found {len(pos_files)} positive review files")
                except Exception as e:
                    print(f"Error reading positive review directory: {str(e)}")
                    pos_files = []
                
                # Read negative reviews
                try:
                    neg_files = [os.path.join(files[neg_key], f) for f in os.listdir(files[neg_key]) 
                               if f.endswith('.txt')]
                    print(f"Found {len(neg_files)} negative review files")
                except Exception as e:
                    print(f"Error reading negative review directory: {str(e)}")
                    neg_files = []
                
                if not pos_files and not neg_files:
                    print("Error: No review files found in IMDb directories")
                    return samples
                
                # Limit and balance samples
                max_per_class = max_samples // 2
                if len(pos_files) > max_per_class:
                    pos_files = random.sample(pos_files, max_per_class)
                if len(neg_files) > max_per_class:
                    neg_files = random.sample(neg_files, max_per_class)
                
                # Load positive samples
                for file_path in pos_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read().strip()
                            if text:  # Only add non-empty texts
                                samples.append({'text': text, 'label': '1'})
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
                        continue
                
                # Load negative samples
                for file_path in neg_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read().strip()
                            if text:  # Only add non-empty texts
                                samples.append({'text': text, 'label': '0'})
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
                        continue
                
                # Shuffle samples
                random.shuffle(samples)
                print(f"Successfully loaded {len(samples)} IMDb samples")
            else:
                print(f"Error: Split '{split}' not found for IMDb. Available splits: {list(files.keys())}")
                print(f"Looking for keys: {pos_key}, {neg_key}")
        
        elif dataset_key == 'CoNLL2003':
            # NER datasets require different processing
            if split in files or (split == 'dev' and 'valid' in files):
                file_path = files.get(split, files.get('valid'))
                print(f"Loading CoNLL2003 {split} data from {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Process CoNLL format (simplified)
                    sentences = []
                    current_sentence = []
                    current_labels = []
                    
                    for line in content.split('\n'):
                        if line.strip() == '':
                            if current_sentence:
                                sentences.append({
                                    'text': ' '.join(current_sentence),
                                    'tokens': current_sentence,
                                    'labels': current_labels
                                })
                                current_sentence = []
                                current_labels = []
                        else:
                            parts = line.strip().split()
                            if len(parts) >= 4:  # CoNLL-2003 format
                                token = parts[0]
                                label = parts[3]
                                current_sentence.append(token)
                                current_labels.append(label)
                    
                    # Add the last sentence if not empty
                    if current_sentence:
                        sentences.append({
                            'text': ' '.join(current_sentence),
                            'tokens': current_sentence,
                            'labels': current_labels
                        })
                    
                    # Limit samples
                    if len(sentences) > max_samples:
                        samples = random.sample(sentences, max_samples)
                    else:
                        samples = sentences
            else:
                print(f"Error: Split '{split}' not found for CoNLL2003. Available splits: {list(files.keys())}")
    
    except Exception as e:
        print(f"Error loading dataset {dataset_key}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"Loaded {len(samples)} samples from {dataset_key} dataset")
    return samples

def scan_datasets():
    """
    Scan available datasets and organize them by task type.
    
    Returns:
        Dictionary with task types as keys and available datasets as values
    """
    from config import DATASET_CONFIG
    
    datasets = {
        "sentiment": {},
        "ner": {}
    }
    
    # Check each configured dataset
    for task_type, task_datasets in DATASET_CONFIG.items():
        for dataset_key, dataset_config in task_datasets.items():
            local_dir = dataset_config.get("local_dir")
            if local_dir and os.path.exists(local_dir):
                datasets[task_type][dataset_key] = {
                    "name": dataset_config.get("name", dataset_key),
                    "display_name": dataset_config.get("name", dataset_key),
                    "local_dir": local_dir,
                    "default_samples": dataset_config.get("default_samples", 100),
                    "default_threshold": dataset_config.get("default_threshold", 0.7)
                }
    
    return datasets 
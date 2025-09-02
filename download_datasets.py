import os
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import io

# List of recommended datasets and their sources
DATASETS = {
    'SST2': {
        'name': 'Stanford Sentiment Treebank',
        'local_dir': 'LocalDatasets/SST2',
        'url': 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
        'type': 'sentiment'
    },
    'IMDb': {
        'name': 'IMDb Movie Reviews',
        'local_dir': 'LocalDatasets/IMDb',
        'url': 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        'type': 'sentiment'
    },
    'CoNLL2003': {
        'name': 'CoNLL-2003 Named Entity Recognition',
        'local_dir': 'LocalDatasets/CoNLL2003',
        'url': 'https://data.deepai.org/conll2003.zip',
        'type': 'ner'
    }
}

def download_file(url, local_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(local_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def extract_zip(zip_path, extract_dir):
    """Extract a zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted {zip_path} to {extract_dir}")

def extract_tar_gz(tar_gz_path, extract_dir):
    """Extract a tar.gz file"""
    import tarfile
    with tarfile.open(tar_gz_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_dir)
    print(f"Extracted {tar_gz_path} to {extract_dir}")

def download_dataset(dataset_key):
    """Download and prepare a specific dataset"""
    dataset_info = DATASETS[dataset_key]
    local_dir = dataset_info['local_dir']
    url = dataset_info['url']
    name = dataset_info['name']
    
    print(f"\nDownloading {name} to {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    
    # Determine file extension
    if url.endswith('.zip'):
        file_path = os.path.join(local_dir, f"{dataset_key}.zip")
        download_file(url, file_path)
        extract_zip(file_path, local_dir)
    elif url.endswith('.tar.gz'):
        file_path = os.path.join(local_dir, f"{dataset_key}.tar.gz")
        download_file(url, file_path)
        extract_tar_gz(file_path, local_dir)
    else:
        file_path = os.path.join(local_dir, f"{dataset_key}.data")
        download_file(url, file_path)
    
    print(f"Downloaded {name} successfully!\n")

def check_dataset_exists(dataset_key):
    """Check if a dataset already exists locally"""
    dataset_info = DATASETS[dataset_key]
    local_dir = dataset_info['local_dir']
    
    if not os.path.exists(local_dir):
        return False
    
    # Check if directory has content
    if len(os.listdir(local_dir)) > 0:
        return True
    
    return False

def main():
    print("Checking for locally available datasets:")
    available_datasets = []
    unavailable_datasets = []
    
    for key, info in DATASETS.items():
        if check_dataset_exists(key):
            print(f"  ✓ {key}: {info['name']} (available locally)")
            available_datasets.append(key)
        else:
            print(f"  ✗ {key}: {info['name']} (not available)")
            unavailable_datasets.append(key)
    
    if not unavailable_datasets:
        print("\nAll datasets are already available locally.")
        return
    
    print("\nDatasets available for download:")
    for i, key in enumerate(unavailable_datasets, 1):
        info = DATASETS[key]
        print(f"  {i}. {key}: {info['name']} ({info['type']})")
    
    choice = input("\nDownload (a)ll missing datasets or (s)elect specific datasets? [a/s]: ").strip().lower()
    to_download = []
    
    if choice == 'a':
        to_download = unavailable_datasets
    else:
        for key in unavailable_datasets:
            info = DATASETS[key]
            yn = input(f"Download {key} ({info['name']})? [y/n]: ").strip().lower()
            if yn == 'y':
                to_download.append(key)
    
    if not to_download:
        print("No datasets selected for download. Exiting.")
        return
    
    for key in to_download:
        download_dataset(key)
    
    print("All selected datasets downloaded.")

if __name__ == "__main__":
    main() 
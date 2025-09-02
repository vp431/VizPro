"""
Similarity analysis functionality for finding similar examples in datasets.
"""
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.dataset_scanner import load_dataset_samples

logger = logging.getLogger(__name__)

def find_similar_examples(text, dataset_name, task_type='sentiment', max_samples=200, similarity_threshold=0.3, top_k=5):
    """
    Find similar examples to the given text in the dataset.
    
    Args:
        text: The input text to find similarities for
        dataset_name: Name of the dataset to search in
        task_type: Type of task ('sentiment' or 'ner')
        max_samples: Maximum number of samples to load from dataset
        similarity_threshold: Minimum similarity score to include
        top_k: Number of top similar examples to return
        
    Returns:
        List of similar examples with similarity scores
    """
    try:
        # Load dataset samples
        if dataset_name == 'IMDb':
            split = 'test'
        else:
            split = 'dev'
        samples = load_dataset_samples(dataset_name, task_type, split=split, max_samples=max_samples)
        
        if not samples:
            logger.warning(f"No samples found in dataset {dataset_name}")
            return []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Prepare texts for vectorization
        sample_texts = [text] + [sample['text'] for sample in samples]
        
        if len(sample_texts) <= 1:
            logger.warning("Not enough texts for similarity analysis")
            return []
        
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Get most similar examples
        similar_indices = similarities[0].argsort()[-top_k:][::-1]
        
        # Create results
        similar_examples = []
        for idx in similar_indices:
            similarity_score = similarities[0][idx]
            if similarity_score > similarity_threshold:
                similar_examples.append({
                    "text": sample_texts[idx + 1],
                    "similarity": similarity_score,
                    "original_data": samples[idx]  # Include original sample data
                })
        
        return similar_examples
        
    except Exception as e:
        logger.error(f"Error in similarity analysis: {str(e)}")
        return []

def update_similarity_analysis(point_data, selected_dataset, selected_model):
    """
    Update similarity analysis for a selected point.
    This function is called when a user clicks on a point in the error analysis plot.
    
    Args:
        point_data: Data of the selected point
        selected_dataset: Currently selected dataset
        selected_model: Currently selected model
        
    Returns:
        HTML content for similarity analysis results
    """
    if not point_data:
        return None
    
    try:
        text = point_data["text"]
        similar_examples = find_similar_examples(text, selected_dataset)
        
        if similar_examples:
            return {
                "original_text": text,
                "similar_examples": similar_examples,
                "dataset": selected_dataset,
                "total_found": len(similar_examples)
            }
        else:
            return {
                "original_text": text,
                "similar_examples": [],
                "dataset": selected_dataset,
                "total_found": 0,
                "message": "No similar examples found in the dataset."
            }
            
    except Exception as e:
        logger.error(f"Error in update_similarity_analysis: {str(e)}")
        return {
            "error": str(e),
            "original_text": point_data.get("text", ""),
            "dataset": selected_dataset
        }
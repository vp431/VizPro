"""
Data storage for analysis results to persist between user interactions.
Results are stored in memory and overwritten when user clicks "Analyze Dataset" again.
"""
import logging
from typing import Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)

class AnalysisStore:
    """
    Thread-safe storage for analysis results.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}
        
    def store_dataset_analysis(self, dataset_name: str, model_path: str, results: Dict[str, Any]):
        """
        Store dataset analysis results.
        
        Args:
            dataset_name: Name of the analyzed dataset
            model_path: Path of the model used for analysis
            results: Analysis results dictionary
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            self._data[key] = {
                "dataset_name": dataset_name,
                "model_path": model_path,
                "results": results,
                "timestamp": self._get_timestamp()
            }
            logger.info(f"Stored analysis results for {key}")
    
    def get_dataset_analysis(self, dataset_name: str, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get stored dataset analysis results.
        
        Args:
            dataset_name: Name of the dataset
            model_path: Path of the model
            
        Returns:
            Stored analysis results or None if not found
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            return self._data.get(key)
    
    def store_error_patterns(self, dataset_name: str, model_path: str, error_patterns: Dict[str, Any]):
        """
        Store error pattern analysis results.
        
        Args:
            dataset_name: Name of the analyzed dataset
            model_path: Path of the model used for analysis
            error_patterns: Error pattern analysis results
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            if key in self._data:
                self._data[key]["results"]["error_patterns"] = error_patterns
            else:
                self._data[key] = {
                    "dataset_name": dataset_name,
                    "model_path": model_path,
                    "results": {"error_patterns": error_patterns},
                    "timestamp": self._get_timestamp()
                }
            logger.info(f"Stored error patterns for {key}")
    
    def get_error_patterns(self, dataset_name: str, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get stored error pattern analysis results.
        
        Args:
            dataset_name: Name of the dataset
            model_path: Path of the model
            
        Returns:
            Stored error patterns or None if not found
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            data = self._data.get(key)
            if data and "error_patterns" in data["results"]:
                return data["results"]["error_patterns"]
            return None
    
    def store_similarity_analysis(self, dataset_name: str, model_path: str, similarity_results: Dict[str, Any]):
        """
        Store similarity analysis results.
        
        Args:
            dataset_name: Name of the analyzed dataset
            model_path: Path of the model used for analysis
            similarity_results: Similarity analysis results
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            if key in self._data:
                self._data[key]["results"]["similarity_analysis"] = similarity_results
            else:
                self._data[key] = {
                    "dataset_name": dataset_name,
                    "model_path": model_path,
                    "results": {"similarity_analysis": similarity_results},
                    "timestamp": self._get_timestamp()
                }
            logger.info(f"Stored similarity analysis for {key}")
    
    def get_similarity_analysis(self, dataset_name: str, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get stored similarity analysis results.
        
        Args:
            dataset_name: Name of the dataset
            model_path: Path of the model
            
        Returns:
            Stored similarity analysis or None if not found
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            data = self._data.get(key)
            if data and "similarity_analysis" in data["results"]:
                return data["results"]["similarity_analysis"]
            return None
    
    def clear_analysis(self, dataset_name: str, model_path: str):
        """
        Clear stored analysis for a specific dataset and model.
        Called when user clicks "Analyze Dataset" again.
        
        Args:
            dataset_name: Name of the dataset
            model_path: Path of the model
        """
        with self._lock:
            key = f"{dataset_name}_{model_path}"
            if key in self._data:
                del self._data[key]
                logger.info(f"Cleared analysis data for {key}")
    
    def clear_all(self):
        """Clear all stored analysis data."""
        with self._lock:
            self._data.clear()
            logger.info("Cleared all analysis data")
    
    def get_all_keys(self):
        """Get all stored analysis keys."""
        with self._lock:
            return list(self._data.keys())
    
    def _get_timestamp(self):
        """Get current timestamp."""
        import time
        return time.time()

# Global analysis store instance
analysis_store = AnalysisStore()
"""
Error analysis functionality for sentiment analysis models.
Moved from app.py for better organization.
"""
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)

def categorize_error_patterns(high_conf_errors):
    """
    Categorize high confidence errors into common error patterns using a more structured approach
    inspired by Azimuth's smart tags concept.
    
    Args:
        high_conf_errors: List of high confidence error examples
        
    Returns:
        Dictionary of error categories and counts
    """
    # Define error categories with detailed linguistic patterns
    error_categories = {
        "negation_errors": {
            "description": "Errors involving negation words that reverse sentiment",
            "examples": [],
            "patterns": {
                "words": ["not", "n't", "no", "never", "none", "nothing", "neither", "nor", "without", "lack", "absent"],
                "phrases": ["far from", "by no means", "not at all", "not even", "not much", "not very", "hardly", "barely"]
            }
        },
        "intensity_errors": {
            "description": "Errors involving intensity modifiers that amplify sentiment",
            "examples": [],
            "patterns": {
                "words": ["very", "extremely", "really", "absolutely", "completely", "totally", "utterly", 
                         "highly", "incredibly", "exceptionally", "too", "so", "quite", "rather"],
                "phrases": ["a lot", "to a great extent", "by far"]
            }
        },
        "context_errors": {
            "description": "Errors in longer sentences with complex context",
            "examples": [],
            "patterns": {
                "length_threshold": 15  # Words
            }
        },
        "comparison_errors": {
            "description": "Errors involving comparison or contrast between different sentiments",
            "examples": [],
            "patterns": {
                "words": ["but", "however", "although", "though", "despite", "in spite", "nevertheless", 
                         "nonetheless", "yet", "still", "while", "whereas", "unlike", "contrary"],
                "phrases": ["on the other hand", "even though", "in contrast", "as opposed to", "rather than"]
            }
        },
        "sarcasm_errors": {
            "description": "Errors potentially involving sarcasm or irony",
            "examples": [],
            "patterns": {
                "phrases": ["yeah right", "sure", "as if", "whatever", "oh great", "big deal", "wow", 
                           "oh joy", "bravo", "how nice", "just what I needed", "good luck with that"]
            }
        },
        "ambiguity_errors": {
            "description": "Errors involving words with ambiguous or context-dependent sentiment",
            "examples": [],
            "patterns": {
                "words": ["interesting", "surprising", "impressive", "remarkable", "notable", "unusual",
                         "different", "special", "particular", "certain", "fine", "okay", "ok"]
            }
        },
        "conditional_errors": {
            "description": "Errors involving conditional statements",
            "examples": [],
            "patterns": {
                "words": ["if", "would", "could", "should", "may", "might", "can", "will", "unless"],
                "phrases": ["as long as", "provided that", "assuming that", "in case"]
            }
        },
        "other_errors": {
            "description": "Errors that don't fit into other categories",
            "examples": []
        }
    }
    
    # Function to check if a sentence contains any pattern from a list
    def contains_pattern(sentence, patterns):
        sentence = sentence.lower()
        words = sentence.split()
        
        # Check for individual words
        if "words" in patterns:
            if any(word in words or f" {word} " in sentence for word in patterns["words"]):
                return True
                
        # Check for phrases
        if "phrases" in patterns:
            if any(phrase in sentence for phrase in patterns["phrases"]):
                return True
                
        return False
    
    # Categorize each error
    for error in high_conf_errors:
        sentence = error["text"].lower()
        categorized = False
        
        # Check for negation errors
        if contains_pattern(sentence, error_categories["negation_errors"]["patterns"]):
            error_categories["negation_errors"]["examples"].append(error)
            categorized = True
            
        # Check for comparison errors (if not already categorized)
        elif contains_pattern(sentence, error_categories["comparison_errors"]["patterns"]):
            error_categories["comparison_errors"]["examples"].append(error)
            categorized = True
            
        # Check for intensity errors
        elif contains_pattern(sentence, error_categories["intensity_errors"]["patterns"]):
            error_categories["intensity_errors"]["examples"].append(error)
            categorized = True
            
        # Check for sarcasm errors
        elif contains_pattern(sentence, error_categories["sarcasm_errors"]["patterns"]):
            error_categories["sarcasm_errors"]["examples"].append(error)
            categorized = True
            
        # Check for ambiguity errors
        elif contains_pattern(sentence, error_categories["ambiguity_errors"]["patterns"]):
            error_categories["ambiguity_errors"]["examples"].append(error)
            categorized = True
            
        # Check for conditional errors
        elif contains_pattern(sentence, error_categories["conditional_errors"]["patterns"]):
            error_categories["conditional_errors"]["examples"].append(error)
            categorized = True
            
        # Check for context errors (longer sentences)
        elif len(sentence.split()) > error_categories["context_errors"]["patterns"]["length_threshold"]:
            error_categories["context_errors"]["examples"].append(error)
            categorized = True
            
        # Other errors
        if not categorized:
            error_categories["other_errors"]["examples"].append(error)
    
    # Create a simplified version for the return value
    result = {}
    for category, data in error_categories.items():
        result[category] = {
            "description": data["description"],
            "examples": data["examples"],
            "count": len(data["examples"])
        }
    
    return result
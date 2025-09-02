"""
Counterfactual analysis functionality for generating what-if scenarios.
"""
import logging
import re
import random
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import wordnet
from models.api import model_api

logger = logging.getLogger(__name__)

# Download required NLTK data if not present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class CounterfactualGenerator:
    """Generate counterfactual examples using various text modification strategies."""
    
    def __init__(self):
        # Common sentiment words for replacement
        self.positive_words = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'nice', 
            'beautiful', 'perfect', 'outstanding', 'brilliant', 'superb', 'marvelous',
            'delightful', 'impressive', 'remarkable', 'exceptional', 'magnificent'
        ]
        
        self.negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing', 'dreadful',
            'disgusting', 'pathetic', 'useless', 'worthless', 'appalling', 'atrocious',
            'abysmal', 'deplorable', 'miserable', 'wretched', 'inferior'
        ]
        
        # Intensity modifiers
        self.intensity_amplifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely']
        self.intensity_diminishers = ['somewhat', 'rather', 'quite', 'fairly', 'slightly', 'moderately']
        
        # Negation words
        self.negation_words = ['not', 'never', 'no', 'nothing', 'none', 'neither', 'nor']

    def generate_counterfactuals(self, text: str, original_prediction: str, max_examples: int = 10) -> List[Dict[str, Any]]:
        """
        Generate counterfactual examples using multiple strategies.
        
        Args:
            text: Original text
            original_prediction: Original model prediction
            max_examples: Maximum number of counterfactuals to generate
            
        Returns:
            List of counterfactual examples with metadata
        """
        counterfactuals = []
        
        # Strategy 1: Synonym/Antonym replacement
        counterfactuals.extend(self._generate_synonym_antonym_replacements(text, original_prediction))
        
        # Strategy 2: Negation injection/removal
        counterfactuals.extend(self._generate_negation_modifications(text, original_prediction))
        
        # Strategy 3: Sentiment word injection
        counterfactuals.extend(self._generate_sentiment_injections(text, original_prediction))
        
        # Strategy 4: Intensity modifier changes
        counterfactuals.extend(self._generate_intensity_modifications(text, original_prediction))
        
        # Strategy 5: Word order changes
        counterfactuals.extend(self._generate_word_order_changes(text, original_prediction))
        
        # Remove duplicates and limit results
        seen_texts = set()
        unique_counterfactuals = []
        for cf in counterfactuals:
            if cf['text'] not in seen_texts and cf['text'] != text:
                seen_texts.add(cf['text'])
                unique_counterfactuals.append(cf)
                if len(unique_counterfactuals) >= max_examples:
                    break
        
        return unique_counterfactuals

    def _generate_synonym_antonym_replacements(self, text: str, original_prediction: str) -> List[Dict[str, Any]]:
        """Generate counterfactuals by replacing words with synonyms/antonyms."""
        counterfactuals = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Clean word for WordNet lookup
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Get synonyms and antonyms
            synonyms, antonyms = self._get_synonyms_antonyms(clean_word)
            
            # Try antonyms first (more likely to flip sentiment)
            for antonym in antonyms[:2]:  # Limit to 2 antonyms per word
                new_words = words.copy()
                new_words[i] = antonym
                new_text = ' '.join(new_words)
                
                counterfactuals.append({
                    'text': new_text,
                    'strategy': 'antonyms',
                    'change': f'"{word}" → "{antonym}"',
                    'original_word': word,
                    'replacement_word': antonym,
                    'position': i
                })
            
            # Try synonyms (less likely to flip but might change confidence)
            for synonym in synonyms[:1]:  # Limit to 1 synonym per word
                new_words = words.copy()
                new_words[i] = synonym
                new_text = ' '.join(new_words)
                
                counterfactuals.append({
                    'text': new_text,
                    'strategy': 'synonyms',
                    'change': f'"{word}" → "{synonym}"',
                    'original_word': word,
                    'replacement_word': synonym,
                    'position': i
                })
        
        return counterfactuals

    def _generate_negation_modifications(self, text: str, original_prediction: str) -> List[Dict[str, Any]]:
        """Generate counterfactuals by adding/removing negations."""
        counterfactuals = []
        
        # Strategy 1: Add negation before adjectives/verbs
        words = text.split()
        for i, word in enumerate(words):
            if self._is_sentiment_word(word.lower()):
                # Add "not" before sentiment words
                new_words = words.copy()
                new_words.insert(i, 'not')
                new_text = ' '.join(new_words)
                
                counterfactuals.append({
                    'text': new_text,
                    'strategy': 'negation',
                    'change': f'Added "not" before "{word}"',
                    'original_word': word,
                    'replacement_word': f'not {word}',
                    'position': i
                })
        
        # Strategy 2: Remove existing negations
        for neg_word in self.negation_words:
            if neg_word in text.lower():
                new_text = re.sub(r'\b' + neg_word + r'\b', '', text, flags=re.IGNORECASE)
                new_text = re.sub(r'\s+', ' ', new_text).strip()  # Clean up extra spaces
                
                if new_text != text:
                    counterfactuals.append({
                        'text': new_text,
                        'strategy': 'negation',
                        'change': f'Removed "{neg_word}"',
                        'original_word': neg_word,
                        'replacement_word': '',
                        'position': -1
                    })
        
        return counterfactuals

    def _generate_sentiment_injections(self, text: str, original_prediction: str) -> List[Dict[str, Any]]:
        """Generate counterfactuals by injecting sentiment words."""
        counterfactuals = []
        
        # Determine opposite sentiment words to inject
        if original_prediction.lower() in ['positive', '1', 'pos']:
            target_words = self.negative_words
            strategy_name = 'sentiment_injection'
        else:
            target_words = self.positive_words
            strategy_name = 'sentiment_injection'
        
        # Inject at the beginning
        for word in target_words[:3]:
            new_text = f"{word} {text}"
            counterfactuals.append({
                'text': new_text,
                'strategy': strategy_name,
                'change': f'Added "{word}" at beginning',
                'original_word': '',
                'replacement_word': word,
                'position': 0
            })
        
        # Inject at the end
        for word in target_words[3:6]:
            new_text = f"{text} {word}"
            counterfactuals.append({
                'text': new_text,
                'strategy': strategy_name,
                'change': f'Added "{word}" at end',
                'original_word': '',
                'replacement_word': word,
                'position': -1
            })
        
        return counterfactuals

    def _generate_intensity_modifications(self, text: str, original_prediction: str) -> List[Dict[str, Any]]:
        """Generate counterfactuals by modifying intensity."""
        counterfactuals = []
        words = text.split()
        
        for i, word in enumerate(words):
            if self._is_sentiment_word(word.lower()):
                # Add intensity amplifiers
                for amplifier in self.intensity_amplifiers[:2]:
                    new_words = words.copy()
                    new_words.insert(i, amplifier)
                    new_text = ' '.join(new_words)
                    
                    counterfactuals.append({
                        'text': new_text,
                        'strategy': 'intensity_modifiers',
                        'change': f'Added "{amplifier}" before "{word}"',
                        'original_word': word,
                        'replacement_word': f'{amplifier} {word}',
                        'position': i
                    })
                
                # Add intensity diminishers
                for diminisher in self.intensity_diminishers[:2]:
                    new_words = words.copy()
                    new_words.insert(i, diminisher)
                    new_text = ' '.join(new_words)
                    
                    counterfactuals.append({
                        'text': new_text,
                        'strategy': 'intensity_modifiers',
                        'change': f'Added "{diminisher}" before "{word}"',
                        'original_word': word,
                        'replacement_word': f'{diminisher} {word}',
                        'position': i
                    })
        
        return counterfactuals

    def _generate_word_order_changes(self, text: str, original_prediction: str) -> List[Dict[str, Any]]:
        """Generate counterfactuals by changing word order."""
        counterfactuals = []
        words = text.split()
        
        if len(words) >= 4:  # Only for longer sentences
            # Swap adjacent words
            for i in range(len(words) - 1):
                new_words = words.copy()
                new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]
                new_text = ' '.join(new_words)
                
                counterfactuals.append({
                    'text': new_text,
                    'strategy': 'word_order',
                    'change': f'Swapped "{words[i]}" and "{words[i+1]}"',
                    'original_word': f'{words[i]} {words[i+1]}',
                    'replacement_word': f'{words[i+1]} {words[i]}',
                    'position': i
                })
        
        return counterfactuals

    def _get_synonyms_antonyms(self, word: str) -> Tuple[List[str], List[str]]:
        """Get synonyms and antonyms for a word using WordNet."""
        synonyms = set()
        antonyms = set()
        
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
                    
                    # Get antonyms
                    for antonym in lemma.antonyms():
                        antonyms.add(antonym.name().replace('_', ' '))
        except Exception as e:
            logger.warning(f"Error getting synonyms/antonyms for '{word}': {e}")
        
        # Remove the original word from synonyms
        synonyms.discard(word)
        
        return list(synonyms)[:5], list(antonyms)[:5]  # Limit results

    def _is_sentiment_word(self, word: str) -> bool:
        """Check if a word is likely to carry sentiment."""
        word = word.lower()
        return (word in self.positive_words or 
                word in self.negative_words or
                any(word in synset.name() for synset in wordnet.synsets(word) 
                    if synset.pos() in ['a', 's']))  # Adjectives and satellite adjectives

def analyze_counterfactuals(text: str, original_prediction: str, confidence: float) -> Dict[str, Any]:
    """
    Perform counterfactual analysis on a given text.
    
    Args:
        text: Original text to analyze
        original_prediction: Original model prediction
        confidence: Original prediction confidence
        
    Returns:
        Dictionary containing counterfactual analysis results
    """
    try:
        generator = CounterfactualGenerator()
        counterfactuals = generator.generate_counterfactuals(text, original_prediction)
        
        # Get predictions for counterfactuals
        results = []
        flip_count = 0
        
        for cf in counterfactuals:
            try:
                # Get model prediction for counterfactual text
                prediction_result = model_api.get_sentiment(cf['text'])
                
                if prediction_result:
                    predicted_class = prediction_result.get('label', 'Unknown')
                    pred_confidence = prediction_result.get('score', 0.0)
                    
                    # Check if prediction flipped
                    flipped = predicted_class != original_prediction
                    if flipped:
                        flip_count += 1
                    
                    results.append({
                        'text': cf['text'],
                        'strategy': cf['strategy'],
                        'change': cf['change'],
                        'original_prediction': original_prediction,
                        'new_prediction': predicted_class,
                        'original_confidence': confidence,
                        'new_confidence': pred_confidence,
                        'flipped': flipped,
                        'confidence_change': pred_confidence - confidence
                    })
                    
            except Exception as e:
                logger.error(f"Error getting prediction for counterfactual: {e}")
                continue
        
        # Calculate statistics
        total_generated = len(results)
        flip_rate = (flip_count / total_generated * 100) if total_generated > 0 else 0
        
        # Group by strategy
        strategy_stats = {}
        for result in results:
            strategy = result['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'total': 0, 'flips': 0}
            strategy_stats[strategy]['total'] += 1
            if result['flipped']:
                strategy_stats[strategy]['flips'] += 1
        
        # Calculate success rates for each strategy
        for strategy in strategy_stats:
            stats = strategy_stats[strategy]
            stats['success_rate'] = (stats['flips'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Find best strategy
        best_strategy = max(strategy_stats.items(), 
                          key=lambda x: x[1]['success_rate']) if strategy_stats else ('none', {'success_rate': 0})
        
        return {
            'original_text': text,
            'original_prediction': original_prediction,
            'original_confidence': confidence,
            'counterfactuals': results,
            'statistics': {
                'total_generated': total_generated,
                'successful_flips': flip_count,
                'flip_rate': flip_rate,
                'strategy_breakdown': strategy_stats,
                'best_strategy': best_strategy[0],
                'best_strategy_rate': best_strategy[1]['success_rate']
            }
        }
        
    except Exception as e:
        logger.error(f"Error in counterfactual analysis: {e}")
        return {
            'error': str(e),
            'original_text': text,
            'original_prediction': original_prediction,
            'counterfactuals': [],
            'statistics': {
                'total_generated': 0,
                'successful_flips': 0,
                'flip_rate': 0,
                'strategy_breakdown': {},
                'best_strategy': 'none',
                'best_strategy_rate': 0
            }
        }
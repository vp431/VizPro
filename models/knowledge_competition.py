"""
Knowledge Competition Analysis for Factual vs Counterfactual Information.
Analyzes how transformer models handle competing factual and counterfactual claims.
Based on research into knowledge editing and fact vs counterfact competition.
"""
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import re

logger = logging.getLogger(__name__)

class KnowledgeCompetitionAnalyzer:
    """
    Analyze how transformer models handle competing factual vs counterfactual information.
    """
    
    def __init__(self):
        self.fact_templates = {
            "capital": "The capital of {country} is {city}.",
            "president": "The president of {country} is {person}.",
            "currency": "The currency of {country} is {currency}.",
            "language": "The official language of {country} is {language}.",
            "continent": "{country} is located in {continent}.",
            "invention": "{item} was invented by {person}.",
            "discovery": "{item} was discovered by {person}.",
            "author": "{book} was written by {author}.",
            "director": "{movie} was directed by {director}.",
            "company": "{product} is made by {company}."
        }
        
        # Example fact pairs for testing
        self.example_facts = {
            "capital": [
                ("France", "Paris", "Italy", "Rome"),
                ("Germany", "Berlin", "Spain", "Madrid"), 
                ("Japan", "Tokyo", "Korea", "Seoul"),
                ("Australia", "Canberra", "Canada", "Ottawa"),
                ("Brazil", "Brasília", "Argentina", "Buenos Aires")
            ],
            "company": [
                ("iPhone", "Apple", "Galaxy", "Samsung"),
                ("Windows", "Microsoft", "macOS", "Apple"),
                ("Tesla", "Tesla", "Prius", "Toyota"),
                ("PlayStation", "Sony", "Xbox", "Microsoft")
            ]
        }
    
    def analyze_fact_counterfact_competition(self, fact_text: str, counterfact_text: str, model_api=None) -> Dict[str, Any]:
        """
        Analyze competition between factual and counterfactual statements.
        
        Args:
            fact_text: Factual statement (e.g., "The capital of France is Paris")
            counterfact_text: Counterfactual statement (e.g., "The capital of Italy is Paris")
            
        Returns:
            Dictionary containing competition analysis results
        """
        try:
            # Check if model_api is provided
            if model_api is None:
                return {"error": "Model API not provided. Please ensure a model is loaded."}
            
            # Get attention weights for both statements
            fact_attention = model_api.get_attention_weights(fact_text)
            counterfact_attention = model_api.get_attention_weights(counterfact_text)
            
            if not fact_attention or not counterfact_attention:
                logger.error("Failed to get attention weights for fact/counterfact analysis")
                return {"error": "Failed to get attention weights. The selected model may not support attention extraction or may not be properly loaded."}
            
            # Validate attention data structure
            fact_tokens = fact_attention.get("tokens", [])
            counterfact_tokens = counterfact_attention.get("tokens", [])
            fact_attentions = fact_attention.get("attention_weights", [])
            counterfact_attentions = counterfact_attention.get("attention_weights", [])
            
            if not fact_tokens or not counterfact_tokens:
                return {"error": "No tokens found in attention analysis. Please check your input text."}
            
            if not fact_attentions or not counterfact_attentions:
                return {"error": "No attention weights found. The model may not support attention extraction for this task."}
            
            # Check if attention matrices are empty
            if len(fact_attentions) == 0 or len(counterfact_attentions) == 0:
                return {"error": "Empty attention matrices. The model may not be compatible with this analysis."}
            
            # Convert to numpy arrays with error handling
            try:
                fact_attentions = [np.array(layer) for layer in fact_attentions if len(layer) > 0]
                counterfact_attentions = [np.array(layer) for layer in counterfact_attentions if len(layer) > 0]
                
                # Final check for valid data
                if len(fact_attentions) == 0 or len(counterfact_attentions) == 0:
                    return {"error": "No valid attention layers found after processing."}
                    
                # Check if arrays have proper dimensions
                for i, layer in enumerate(fact_attentions):
                    if layer.size == 0:
                        return {"error": f"Empty attention layer {i} found in factual analysis."}
                        
                for i, layer in enumerate(counterfact_attentions):
                    if layer.size == 0:
                        return {"error": f"Empty attention layer {i} found in counterfactual analysis."}
                        
            except Exception as conv_error:
                return {"error": f"Error converting attention data: {str(conv_error)}"}
            
            # Analyze layer-wise competition
            layerwise_competition = self._analyze_layerwise_competition(
                fact_attentions, counterfact_attentions, fact_tokens, counterfact_tokens
            )
            
            # Analyze information flow (trails)
            fact_trails = self._extract_information_trails(fact_attentions, fact_tokens)
            counterfact_trails = self._extract_information_trails(counterfact_attentions, counterfact_tokens)
            
            # Calculate difference maps
            difference_maps = self._calculate_difference_maps(fact_attentions, counterfact_attentions)
            
            # Identify competing information pathways
            competing_pathways = self._identify_competing_pathways(
                fact_trails, counterfact_trails, fact_tokens, counterfact_tokens
            )
            
            return {
                "fact_text": fact_text,
                "counterfact_text": counterfact_text,
                "fact_tokens": fact_tokens,
                "counterfact_tokens": counterfact_tokens,
                "layerwise_competition": layerwise_competition,
                "fact_trails": fact_trails,
                "counterfact_trails": counterfact_trails,
                "difference_maps": difference_maps,
                "competing_pathways": competing_pathways,
                "metadata": {
                    "num_layers": len(fact_attentions),
                    "num_heads": fact_attentions[0].shape[1] if fact_attentions else 0,
                    "fact_length": len(fact_tokens),
                    "counterfact_length": len(counterfact_tokens)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fact-counterfact competition analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_layerwise_competition(self, fact_attentions: List[np.ndarray], 
                                     counterfact_attentions: List[np.ndarray],
                                     fact_tokens: List[str], counterfact_tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze competition at each layer and head.
        
        Returns heatmap data showing factual vs counterfactual dominance.
        """
        num_layers = len(fact_attentions)
        
        # Determine number of heads based on tensor shape
        if fact_attentions:
            first_layer = fact_attentions[0]
            if len(first_layer.shape) == 4:  # batch, heads, seq, seq
                num_heads = first_layer.shape[1]
            elif len(first_layer.shape) == 3:  # heads, seq, seq (QA models)
                num_heads = first_layer.shape[0]
            else:
                num_heads = 1  # Fallback for unexpected shapes
        else:
            num_heads = 0
        
        # Initialize competition matrix (layers x heads)
        competition_matrix = np.zeros((num_layers, num_heads))
        
        for layer_idx in range(num_layers):
            try:
                fact_layer = fact_attentions[layer_idx]
                counterfact_layer = counterfact_attentions[layer_idx]
                
                # Handle tensor shapes for different model types
                
                # Handle different tensor dimensions - QA models return (heads, seq, seq)
                if len(fact_layer.shape) == 4:  # batch, heads, seq, seq
                    fact_layer = fact_layer[0]  # First batch
                    counterfact_layer = counterfact_layer[0]
                elif len(fact_layer.shape) == 2:  # seq, seq (single head)
                    fact_layer = fact_layer.reshape(1, *fact_layer.shape)  # Add head dimension
                    counterfact_layer = counterfact_layer.reshape(1, *counterfact_layer.shape)
                # 3D case (heads, seq, seq) - this is what QA models return
                elif len(fact_layer.shape) == 3:
                    # Already in correct format (heads, seq, seq)
                    pass
                
                # Ensure we have at least 3D tensors
                if len(fact_layer.shape) < 3:
                    logger.warning(f"Unexpected tensor shape in layer {layer_idx}: {fact_layer.shape}")
                    competition_matrix[layer_idx, :] = 0
                    continue
                
                current_num_heads = min(fact_layer.shape[0], num_heads)
                
                for head_idx in range(current_num_heads):
                    try:
                        # Calculate attention entropy for each head with safety checks
                        fact_head = fact_layer[head_idx] if len(fact_layer.shape) > 2 else fact_layer
                        counterfact_head = counterfact_layer[head_idx] if len(counterfact_layer.shape) > 2 else counterfact_layer
                        
                        if fact_head.size == 0 or counterfact_head.size == 0:
                            competition_matrix[layer_idx, head_idx] = 0
                            continue
                            
                        fact_entropy = self._calculate_attention_entropy(fact_head)
                        counterfact_entropy = self._calculate_attention_entropy(counterfact_head)
                        
                        # Calculate dominance score: positive = fact wins, negative = counterfact wins
                        if fact_entropy + counterfact_entropy > 0:
                            dominance = (fact_entropy - counterfact_entropy) / (fact_entropy + counterfact_entropy)
                        else:
                            dominance = 0
                        
                        competition_matrix[layer_idx, head_idx] = dominance
                        
                    except Exception as head_error:
                        logger.warning(f"Error processing head {head_idx} in layer {layer_idx}: {head_error}")
                        competition_matrix[layer_idx, head_idx] = 0
                        
            except Exception as layer_error:
                logger.warning(f"Error processing layer {layer_idx}: {layer_error}")
                # Fill the layer with zeros
                if layer_idx < competition_matrix.shape[0]:
                    competition_matrix[layer_idx, :] = 0
        
        return {
            "competition_matrix": competition_matrix.tolist(),
            "layer_labels": [f"Layer {i+1}" for i in range(num_layers)],
            "head_labels": [f"Head {i+1}" for i in range(num_heads)],
            "colorscale_info": {
                "description": "Blue = Fact Dominant, Red = Counterfact Dominant",
                "range": [-1, 1]
            }
        }
    
    def _extract_information_trails(self, attentions: List[np.ndarray], tokens: List[str]) -> Dict[str, Any]:
        """
        Extract information flow trails through the network.
        """
        num_layers = len(attentions)
        trails = []
        
        # Find key tokens (subject, object, relation)
        key_positions = self._identify_key_token_positions(tokens)
        
        for layer_idx in range(num_layers):
            layer_attention = attentions[layer_idx]
            
            # Handle different tensor shapes
            if len(layer_attention.shape) == 4:  # batch, heads, seq, seq
                layer_attention = layer_attention[0]  # First batch
            elif len(layer_attention.shape) == 3:  # heads, seq, seq (QA models)
                # Already in correct format
                pass
            else:
                logger.warning(f"Unexpected attention shape in layer {layer_idx}: {layer_attention.shape}")
                continue
                
            layer_trails = []
            
            # For each head in this layer
            for head_idx in range(layer_attention.shape[0]):
                head_attention = layer_attention[head_idx]
                
                # Extract strong attention flows (> threshold)
                threshold = 0.1
                strong_flows = []
                
                for i in range(len(tokens)):
                    for j in range(len(tokens)):
                        if head_attention[i, j] > threshold:
                            strong_flows.append({
                                "from_token": tokens[i],
                                "to_token": tokens[j],
                                "from_pos": i,
                                "to_pos": j,
                                "strength": float(head_attention[i, j]),
                                "involves_key_token": i in key_positions or j in key_positions
                            })
                
                layer_trails.append({
                    "head": head_idx,
                    "flows": strong_flows
                })
            
            trails.append({
                "layer": layer_idx,
                "heads": layer_trails
            })
        
        return {
            "trails": trails,
            "key_positions": key_positions,
            "tokens": tokens
        }
    
    def _calculate_difference_maps(self, fact_attentions: List[np.ndarray], 
                                 counterfact_attentions: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate difference maps showing where fact and counterfact diverge.
        """
        num_layers = len(fact_attentions)
        layer_differences = []
        
        for layer_idx in range(num_layers):
            fact_layer = fact_attentions[layer_idx]
            counterfact_layer = counterfact_attentions[layer_idx]
            
            # Handle different tensor shapes
            if len(fact_layer.shape) == 4:  # batch, heads, seq, seq
                fact_layer = fact_layer[0]  # First batch
                counterfact_layer = counterfact_layer[0]
            elif len(fact_layer.shape) == 3:  # heads, seq, seq (QA models)
                # Already in correct format
                pass
            else:
                logger.warning(f"Unexpected attention shape in difference calculation layer {layer_idx}: {fact_layer.shape}")
                continue
            
            # Calculate mean attention per head
            fact_means = np.mean(fact_layer, axis=(1, 2))  # Mean across tokens
            counterfact_means = np.mean(counterfact_layer, axis=(1, 2))
            
            # Calculate difference (fact - counterfact)
            difference = fact_means - counterfact_means
            
            layer_differences.append({
                "layer": layer_idx,
                "fact_attention": fact_means.tolist(),
                "counterfact_attention": counterfact_means.tolist(),
                "difference": difference.tolist()
            })
        
        return {
            "layer_differences": layer_differences,
            "summary": {
                "total_divergence": float(np.sum([np.abs(ld["difference"]).sum() for ld in layer_differences])),
                "max_divergence_layer": int(np.argmax([np.abs(ld["difference"]).max() for ld in layer_differences])),
                "divergence_trend": [float(np.abs(ld["difference"]).mean()) for ld in layer_differences]
            }
        }
    
    def _identify_competing_pathways(self, fact_trails: Dict, counterfact_trails: Dict,
                                   fact_tokens: List[str], counterfact_tokens: List[str]) -> Dict[str, Any]:
        """
        Identify where factual and counterfactual information pathways compete.
        """
        # Find common tokens between fact and counterfact
        fact_set = set(fact_tokens)
        counterfact_set = set(counterfact_tokens)
        common_tokens = fact_set.intersection(counterfact_set)
        different_tokens = fact_set.symmetric_difference(counterfact_set)
        
        # Analyze pathway competition for common tokens
        pathway_competition = []
        
        for token in common_tokens:
            if token in [".", ",", "the", "is", "of"]:  # Skip common words
                continue
                
            fact_pathways = self._get_token_pathways(token, fact_trails, fact_tokens)
            counterfact_pathways = self._get_token_pathways(token, counterfact_trails, counterfact_tokens)
            
            pathway_competition.append({
                "token": token,
                "fact_pathways": fact_pathways,
                "counterfact_pathways": counterfact_pathways,
                "competition_score": self._calculate_pathway_competition_score(fact_pathways, counterfact_pathways)
            })
        
        return {
            "common_tokens": list(common_tokens),
            "different_tokens": list(different_tokens),
            "pathway_competition": pathway_competition,
            "competition_summary": {
                "high_competition_tokens": [pc["token"] for pc in pathway_competition if pc["competition_score"] > 0.5],
                "avg_competition": float(np.mean([pc["competition_score"] for pc in pathway_competition]) if pathway_competition else 0)
            }
        }
    
    def _identify_key_token_positions(self, tokens: List[str]) -> List[int]:
        """Identify positions of key tokens (nouns, entities, etc.)."""
        key_positions = []
        
        # Simple heuristic: identify likely entities and important words
        for i, token in enumerate(tokens):
            if token.istitle() or len(token) > 3:  # Capitalized or longer words
                if token not in ["The", "Is", "Was", "Are", "Were", "By", "In", "Of", "To"]:
                    key_positions.append(i)
        
        return key_positions
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        try:
            # Handle empty or invalid matrices
            if attention_matrix.size == 0:
                return 0.0
            
            # Flatten and normalize
            attention_flat = attention_matrix.flatten()
            
            # Check for all-zero matrices
            if np.all(attention_flat == 0):
                return 0.0
            
            attention_flat = attention_flat + 1e-10  # Avoid log(0)
            sum_attention = attention_flat.sum()
            
            # Handle case where sum is zero or very small
            if sum_attention <= 1e-10:
                return 0.0
                
            attention_flat = attention_flat / sum_attention
            
            # Calculate entropy with safety check
            log_attention = np.log(attention_flat)
            if np.any(np.isnan(log_attention)) or np.any(np.isinf(log_attention)):
                return 0.0
                
            entropy = -np.sum(attention_flat * log_attention)
            
            # Ensure entropy is a valid number
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.0
                
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"Error calculating attention entropy: {e}")
            return 0.0
    
    def _get_token_pathways(self, token: str, trails: Dict, tokens: List[str]) -> List[Dict]:
        """Get attention pathways involving a specific token."""
        token_positions = [i for i, t in enumerate(tokens) if t == token]
        pathways = []
        
        for trail in trails["trails"]:
            for head in trail["heads"]:
                for flow in head["flows"]:
                    if flow["from_pos"] in token_positions or flow["to_pos"] in token_positions:
                        pathways.append({
                            "layer": trail["layer"],
                            "head": head["head"],
                            "flow": flow
                        })
        
        return pathways
    
    def _calculate_pathway_competition_score(self, fact_pathways: List, counterfact_pathways: List) -> float:
        """Calculate how much the pathways compete (higher = more competition)."""
        if not fact_pathways or not counterfact_pathways:
            return 0.0
        
        # Simple scoring based on pathway differences
        fact_strength = sum(p["flow"]["strength"] for p in fact_pathways)
        counterfact_strength = sum(p["flow"]["strength"] for p in counterfact_pathways)
        
        total_strength = fact_strength + counterfact_strength
        if total_strength == 0:
            return 0.0
        
        # Competition score: higher when strengths are more balanced
        competition = 1.0 - abs(fact_strength - counterfact_strength) / total_strength
        return competition
    
    def generate_fact_counterfact_pairs(self, template_type: str = "capital", num_pairs: int = 5) -> List[Tuple[str, str]]:
        """
        Generate fact-counterfact pairs for analysis.
        
        Args:
            template_type: Type of fact template to use
            num_pairs: Number of pairs to generate
            
        Returns:
            List of (fact, counterfact) text pairs
        """
        if template_type not in self.fact_templates:
            template_type = "capital"
        
        template = self.fact_templates[template_type]
        examples = self.example_facts.get(template_type, [])
        
        pairs = []
        for example in examples[:num_pairs]:
            if template_type == "capital":
                country1, city1, country2, city2 = example
                fact = template.format(country=country1, city=city1)
                counterfact = template.format(country=country2, city=city1)  # Same city, different country
            elif template_type == "company":
                product1, company1, product2, company2 = example
                fact = template.format(product=product1, company=company1)
                counterfact = template.format(product=product1, company=company2)  # Same product, different company
            else:
                continue
                
            pairs.append((fact, counterfact))
        
        return pairs

# Global analyzer instance
knowledge_competition_analyzer = KnowledgeCompetitionAnalyzer()

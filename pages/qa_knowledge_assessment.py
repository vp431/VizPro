"""
QA Knowledge Assessment Page
Analyzes whether the QA model "knows" the answer by testing consistency across multiple runs.
Uses embedding similarity analysis to determine knowledge vs guessing patterns.
"""
import logging
import numpy as np
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict, Any

from models.api import model_api

logger = logging.getLogger(__name__)

# Global sentence transformer for answer embedding
_sentence_transformer = None

def get_sentence_transformer():
    """Get or create sentence transformer for answer embeddings."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully loaded SentenceTransformer")
        except ImportError:
            logger.warning("sentence_transformers not available, will use TF-IDF fallback")
            _sentence_transformer = "fallback"
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            _sentence_transformer = "fallback"
    return _sentence_transformer

def create_layout(default_context="", default_question=""):
    """Create the QA knowledge assessment layout."""
    
    return html.Div([
        # Header section
        html.Div([
            html.H3([
                html.I(className="fas fa-brain me-3", style={"color": "#6c5ce7"}),
                "QA Knowledge Assessment"
            ], className="knowledge-header-title"),
            
            html.P([
                "Analyze whether the QA model truly 'knows' the answer by testing consistency across multiple predictions. ",
                "High embedding similarity indicates knowledge; scattered results suggest guessing."
            ], className="knowledge-header-subtitle")
        ], className="knowledge-header"),
        
        # Input section with multiline configuration
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-cogs me-2"),
                html.Span("Analysis Configuration", className="fw-bold"),
                html.Small(" Configure your knowledge assessment parameters", className="text-muted ms-2")
            ], className="config-header"),
            dbc.CardBody([
                # Compact 3-line configuration
                html.Div([
                    # Line 1: Context and Question inputs
                    dbc.Row([
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-file-text me-2"),
                                "Context"
                            ], className="config-compact-label"),
                            dbc.Textarea(
                                id="qa-knowledge-context-input",
                                placeholder="Enter the context for QA analysis...",
                                value=default_context,
                                rows=3,
                                className="config-compact-textarea"
                            )
                        ], width=7),
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-question-circle me-2"),
                                "Question"
                            ], className="config-compact-label"),
                            dbc.Input(
                                id="qa-knowledge-question-input",
                                placeholder="What question should the model answer?",
                                value=default_question,
                                className="config-compact-input mb-2"
                            ),
                            html.Small("This question will be asked multiple times to test consistency", 
                                     className="text-muted")
                        ], width=5)
                    ], className="mb-3"),
                    
                    # Line 2: Configuration parameters
                    dbc.Row([
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-repeat me-1"),
                                "Runs"
                            ], className="config-compact-label"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="qa-knowledge-num-runs",
                                    type="number",
                                    value=10,
                                    min=3,
                                    max=50,
                                    className="config-compact-number"
                                ),
                                dbc.InputGroupText("times", className="config-compact-addon")
                            ])
                        ], width=3),
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-brain me-1"),
                                "Embedding"
                            ], className="config-compact-label"),
                            dbc.Select(
                                id="qa-knowledge-embedding-type",
                                options=[
                                    {"label": "📝 Text Embeddings", "value": "text"},
                                    {"label": "🧠 Hidden States", "value": "hidden"}
                                ],
                                value="text",
                                className="config-compact-select"
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-project-diagram me-1"),
                                "Visualization"
                            ], className="config-compact-label"),
                            dbc.Select(
                                id="qa-knowledge-reduction-method",
                                options=[
                                    {"label": "🔍 t-SNE", "value": "tsne"},
                                    {"label": "📊 PCA", "value": "pca"},
                                    {"label": "🎯 Both", "value": "both"}
                                ],
                                value="both",
                                className="config-compact-select"
                            )
                        ], width=5)
                    ], className="mb-3"),
                    
                    # Line 3: Action buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-rocket me-2"),
                                "Run Knowledge Assessment"
                            ], id="run-qa-knowledge-btn", className="config-compact-primary-btn w-100")
                        ], width=8),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-lightbulb me-2"),
                                "Example"
                            ], id="load-qa-knowledge-example-btn", className="config-compact-secondary-btn w-100")
                        ], width=4)
                    ])
                ], className="config-compact-content")
            ], className="config-card-body")
        ], className="knowledge-input-card mb-4"),
        
        # Results area
        html.Div(id="qa-knowledge-results", className="knowledge-results-area"),
        
        # Loading overlay
        dcc.Loading(
            id="qa-knowledge-loading",
            type="default",
            children=html.Div(id="qa-knowledge-loading-output"),
            className="knowledge-loading"
        )
    ], className="qa-knowledge-assessment-container")

def run_knowledge_assessment(context: str, question: str, num_runs: int, embedding_type: str) -> Dict[str, Any]:
    """
    Run the knowledge assessment by asking the model multiple times and analyzing consistency.
    
    Args:
        context: QA context
        question: Question to ask
        num_runs: Number of times to run the model
        embedding_type: 'text' for answer embeddings, 'hidden' for model states
        
    Returns:
        Dictionary containing assessment results
    """
    try:
        results = []
        answers = []
        confidences = []
        
        # Run QA multiple times
        for i in range(num_runs):
            qa_result = model_api.answer_question(context, question)
            
            if qa_result and "answer" in qa_result:
                answer = qa_result.get("answer", "")
                confidence = qa_result.get("score", 0.0)
                
                results.append({
                    "run": i + 1,
                    "answer": answer,
                    "confidence": confidence,
                    "start": qa_result.get("start", -1),
                    "end": qa_result.get("end", -1)
                })
                
                answers.append(answer)
                confidences.append(confidence)
            else:
                logger.warning(f"Failed to get answer for run {i+1}")
        
        if not results:
            return {"error": "No successful QA runs completed"}
        
        # Get embeddings based on type
        if embedding_type == "text":
            embeddings = get_answer_text_embeddings(answers)
        else:
            embeddings = get_model_hidden_state_embeddings(context, question, answers)
        
        if embeddings is None:
            return {"error": "Failed to extract embeddings"}
        
        # Calculate similarity metrics
        similarity_matrix = cosine_similarity(embeddings)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        similarity_std = np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        
        # Analyze answer consistency
        unique_answers = list(set(answers))
        answer_consistency = len(unique_answers) / len(answers) if answers else 0
        
        # Calculate confidence statistics
        conf_mean = np.mean(confidences) if confidences else 0
        conf_std = np.std(confidences) if confidences else 0
        
        return {
            "results": results,
            "embeddings": embeddings,
            "answers": answers,
            "confidences": confidences,
            "unique_answers": unique_answers,
            "similarity_matrix": similarity_matrix,
            "avg_similarity": avg_similarity,
            "similarity_std": similarity_std,
            "answer_consistency": answer_consistency,
            "confidence_mean": conf_mean,
            "confidence_std": conf_std,
            "knowledge_score": calculate_knowledge_score(avg_similarity, answer_consistency, conf_std)
        }
        
    except Exception as e:
        logger.error(f"Error in knowledge assessment: {str(e)}")
        return {"error": str(e)}

def get_answer_text_embeddings(answers: List[str]) -> np.ndarray:
    """Get embeddings for answer texts using sentence transformer."""
    try:
        transformer = get_sentence_transformer()
        if transformer == "fallback" or transformer is None:
            # Fallback to simple word embeddings if sentence transformer fails
            logger.info("Using TF-IDF fallback for embeddings")
            return get_simple_text_embeddings(answers)
        
        embeddings = transformer.encode(answers)
        return embeddings
        
    except Exception as e:
        logger.error(f"Error getting text embeddings: {str(e)}")
        return get_simple_text_embeddings(answers)

def get_simple_text_embeddings(answers: List[str]) -> np.ndarray:
    """Fallback simple text embeddings using TF-IDF."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if not answers:
            return np.array([])
        
        # Handle case where all answers are identical
        if len(set(answers)) == 1:
            # For identical answers, return identical embeddings (single point)
            base_embedding = np.random.random(50)
            np.random.seed(42)  # Fixed seed for consistency
            base_embedding = np.random.random(50)
            embeddings = []
            for _ in answers:
                # Use identical embeddings for identical answers
                embeddings.append(base_embedding.copy())
            return np.array(embeddings)
        
        # Clean and prepare answers
        clean_answers = [str(answer).strip() if answer else "empty" for answer in answers]
        
        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            embeddings = vectorizer.fit_transform(clean_answers).toarray()
            return embeddings
        except ValueError:
            # If TF-IDF fails (e.g., empty vocabulary), use character-level features
            return get_character_level_embeddings(clean_answers)
        
    except ImportError:
        logger.warning("sklearn not available, using basic character embeddings")
        return get_character_level_embeddings(answers)
    except Exception as e:
        logger.error(f"Error getting simple embeddings: {str(e)}")
        # Return random embeddings as last resort
        return get_basic_random_embeddings(answers)

def get_character_level_embeddings(answers: List[str]) -> np.ndarray:
    """Very basic character-level embeddings as fallback."""
    try:
        # Create simple character frequency vectors
        all_chars = set(''.join(answers))
        char_to_idx = {char: i for i, char in enumerate(sorted(all_chars))}
        
        embeddings = []
        for answer in answers:
            vec = np.zeros(len(char_to_idx))
            for char in answer.lower():
                if char in char_to_idx:
                    vec[char_to_idx[char]] += 1
            # Normalize
            if np.sum(vec) > 0:
                vec = vec / np.sum(vec)
            embeddings.append(vec)
        
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Character embeddings failed: {e}")
        return get_basic_random_embeddings(answers)

def get_basic_random_embeddings(answers: List[str]) -> np.ndarray:
    """Most basic fallback - random embeddings with some structure."""
    logger.info("Using basic random embeddings as final fallback")
    
    # Check if all answers are identical
    unique_answers = list(set(answers))
    if len(unique_answers) == 1:
        # For identical answers, return identical embeddings
        seed = hash(str(unique_answers[0])) % 1000
        np.random.seed(seed)
        base_embedding = np.random.random(20)
        embeddings = [base_embedding.copy() for _ in answers]
        return np.array(embeddings)
    
    # Create somewhat structured random embeddings for different answers
    embeddings = []
    for i, answer in enumerate(answers):
        # Use answer length and content to create some structure
        seed = hash(str(answer)) % 1000
        np.random.seed(seed)
        embedding = np.random.random(20)
        embeddings.append(embedding)
    
    return np.array(embeddings)

def get_model_hidden_state_embeddings(context: str, question: str, answers: List[str]) -> np.ndarray:
    """Get embeddings from model hidden states (placeholder for now)."""
    # This would require access to the internal model states
    # For now, return text embeddings as fallback
    logger.info("Hidden state embeddings not implemented, falling back to text embeddings")
    return get_answer_text_embeddings(answers)

def calculate_knowledge_score(avg_similarity: float, answer_consistency: float, conf_std: float) -> float:
    """
    Calculate a knowledge score based on multiple factors.
    
    Args:
        avg_similarity: Average embedding similarity
        answer_consistency: Ratio of unique answers to total answers (lower is better)
        conf_std: Standard deviation of confidence scores (lower is better)
        
    Returns:
        Knowledge score between 0 and 1 (higher means more knowledge)
    """
    # High similarity, low answer variance, low confidence variance = high knowledge
    similarity_score = avg_similarity  # 0-1, higher is better
    consistency_score = 1 - answer_consistency  # 0-1, higher is better (fewer unique answers)
    confidence_score = max(0, 1 - conf_std)  # 0-1, higher is better (less variance)
    
    # Weighted average
    knowledge_score = (0.5 * similarity_score + 0.3 * consistency_score + 0.2 * confidence_score)
    return min(1.0, max(0.0, knowledge_score))

def create_knowledge_results(assessment_data: Dict[str, Any], reduction_method: str) -> html.Div:
    """Create the knowledge assessment results visualization."""
    
    if "error" in assessment_data:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Assessment Error: {assessment_data['error']}"
        ], color="danger", className="mt-4")
    
    # Extract data
    results = assessment_data["results"]
    embeddings = assessment_data["embeddings"]
    knowledge_score = assessment_data["knowledge_score"]
    avg_similarity = assessment_data["avg_similarity"]
    
    # Create summary cards
    summary_cards = create_knowledge_summary_cards(assessment_data)
    
    # Create visualizations
    if reduction_method == "both":
        viz_content = create_combined_visualizations(embeddings, results, assessment_data)
    elif reduction_method == "tsne":
        viz_content = create_tsne_visualization(embeddings, results, assessment_data)
    else:  # pca
        viz_content = create_pca_visualization(embeddings, results, assessment_data)
    
    # Create detailed analysis
    detailed_analysis = create_detailed_analysis(assessment_data)
    
    return html.Div([
        # Success message
        dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"Knowledge assessment completed! Knowledge Score: {knowledge_score:.2f}"
        ], color="success" if knowledge_score > 0.7 else "warning" if knowledge_score > 0.4 else "info", className="mb-4"),
        
        # Summary cards
        summary_cards,
        
        # Visualizations
        viz_content,
        
        # Detailed analysis
        detailed_analysis
    ], className="knowledge-results-container")

def create_knowledge_summary_cards(assessment_data: Dict[str, Any]) -> html.Div:
    """Create summary cards showing key metrics."""
    
    knowledge_score = assessment_data["knowledge_score"]
    avg_similarity = assessment_data["avg_similarity"]
    answer_consistency = assessment_data["answer_consistency"]
    confidence_mean = assessment_data["confidence_mean"]
    num_runs = len(assessment_data["results"])
    unique_answers = len(assessment_data["unique_answers"])
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{knowledge_score:.2f}", className="knowledge-metric-number text-primary"),
                    html.P("Knowledge Score", className="knowledge-metric-label"),
                    html.Small("0.0 = Guessing, 1.0 = Knows", className="knowledge-metric-desc")
                ])
            ], className="knowledge-summary-card text-center")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{avg_similarity:.3f}", className="knowledge-metric-number text-info"),
                    html.P("Avg Similarity", className="knowledge-metric-label"),
                    html.Small("Embedding similarity", className="knowledge-metric-desc")
                ])
            ], className="knowledge-summary-card text-center")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{unique_answers}/{num_runs}", className="knowledge-metric-number text-success"),
                    html.P("Unique Answers", className="knowledge-metric-label"),
                    html.Small("Answer diversity", className="knowledge-metric-desc")
                ])
            ], className="knowledge-summary-card text-center")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{confidence_mean:.2f}", className="knowledge-metric-number text-warning"),
                    html.P("Avg Confidence", className="knowledge-metric-label"),
                    html.Small("Model certainty", className="knowledge-metric-desc")
                ])
            ], className="knowledge-summary-card text-center")
        ], width=3)
    ], className="mb-4")

def create_combined_visualizations(embeddings: np.ndarray, results: List[Dict], assessment_data: Dict[str, Any]) -> html.Div:
    """Create combined t-SNE and PCA visualizations."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("t-SNE Embedding Space", "PCA Embedding Space", 
                       "Answer Similarity Heatmap", "Confidence Distribution"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "histogram"}]]
    )
    
    # Prepare data
    answers = [r["answer"] for r in results]
    confidences = [r["confidence"] for r in results]
    runs = [r["run"] for r in results]
    
    # t-SNE visualization
    if len(embeddings) > 1:
        try:
            # Check if all embeddings are identical
            embeddings_array = np.array(embeddings)
            if np.allclose(embeddings_array, embeddings_array[0], rtol=1e-10):
                # All embeddings are identical - create a single point with small spread for visibility
                center_x, center_y = 0, 0
                tsne_coords = np.array([[center_x + np.random.normal(0, 0.1), center_y + np.random.normal(0, 0.1)] for _ in range(len(embeddings))])
            else:
                perplexity = min(30, max(1, len(embeddings)-1))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_coords = tsne.fit_transform(embeddings)
            
            fig.add_trace(
                go.Scatter(
                    x=tsne_coords[:, 0], y=tsne_coords[:, 1],
                    mode='markers+text',
                    marker=dict(size=12, color=confidences, colorscale='Viridis', 
                              colorbar=dict(title="Confidence"), line=dict(width=2, color='white')),
                    text=[f"Run {r}" for r in runs],
                    textposition="top center",
                    hovertemplate="<b>Run %{text}</b><br>Answer: %{customdata}<br>Confidence: %{marker.color:.3f}<extra></extra>",
                    customdata=answers,
                    name="t-SNE"
                ),
                row=1, col=1
            )
        except ImportError:
            logger.warning("sklearn not available for t-SNE, using simple scatter plot")
            # Fallback to simple scatter plot
            fig.add_trace(
                go.Scatter(
                    x=range(len(embeddings)), y=[np.mean(emb) for emb in embeddings],
                    mode='markers+text',
                    marker=dict(size=12, color=confidences, colorscale='Viridis',
                              line=dict(width=2, color='white')),
                    text=[f"Run {r}" for r in runs],
                    textposition="top center",
                    name="Simple Plot"
                ),
                row=1, col=1
            )
        except Exception as e:
            logger.error(f"t-SNE failed: {e}")
    
    # PCA visualization
    if len(embeddings) > 1:
        try:
            # Check if all embeddings are identical
            embeddings_array = np.array(embeddings)
            if np.allclose(embeddings_array, embeddings_array[0], rtol=1e-10):
                # All embeddings are identical - create a single point with small spread for visibility
                center_x, center_y = 0, 0
                pca_coords = np.array([[center_x + np.random.normal(0, 0.1), center_y + np.random.normal(0, 0.1)] for _ in range(len(embeddings))])
            else:
                pca = PCA(n_components=2, random_state=42)
                pca_coords = pca.fit_transform(embeddings)
            
            fig.add_trace(
                go.Scatter(
                    x=pca_coords[:, 0], y=pca_coords[:, 1],
                    mode='markers+text',
                    marker=dict(size=12, color=confidences, colorscale='Plasma',
                              line=dict(width=2, color='white')),
                    text=[f"Run {r}" for r in runs],
                    textposition="top center",
                    hovertemplate="<b>Run %{text}</b><br>Answer: %{customdata}<br>Confidence: %{marker.color:.3f}<extra></extra>",
                    customdata=answers,
                    name="PCA"
                ),
                row=1, col=2
            )
        except ImportError:
            logger.warning("sklearn not available for PCA, using simple scatter plot")
            # Fallback to simple scatter plot
            fig.add_trace(
                go.Scatter(
                    x=[np.std(emb) for emb in embeddings], y=[np.mean(emb) for emb in embeddings],
                    mode='markers+text',
                    marker=dict(size=12, color=confidences, colorscale='Plasma',
                              line=dict(width=2, color='white')),
                    text=[f"Run {r}" for r in runs],
                    textposition="top center",
                    name="Variance Plot"
                ),
                row=1, col=2
            )
        except Exception as e:
            logger.error(f"PCA failed: {e}")
    
    # Similarity heatmap
    similarity_matrix = assessment_data["similarity_matrix"]
    fig.add_trace(
        go.Heatmap(
            z=similarity_matrix,
            x=[f"Run {i+1}" for i in range(len(similarity_matrix))],
            y=[f"Run {i+1}" for i in range(len(similarity_matrix))],
            colorscale='RdYlBu_r',
            zmin=0, zmax=1,
            colorbar=dict(title="Similarity"),
            name="Similarity"
        ),
        row=2, col=1
    )
    
    # Confidence distribution
    fig.add_trace(
        go.Histogram(
            x=confidences,
            nbinsx=10,
            marker_color='lightblue',
            marker_line=dict(color='darkblue', width=1),
            name="Confidence Dist"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="QA Knowledge Assessment - Combined Analysis",
        showlegend=False
    )
    
    # Check if all answers were identical
    unique_answers = list(set(answers))
    identical_answers = len(unique_answers) == 1
    
    explanation_text = []
    if identical_answers:
        explanation_text = [
            html.P([
                html.Strong("Perfect Consistency: "),
                f"All {len(results)} runs produced the identical answer: \"{unique_answers[0]}\". ",
                "Points are clustered tightly around the center, indicating the model has strong, consistent knowledge."
            ], className="text-success mt-2"),
            html.P([
                "The small spread you see is for visualization purposes only - the actual embeddings are identical. ",
                "This represents the highest possible knowledge score."
            ], className="text-muted small")
        ]
    else:
        explanation_text = [
            html.P([
                "Each point represents one QA run. Tight clusters indicate consistent knowledge; ",
                "scattered points suggest guessing or uncertainty."
            ], className="text-muted mt-2")
        ]
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-area me-2"),
            html.Span("Knowledge Assessment Visualizations")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': True}),
            html.Div(explanation_text)
        ])
    ], className="knowledge-viz-card mb-4")

def create_tsne_visualization(embeddings: np.ndarray, results: List[Dict], assessment_data: Dict[str, Any]) -> html.Div:
    """Create t-SNE only visualization."""
    # Implementation similar to combined but only t-SNE
    pass

def create_pca_visualization(embeddings: np.ndarray, results: List[Dict], assessment_data: Dict[str, Any]) -> html.Div:
    """Create PCA only visualization."""
    # Implementation similar to combined but only PCA
    pass

def create_detailed_analysis(assessment_data: Dict[str, Any]) -> html.Div:
    """Create detailed analysis section."""
    
    results = assessment_data["results"]
    unique_answers = assessment_data["unique_answers"]
    knowledge_score = assessment_data["knowledge_score"]
    
    # Determine knowledge level
    if knowledge_score > 0.8:
        knowledge_level = "High Knowledge"
        knowledge_color = "success"
        knowledge_icon = "fas fa-brain"
    elif knowledge_score > 0.5:
        knowledge_level = "Moderate Knowledge"
        knowledge_color = "warning"
        knowledge_icon = "fas fa-question-circle"
    else:
        knowledge_level = "Low Knowledge / Guessing"
        knowledge_color = "danger"
        knowledge_icon = "fas fa-dice"
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-microscope me-2"),
            html.Span("Detailed Analysis")
        ]),
        dbc.CardBody([
            # Knowledge assessment
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className=f"{knowledge_icon} me-2"),
                        knowledge_level
                    ], className=f"text-{knowledge_color}"),
                    html.P("The model shows consistent answer patterns, suggesting genuine knowledge rather than random guessing." if knowledge_score > 0.6 else 
                          "The model shows inconsistent patterns, suggesting uncertainty or lack of knowledge.", 
                          className="knowledge-analysis-text")
                ], width=8),
                dbc.Col([
                    html.H6("Assessment Criteria:"),
                    html.Ul([
                        html.Li("Embedding similarity"),
                        html.Li("Answer consistency"),
                        html.Li("Confidence stability")
                    ], className="knowledge-criteria-list")
                ], width=4)
            ], className="mb-4"),
            
            # Answer breakdown
            html.H6("Answer Breakdown:"),
            html.Div([
                dbc.Badge(f'"{answer}" (appears {sum(1 for r in results if r["answer"] == answer)} times)', 
                         color="primary" if sum(1 for r in results if r["answer"] == answer) > 1 else "secondary",
                         className="me-2 mb-2")
                for answer in unique_answers
            ], className="mb-3"),
            
            # Recommendations
            html.H6("Recommendations:"),
            html.Ul([
                html.Li("High consistency suggests the model has learned this knowledge during training") if knowledge_score > 0.7 
                else html.Li("Low consistency may indicate insufficient training data or conflicting information"),
                html.Li("Consider testing with variations of the question to assess robustness"),
                html.Li("Compare results across different model sizes or architectures")
            ], className="knowledge-recommendations")
        ])
    ], className="knowledge-analysis-card")

# Note: Callbacks are defined in app.py to ensure proper registration
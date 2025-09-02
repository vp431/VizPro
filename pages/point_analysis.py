"""
Enhanced point analysis page with LIME, Attention Entropy, Token Embeddings, 
Counterfactual Testing, and Similarity Analysis options.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import numpy as np
import time
import logging
import plotly.express as px
import plotly.graph_objects as go

from models.api import model_api
from models.analysis_store import analysis_store
from models.error_analysis import categorize_error_patterns
from utils.dataset_scanner import scan_datasets, load_dataset_samples

logger = logging.getLogger(__name__)

def create_point_analysis_layout(point_data):
    """
    Create the enhanced layout for point analysis with all analysis options.
    
    Args:
        point_data: Dictionary containing the selected point data
        
    Returns:
        A Dash layout object with analysis options
    """
    if not point_data or 'text' not in point_data:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-warning me-2"),
            html.P("No valid data point selected for analysis.")
        ])
    
    text = point_data['text']
    actual_label = point_data.get('true_label', 'Unknown')
    predicted_label = point_data.get('predicted_label', 'Unknown')
    confidence = point_data.get('confidence', 0)
    
    # Convert labels to readable format
    actual_sentiment = "Positive" if str(actual_label) == "1" else "Negative"
    predicted_sentiment = "Positive" if str(predicted_label) == "1" else "Negative"
    
    layout = html.Div([
        # Text and prediction info
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-file-text me-2"),
                    "Selected Text Analysis"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P([
                    html.Strong("Text: "),
                    html.Span(text[:200] + "..." if len(text) > 200 else text)
                ], className="mb-2"),
                
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("Actual: "),
                            html.Span(actual_sentiment, 
                                     className=f"badge bg-{'success' if actual_sentiment == 'Positive' else 'danger'}")
                        ], className="mb-1")
                    ], width=4),
                    dbc.Col([
                        html.P([
                            html.Strong("Predicted: "),
                            html.Span(predicted_sentiment, 
                                     className=f"badge bg-{'success' if predicted_sentiment == 'Positive' else 'danger'}")
                        ], className="mb-1")
                    ], width=4),
                    dbc.Col([
                        html.P([
                            html.Strong("Confidence: "),
                            html.Span(f"{confidence:.2f}", className="badge bg-info")
                        ], className="mb-1")
                    ], width=4)
                ])
            ])
        ], className="mb-4"),
        
        # Analysis options buttons
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-tools me-2"),
                    "Analysis Options"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P("Select an analysis method to understand this prediction:", className="text-muted mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-lightbulb me-2"),
                            "LIME"
                        ], id="point-lime-btn", color="primary", className="w-100 mb-2",
                           title="Local Interpretable Model-agnostic Explanations")
                    ], width=6, md=4),
                    
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-chart-line me-2"),
                            "Attention Entropy"
                        ], id="point-attention-btn", color="info", className="w-100 mb-2",
                           title="Attention pattern analysis")
                    ], width=6, md=4),
                    
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Token Embeddings"
                        ], id="point-embeddings-btn", color="success", className="w-100 mb-2",
                           title="Token representation visualization")
                    ], width=6, md=4),
                    
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-flask me-2"),
                            "Test Counterfactuals"
                        ], id="point-counterfactual-btn", color="warning", className="w-100 mb-2",
                           title="Generate counterfactual examples")
                    ], width=6, md=4)
                ])
            ])
        ], className="mb-4"),
        
        # Results area
        html.Div(id="point-analysis-results", children=[
            html.Div([
                html.I(className="fas fa-arrow-up text-muted me-2"),
                html.P("Select an analysis method above to see detailed results.", className="text-muted mb-0")
            ], className="text-center py-4")
        ])
    ])
    
    return layout

def create_lime_analysis(text):
    """Create LIME analysis for the selected text."""
    try:
        result = model_api.get_lime_explanation(text, num_features=10, num_samples=1000)
        
        if not result:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.P("Error: Could not generate LIME explanation.")
            ])
        
        # Extract LIME data
        words = [item[0] for item in result.get("feature_importance", [])]
        weights = [item[1] for item in result.get("feature_importance", [])]
        
        if not words or not weights:
            return html.Div([
                html.I(className="fas fa-info-circle text-warning me-2"),
                html.P("No LIME features found for this text.")
            ])
        
        # Create visualization
        from components.visualizations import create_lime_bar_chart, highlight_lime_text
        fig = create_lime_bar_chart(words, weights)
        highlighted_text = highlight_lime_text(text, words, weights)
        
        return html.Div([
            html.H5("LIME Explanation Results"),
            html.P(f"Prediction: {result.get('predicted_class', 'Unknown')} (Score: {result.get('prediction_score', 0):.3f})"),
            
            dbc.Card([
                dbc.CardHeader("Feature Importance"),
                dbc.CardBody([
                    dcc.Graph(figure=fig, config={'displayModeBar': True})
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("Highlighted Text"),
                dbc.CardBody([highlighted_text])
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error in LIME analysis: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error generating LIME explanation: {str(e)}")
        ])

def create_attention_analysis(text):
    """Create attention entropy analysis for the selected text."""
    try:
        result = model_api.get_attention_entropy(text)
        
        if not result:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.P("Error: Could not generate attention entropy data.")
            ])
        
        # Create attention entropy visualization
        from components.visualizations import create_attention_entropy_plot
        
        entropy_data = result.get("entropy_by_layer", [])
        if not entropy_data:
            return html.Div([
                html.I(className="fas fa-info-circle text-warning me-2"),
                html.P("No attention entropy data found.")
            ])
        
        # Create simple entropy display for now
        return html.Div([
            html.H5("Attention Entropy Analysis"),
            html.P(f"Average entropy across layers: {result.get('average_entropy', 0):.3f}"),
            
            dbc.Card([
                dbc.CardHeader("Entropy by Layer"),
                dbc.CardBody([
                    html.Ul([
                        html.Li(f"Layer {i}: {entropy:.3f}") 
                        for i, entropy in enumerate(entropy_data)
                    ])
                ])
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error in attention analysis: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error generating attention analysis: {str(e)}")
        ])

def create_embeddings_analysis(text):
    """Create token embeddings analysis for the selected text."""
    try:
        result = model_api.get_sentence_embedding(text)
        
        if not result or "token_embeddings" not in result:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.P("Error: Could not extract embeddings from the model.")
            ])
        
        token_embeddings = result["token_embeddings"]
        tokens = result.get("tokens", text.split())
        
        # Filter out special tokens
        filtered_tokens = []
        filtered_embeddings = []
        for i, token in enumerate(tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and not token.startswith("["):
                filtered_tokens.append(token.replace("##", ""))
                filtered_embeddings.append(token_embeddings[i])
        
        if len(filtered_tokens) < 2:
            return html.Div([
                html.I(className="fas fa-info-circle text-warning me-2"),
                html.P("Need at least 2 tokens for embedding visualization.")
            ])
        
        # Create embedding plot
        from components.visualizations import create_embedding_plot
        fig = create_embedding_plot(filtered_tokens, filtered_embeddings, method="tsne")
        
        return html.Div([
            html.H5("Token Embeddings Analysis"),
            html.P(f"Visualizing {len(filtered_tokens)} tokens in 2D space using t-SNE"),
            
            dbc.Card([
                dbc.CardHeader("Token Embeddings Visualization"),
                dbc.CardBody([
                    dcc.Graph(figure=fig, config={'displayModeBar': True})
                ])
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error in embeddings analysis: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error generating embeddings analysis: {str(e)}")
        ])

# Note: Function moved to pages/error_analysis.py
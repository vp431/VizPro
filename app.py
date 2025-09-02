"""
Single-Page Transformers Visualization Tool
Ported from OldApp multi-page structure to unified single-page layout
"""
import os
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update, ALL
import traceback
import logging
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.dataset_scanner import load_dataset_samples

# Import visualization components
from components.visualizations import (
    create_lime_bar_chart,
    create_embedding_plot,
    create_attention_heatmap_matrix,
    create_heatmap,
    create_attention_heatmap_lines,
    create_attention_all_heads_grid,
    create_attention_all_matrices_grid
)

# Import model API
from models.api import model_api

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from config import UI_CONFIG, DEBUG_MODE, PORT, HOST, FEATURE_CONFIG

# Import utilities
from utils.model_scanner import scan_local_models, get_model_display_name, validate_model_compatibility
from utils.dataset_scanner import scan_datasets

# Import model API
from models.api import model_api
from models.error_analysis import categorize_error_patterns
from models.similarity_analysis import find_similar_examples, update_similarity_analysis
from models.analysis_store import analysis_store
from pages import sentiment_lime, sentiment_attention_entropy, sentiment_token_embeddings, point_analysis, ner_entity_visualization, ner_attention_entropy, logit_matrix
from pages.similarity_analysis import handle_similarity_analysis
from pages.error_patterns import handle_error_patterns_analysis
from pages.knowledge_competition import create_knowledge_competition_layout

# Initialize Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, 
                                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
                                    "/assets/error_analysis_styles.css",
                                    "/assets/error_patterns_styles.css", 
                                    "/assets/similarity_analysis_styles.css",
                                    "/assets/qa_styles.css",
                                    "/assets/qa_knowledge_styles.css",
                                    "/assets/qa_model_viz.css",
                                    "/assets/qa_counterfactual_flow.css"],
                suppress_callback_exceptions=True)

app.title = "Single-Page Transformers Visualization Tool"


def create_control_panel():
    """Create a compact horizontal control panel."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Task:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dcc.Dropdown(
                        id="task-dropdown",
                        options=[
                            {"label": "Sentiment Analysis", "value": "sentiment"},
                            {"label": "Named Entity Recognition", "value": "ner"},
                            {"label": "Question Answering", "value": "qa"}
                        ],
                        value="sentiment",
                        clearable=False,
                        style={"zIndex": "1050", "position": "relative", "width": "100%"}
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("Model:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dcc.Dropdown(
                        id="model-dropdown",
                        placeholder="Select model...",
                        style={"zIndex": "1049", "position": "relative", "width": "100%"}
                    )
                ], width=5),
                
                dbc.Col([
                    html.Label("Level:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    dbc.RadioItems(
                        id="level-toggle",
                        options=[
                            {"label": "Sentence", "value": "sentence"},
                            {"label": "Model", "value": "model"}
                        ],
                        value="sentence",
                        inline=True,
                        style={"fontSize": "0.85rem"}
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("Selected Model:", className="fw-bold mb-1", style={"fontSize": "0.9rem"}),
                    html.Div(id="selected-model-display", 
                            children="None", 
                            className="text-muted selected-model-display",
                            style={"fontSize": "0.8rem", "wordBreak": "break-word"})
                ], width=3)
            ], align="center", style={"width": "100%"})
        ], className="py-2", style={"width": "100%"})
    ], className="mb-3 control-panel-card", style={"width": "100%"})


def create_feature_buttons_area():
    """Create the dynamic feature buttons area."""
    return html.Div(id="feature-buttons-area", className="mb-3")


def create_input_area():
    """Create the dynamic input area that changes based on level."""
    return dbc.Card([
        dbc.CardHeader([
            html.H6("Input", className="mb-0 fw-bold")
        ]),
        dbc.CardBody([
            html.Div([
                html.Div(create_sentence_input(), id="sentence-input-container", style={'display': 'block'}),
                html.Div(create_ner_sentence_input(), id="ner-sentence-input-container", style={'display': 'none'}),
                html.Div(create_qa_sentence_input(), id="qa-sentence-input-container", style={'display': 'none'}),
                html.Div(create_model_input(), id="model-input-container", style={'display': 'none'})
            ], id="input-area-content")
        ])
    ], className="mb-3 input-card", style={"width": "100%"})


def create_analysis_area():
    """Create the main analysis area that changes based on level (attention for sentence, performance for model)."""
    return dbc.Card([
        dbc.CardHeader([
            html.H6(id="analysis-area-title", className="mb-0 fw-bold")
        ]),
        dbc.CardBody([
            html.Div(id="analysis-area-content", style={"width": "100%"})
        ], style={"width": "100%"})
    ], className="analysis-card")



def create_attention_visualization_content():
    """Create attention visualization content for sentence level."""
    return dbc.Row([
        # Left side: Controls
        dbc.Col([
            html.Div([
                # View Type Buttons
                html.Label("View Type:", className="fw-bold mb-2"),
                dbc.ButtonGroup([
                    dbc.Button("Matrix", id="view-matrix-btn", size="sm", outline=True, color="primary", active=True),
                    dbc.Button("Line Graph", id="view-line-btn", size="sm", outline=True, color="primary")
                ], className="mb-3 d-block"),
                
                # Layer Selection
                html.Label("Layer:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="layer-dropdown",
                    options=[{"label": f"Layer {i}", "value": i} for i in range(12)],
                    value=0,
                    className="mb-3"
                ),
                
                # Head Selection
                html.Label("Head:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="head-dropdown", 
                    options=[{"label": f"Head {i}", "value": i} for i in range(12)],
                    value=0,
                    className="mb-3"
                ),
                
                # Grid View Buttons
                html.Label("Grid Views:", className="fw-bold mb-2"),
                dbc.ButtonGroup([
                    dbc.Button("All Heads Grid", id="view-all-heads-btn", size="sm", outline=True, color="secondary"),
                    dbc.Button("All Matrices Grid", id="view-all-matrices-btn", size="sm", outline=True, color="secondary")
                ], className="d-block", vertical=True)
            ])
        ], width=3),
        
        # Right side: Visualization
        dbc.Col([
            html.Div(id="attention-visualization-content", 
                    children=[
                        html.Div([
                            html.I(className="fas fa-chart-line fa-3x text-muted mb-3"),
                            html.H5("Click 'Analyze Text' to see attention patterns", className="text-muted"),
                            html.P("Attention visualization will appear here after analysis.", 
                                  className="text-muted")
                        ], className="text-center py-5")
                    ],
                    style={"width": "100%"})
        ], width=9)
    ], style={"width": "100%"})

def create_model_performance_content():
    """Create model performance content for model level."""
    return html.Div([
        # Performance Summary and Confusion Matrix side by side
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Performance Summary"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="model-summary-loading",
                            children=[
                                html.Div(id="model-summary-stats", children=[
                                    html.Div([
                                        html.I(className="fas fa-chart-bar fa-3x text-muted mb-3"),
                                        html.H5("Click 'Analyze Dataset' to see performance", className="text-muted"),
                                        html.P("Performance summary will appear here after analysis.", 
                                              className="text-muted")
                                    ], className="text-center py-5")
                                ])
                            ],
                            type="default"
                        )
                    ])
                ], className="h-100")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Confusion Matrix"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="model-confusion-loading",
                            children=[
                                html.Div(id="model-confusion-matrix", children=[
                                    html.Div([
                                        html.I(className="fas fa-table fa-3x text-muted mb-3"),
                                        html.H5("Confusion matrix will appear here", className="text-muted"),
                                        html.P("After dataset analysis is complete.", 
                                              className="text-muted")
                                    ], className="text-center py-5")
                                ])
                            ],
                            type="default"
                        )
                    ])
                ], className="h-100")
            ], width=6)
        ], className="mb-4"),
        
        # Modal for detailed error analysis
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Error Pattern Analysis")),
            dbc.ModalBody([
                dcc.Loading(
                    id="error-pattern-loading",
                    children=[html.Div(id="error-pattern-content")],
                    type="default"
                )
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-error-modal", className="ms-auto", n_clicks=0)
            ),
        ], id="error-analysis-modal", size="xl", is_open=False),
        
        # Store for selected error point
        dcc.Store(id="selected-error-store")
    ], style={"width": "100%"})

def create_in_development_content():
    """Create 'In Development' content for NER + Model level."""
    return html.Div([
        html.Div([
            html.I(className="fas fa-tools fa-4x text-muted mb-4"),
            html.H3("In Development", className="text-muted mb-3"),
            html.P("This feature is currently under development and will be available soon.", 
                  className="text-muted text-center", style={"fontSize": "1.2rem"}),
            html.P("Please check back later for NER model-level analysis capabilities.", 
                  className="text-muted text-center")
        ], className="text-center py-5", style={"minHeight": "400px", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
    ], style={"width": "100%"})

def create_ner_results_content():
    """Create NER results content for sentence level."""
    return html.Div([
        html.Div([
            html.I(className="fas fa-search fa-3x text-muted mb-3"),
            html.H5("Click 'Perform NER' to identify entities", className="text-muted"),
            html.P("Entity recognition results will appear here after analysis.", 
                  className="text-muted")
        ], className="text-center py-5")
    ], style={"width": "100%"})


def create_sentence_input():
    """Create sentence-level input (text box)."""
    return html.Div([
        html.Label("Enter text to analyze:", className="fw-bold mb-2", style={"width": "100%"}),
        dbc.Textarea(
            id="sentence-input",
            placeholder="Type or paste your text here...",
            value="I really enjoyed this movie. The acting was superb and the story was engaging",
            rows=4,
            className="mb-3",
            style={"width": "100%", "resize": "vertical"}
        ),
        dbc.Button(
            "Analyze Text",
            id="analyze-sentence-btn",
            color="primary",
            disabled=False,
            className="w-100 mb-3",
            style={"width": "100%"}
        ),
        # Analysis Results Area
        html.Div(id="analysis-results", className="mb-3", style={"width": "100%"})
    ], style={"width": "100%", "display": "block"})

def create_ner_sentence_input():
    """Create NER sentence-level input (text box + NER button + results)."""
    return html.Div([
        html.Label("Enter text to analyze:", className="fw-bold mb-2", style={"width": "100%"}),
        dbc.Textarea(
            id="ner-sentence-input",
            placeholder="Type or paste your text here...",
            value="Apple Inc. is planning to open a new store in New York City next month. CEO Tim Cook announced this during his visit to Berlin, Germany.",
            rows=4,
            className="mb-3",
            style={"width": "100%", "resize": "vertical"}
        ),
        dbc.Button(
            "Perform NER",
            id="perform-ner-btn",
            color="primary",
            disabled=False,
            className="w-100 mb-3",
            style={"width": "100%"}
        ),
        # NER Results Area
        html.Div(id="ner-results", className="mb-3", style={"width": "100%"})
    ], style={"width": "100%", "display": "block"})

def create_qa_sentence_input():
    """Create QA sentence-level input (context + question + button + results)."""
    return html.Div([
        html.Label("Enter context:", className="fw-bold mb-2", style={"width": "100%"}),
        dbc.Textarea(
            id="qa-context-input",
            placeholder="Paste or type the context paragraph here...",
            value="The Apollo program was the third United States human spaceflight program carried out by NASA, which accomplished landing the first humans on the Moon from 1969 to 1972.",
            rows=5,
            className="mb-3",
            style={"width": "100%", "resize": "vertical"}
        ),
        html.Label("Enter question:", className="fw-bold mb-2", style={"width": "100%"}),
        dbc.Input(
            id="qa-question-input",
            placeholder="Type your question about the context...",
            value="Which organization ran the Apollo program?",
            className="mb-3"
        ),
        dbc.Button(
            "Get Answer",
            id="perform-qa-btn",
            color="primary",
            disabled=False,
            className="w-100 mb-3",
            style={"width": "100%"}
        ),
        html.Div(id="qa-results", className="mb-3", style={"width": "100%"})
    ], style={"width": "100%", "display": "block"})


def create_model_input():
    """Create model-level input (dataset selector + controls)."""
    return html.Div([
        html.Label("Select Dataset:", className="fw-bold mb-2"),
        dcc.Dropdown(
            id="dataset-dropdown",
            placeholder="Choose a dataset...",
            className="mb-3",
            style={"zIndex": "1048", "position": "relative"}
        ),
        dbc.Row([
            dbc.Col([
                html.Label("Sample Size:", className="fw-bold mb-2"),
                dbc.Input(
                    id="sample-size-input",
                    type="number",
                    value=100,
                    min=10,
                    max=1000,
                    step=10,
                    className="w-100"
                )
            ], width=6),
            dbc.Col([
                html.Label("Confidence Threshold:", className="fw-bold mb-2"),
                dbc.Input(
                    id="confidence-threshold-input",
                    type="number",
                    value=0.7,
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    className="w-100"
                )
            ], width=6)
        ], className="mb-3"),
        dbc.Button(
            "Analyze Dataset",
            id="analyze-dataset-btn",
            color="primary",
            disabled=True,
            className="w-100"
        )
    ], style={"width": "100%"})

# Main layout
app.layout = dbc.Container([
    # Stores for state management
    dcc.Store(id="available-models-store", storage_type="session"),
    dcc.Store(id="selected-model-store", storage_type="session"),
    dcc.Store(id="available-datasets-store", storage_type="session"),
    dcc.Store(id="selected-dataset-store", storage_type="session"),
    dcc.Store(id="current-analysis-store", storage_type="session"),
    dcc.Store(id="sentiment-popup-data", storage_type="session"),
    dcc.Store(id="attention-analysis-store", storage_type="session"),
    
    # Hidden divs for model performance components (to ensure IDs exist) - REMOVED
    # These are now properly handled in the analysis area
    
    
    # Layer 1: Control Panel
    create_control_panel(),
    
    
    # Layer 2: Input and Analysis (side by side)
    dbc.Row([
        dbc.Col([
            create_input_area()
        ], width=4),
        dbc.Col([
            create_analysis_area()
        ], width=8)
    ], className="mb-4"),
    
    # Layer 3: Visualizations (Feature Buttons)
    create_feature_buttons_area(),
    
    # Popup Modal for Visualizations
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ])
    ], id="visualization-modal", size="xl", is_open=False)
    
], fluid=True, className="py-3")

# Callback to load available models when task changes
@callback(
    Output("available-models-store", "data"),
    Input("task-dropdown", "value")
)
def load_available_models(task):
    """Load and store available models, removing duplicates."""
    try:
        logger.info(f"Loading available models for task: {task}")
        raw_models = scan_local_models()
        
        # Remove duplicates by grouping models with same base name
        unique_models = {}
        seen_base_names = set()
        
        for key, model in raw_models.items():
            # Extract base model name (remove directory prefixes)
            base_name = model.get("name", "").replace("models--", "").replace("--", "/")
            display_name = model.get("display_name", base_name)
            
            # Skip if we've already seen this base model
            if base_name in seen_base_names:
                logger.info(f"Skipping duplicate model: {base_name}")
                continue
            
            seen_base_names.add(base_name)
            unique_models[key] = model
            logger.info(f"Added unique model: {key} -> {base_name} (type: {model.get('type')})")
        
        logger.info(f"After deduplication: {len(unique_models)} unique models")
        return unique_models
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Callback to update model dropdown and lock dataset selection
@callback(
    [Output("model-dropdown", "options"),
     Output("model-dropdown", "value"),
     Output("dataset-dropdown", "disabled"),
     Output("analyze-dataset-btn", "disabled")],
    [Input("task-dropdown", "value"),
     Input("available-models-store", "data"),
     Input("model-dropdown", "value")],
    [State("dataset-dropdown", "value")]
)
def update_model_dropdown(selected_task, available_models, selected_model, dataset_value):
    """Update model dropdown options based on selected task and manage UI state."""
    if not available_models:
        return [], None, True, True
    
    options = []
    default_model = None
    logger.info(f"Updating model dropdown for task: {selected_task}")
    logger.info(f"Available models: {list(available_models.keys())}")
    
    for model_key, model_info in available_models.items():
        # Check compatibility
        try:
            # Ensure model_info is a dictionary
            if not isinstance(model_info, dict):
                logger.error(f"Model {model_key} info is not a dictionary: {type(model_info)}")
                continue
                
            model_path = model_info.get("path")
            if not model_path:
                logger.error(f"Model {model_key} has no path")
                continue
                
            compatibility, message = validate_model_compatibility(model_path, selected_task)
            
            logger.info(f"Model {model_key}: {compatibility} - {message}")
            
            # Include compatible and adaptable models
            if compatibility in ["compatible", "adaptable"]:
                label = get_model_display_name(model_info)
                
                # For BERT models, show what tasks they can adapt to
                if compatibility == "adaptable" and model_info.get("type") == "bert":
                    if selected_task == "sentiment":
                        label += " (will adapt for sentiment)"
                    elif selected_task == "ner":
                        label += " (will adapt for NER)"
                    else:
                        label += " (adaptable)"
                elif compatibility == "compatible":
                    label += " (native)"
                
                options.append({
                    "label": label,
                    "value": model_key
                })
                
                # Set default model for sentiment analysis
                if selected_task == "sentiment" and "distilbert-base-uncased-finetuned-sst-2-english" in model_key:
                    default_model = model_key
                    logger.info(f"Setting default sentiment model: {model_key}")
                
                # Set default model for NER analysis
                if selected_task == "ner" and "bert-base-NER" in model_key:
                    default_model = model_key
                    logger.info(f"Setting default NER model: {model_key}")

                # Set default model for QA analysis
                if selected_task == "qa" and "distilbert-base-uncased-distilled-squad" in model_key:
                    default_model = model_key
                    logger.info(f"Setting default QA model: {model_key}")
                    
            else:
                # Don't show incompatible models to reduce clutter
                logger.info(f"Hiding incompatible model: {model_key}")
        except Exception as e:
            logger.error(f"Error checking compatibility for {model_key}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If no specific default found, use first compatible model
    if not default_model and options:
        for option in options:
            if not option.get("disabled", False):
                default_model = option["value"]
                break
    
    logger.info(f"Final options: {len(options)} models, default: {default_model}")
    
    # Determine if dataset selection should be enabled
    dataset_disabled = not selected_model  # Disable if no model selected
    analyze_disabled = not (selected_model and dataset_value)  # Disable if either model or dataset not selected
    
    # If there's a model change, keep the current selection if valid, otherwise use default
    if selected_model in [opt["value"] for opt in options]:
        model_value = selected_model
    else:
        model_value = default_model
    
    return options, model_value, dataset_disabled, analyze_disabled

# Callback to update selected model display
@callback(
    [Output("selected-model-display", "children"),
     Output("selected-model-store", "data")],
    [Input("model-dropdown", "value")],
    [State("available-models-store", "data")]
)
def update_selected_model(selected_model_key, available_models):
    """Update the selected model display and store."""
    logger.info(f"Updating selected model: {selected_model_key}")
    logger.info(f"Available models keys: {list(available_models.keys()) if available_models else 'None'}")
    
    if not selected_model_key or not available_models:
        return "None", None
    
    if selected_model_key in available_models:
        model_info = available_models[selected_model_key]
        display_name = get_model_display_name(model_info)
        
        # Set the selected model in the API
        model_api.set_selected_model(model_info["path"], model_info["type"])
        
        logger.info(f"Selected model set: {display_name}")

        # Ensure model is preloaded in memory (with timeout protection)
        try:
            logger.info(f"Ensuring {model_info['type']} model is loaded in memory")
            if model_info["type"] == 'bert' and (not hasattr(model_api, '_bert_visualizer') or model_api._bert_visualizer is None):
                model_api.get_bert_visualizer(model_info["path"])
            elif model_info["type"] == 'sentiment' and (not hasattr(model_api, '_sentiment_analyzer') or model_api._sentiment_analyzer is None):
                model_api.get_sentiment_analyzer(model_info["path"])
            elif model_info["type"] == 'ner' and (not hasattr(model_api, '_ner_model') or model_api._ner_model is None):
                model_api.get_ner_model(model_info["path"])
            logger.info(f"Model {model_info['type']} is ready in memory")
        except Exception as e:
            logger.error(f"Model preloading failed: {str(e)}")
            # Clear the problematic model from API to force reload
            model_api.clear_cache()
            # Try to reload
            try:
                if model_info["type"] == 'sentiment':
                    model_api.get_sentiment_analyzer(model_info["path"])
                logger.info(f"Model {model_info['type']} reloaded successfully")
            except Exception as e2:
                logger.error(f"Model reload also failed: {str(e2)}")
                # Don't fail the callback, but the model won't be available
        
        return display_name, {
            "model_key": selected_model_key,
            "model_path": model_info["path"],
            "model_type": model_info["type"],
            "display_name": display_name
        }
    
    logger.warning(f"Model key {selected_model_key} not found in available models")
    return "None", None

# Callback to generate feature buttons based on task+level combination
@callback(
    Output("feature-buttons-area", "children"),
    [Input("task-dropdown", "value"),
     Input("level-toggle", "value")]
)
def update_feature_buttons(task, level):
    """Generate feature buttons based on task and level combination."""
    if not task or not level:
        return []
    
    # For NER + Model, don't show any feature buttons (In Development)
    if task == "ner" and level == "model":
        return html.Div()  # Empty div, no buttons
    
    features = FEATURE_CONFIG.get(task, {}).get(level, [])
    
    if not features:
        return html.Div("No features available for this combination.", 
                       className="text-muted text-center py-3")
    
    buttons = []
    for feature in features:
        button_id = {"type": "feature-btn", "index": feature['id']}
        logger.info(f"Creating button with ID: {button_id}")
        button = dbc.Button(
            feature["label"],
            id=button_id,
            color=feature["color"],
            outline=True,
            className="me-2 mb-2"
        )
        buttons.append(button)
    
    return html.Div([
        html.Label("Visualizations:", className="fw-bold mb-2"),
        html.Div(buttons)
    ])

# Callback to update input area based on level and task
@callback(
    [Output("sentence-input-container", "style"),
     Output("ner-sentence-input-container", "style"),
     Output("qa-sentence-input-container", "style"),
     Output("model-input-container", "style")],
    [Input("level-toggle", "value"),
     Input("task-dropdown", "value")]
)
def update_input_area(level, task):
    """Update input area content based on analysis level and task."""
    if task == "sentiment":
        if level == "sentence":
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif level == "model":
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    elif task == "ner":
        if level == "sentence":
            return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
        elif level == "model":
            # For NER + Model, hide all input areas completely (In Development mode)
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    elif task == "qa":
        if level == "sentence":
            return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
        elif level == "model":
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    # Default case
    return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback to update analysis area based on level and task
@callback(
    Output("analysis-area-title", "children"),
    [Input("level-toggle", "value"),
     Input("task-dropdown", "value")]
)
def update_analysis_area_title(level, task):
    """Update analysis area title based on level and task."""
    if task == "sentiment":
        if level == "sentence":
            return "Attention Visualization"
        elif level == "model":
            return "Model Performance"
    elif task == "ner":
        if level == "sentence":
            return "Results"
        elif level == "model":
            return "In Development"
    elif task == "qa":
        if level == "sentence":
            return "Answer"
        elif level == "model":
            return "In Development"
    
    return "Analysis"

# Callback to initialize analysis area content based on level and task (only when no analysis has been done)
@callback(
    Output("analysis-area-content", "children", allow_duplicate=True),
    [Input("level-toggle", "value"),
     Input("task-dropdown", "value")],
    prevent_initial_call=True
)
def initialize_analysis_area_content(level, task):
    """Initialize analysis area content based on level and task."""
        
    if task == "sentiment":
        if level == "sentence":
            return create_attention_visualization_content()
        elif level == "model":
            return create_model_performance_content()
    elif task == "ner":
        if level == "sentence":
            return create_ner_results_content()
        elif level == "model":
            # For NER + Model, return a simple centered message
            return html.Div([
                html.Div([
                    html.I(className="fas fa-tools fa-4x text-muted mb-4"),
                    html.H3("In Development", className="text-muted mb-3"),
                    html.P("This feature is currently under development and will be available soon.", 
                          className="text-muted text-center", style={"fontSize": "1.2rem"}),
                    html.P("Please check back later for NER model-level analysis capabilities.", 
                          className="text-muted text-center")
                ], className="text-center", style={"minHeight": "60vh", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
            ])
    elif task == "qa":
        if level == "sentence":
            # Blank area; the answer will appear after clicking Get Answer
            return html.Div([
                html.Div([
                    html.I(className="fas fa-comments fa-3x text-muted mb-3"),
                    html.H5("Enter context and question, then click 'Get Answer'", className="text-muted"),
                ], className="text-center py-5")
            ])
        elif level == "model":
            return html.Div([
                html.Div([
                    html.I(className="fas fa-tools fa-4x text-muted mb-4"),
                    html.H3("In Development", className="text-muted mb-3"),
                    html.P("This feature is currently under development and will be available soon.", 
                          className="text-muted text-center", style={"fontSize": "1.2rem"})
                ], className="text-center", style={"minHeight": "60vh", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
            ])
    
    return html.Div("Select a task and level to see analysis options.", className="text-muted text-center py-5")

# This callback was removed - duplicate of analyze_dataset_final

# Callback to populate dataset dropdown options
@callback(
    Output("dataset-dropdown", "options"),
    [Input("task-dropdown", "value"),
     Input("selected-model-store", "data")]
)
def update_dataset_dropdown(selected_task, selected_model):
    """Update dataset dropdown options based on selected task."""
    if not selected_task:
        return []
    
    try:
        datasets = scan_datasets()
        task_datasets = datasets.get(selected_task, {})
        
        options = []
        for dataset_key, dataset_info in task_datasets.items():
            options.append({
                "label": dataset_info.get("display_name", dataset_key),
                "value": dataset_key
            })
        
        logger.info(f"Dataset options for {selected_task}: {[opt['label'] for opt in options]}")
        return options
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        return []

# Callback to update selected dataset store
@callback(
    [Output("selected-dataset-store", "data", allow_duplicate=True),
     Output("analyze-dataset-btn", "disabled", allow_duplicate=True)],
    [Input("dataset-dropdown", "value"),
     Input("selected-model-store", "data")],
    prevent_initial_call=True
)
def update_selected_dataset(dataset_value, selected_model):
    """Update selected dataset store and manage analyze button state."""
    if not dataset_value or not selected_model:
        return None, True
    
    try:
        datasets = scan_datasets()
        for task_datasets in datasets.values():
            if dataset_value in task_datasets:
                dataset_info = task_datasets[dataset_value]
                return {
                    "key": dataset_value,
                    "name": dataset_info.get("display_name", dataset_value),
                    "path": dataset_info.get("path", ""),
                    "default_samples": dataset_info.get("default_samples", 200),
                    "default_threshold": dataset_info.get("default_threshold", 0.7)
                }, False
        
        return None, True
    except Exception as e:
        logger.error(f"Error updating selected dataset: {str(e)}")
        return None, True

# Callback for analyze text button
@callback(
    [Output("analysis-results", "children", allow_duplicate=True),
     Output("analysis-area-content", "children", allow_duplicate=True),
     Output("attention-analysis-store", "data", allow_duplicate=True)],
    [Input("analyze-sentence-btn", "n_clicks")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data"),
     State("level-toggle", "value"),
     State("task-dropdown", "value")],
    prevent_initial_call=True
)
def analyze_text(n_clicks, sentence_text, selected_model, level, task):
    """Analyze text and show result plus appropriate visualization."""
    if not n_clicks or not sentence_text or not selected_model:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Perform analysis based on task type
        if task == "sentiment":
            sentiment_result = model_api.analyze_sentiment(sentence_text)
        elif task == "ner":
            # For NER, we don't show results in this callback - results are shown via feature buttons
            # Just return empty visualization
            visualization = html.Div([
                html.P("Use the feature buttons below to explore NER analysis.", 
                       className="text-muted text-center py-3")
            ])
            return dash.no_update, visualization, dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update
            
        # For sentiment analysis, show results and update visualization
        if task == "sentiment":
            # Create result display
            label = sentiment_result.get("label", "Unknown")
            score = sentiment_result.get("score", 0)
            
            if str(label).lower() in ["positive", "1", "1.0"]:
                label_text = "Positive"
                color = "success"
            else:
                label_text = "Negative" 
                color = "danger"
            
            result_card = dbc.Alert([
                html.H5(f"Prediction: {label_text}", className="mb-2"),
                html.P(f"Confidence: {score:.2%}", className="mb-0")
            ], color=color, className="mb-3")
            
            # Update attention visualization area with actual attention data
            try:
                # Get attention weights for the analyzed text
                attention_data = model_api.get_attention_weights(sentence_text, layer_idx=0, head_idx=0)
                
                if attention_data and 'attention_weights' in attention_data:
                    # Create actual attention visualization
                    from components.visualizations import create_attention_heatmap_matrix
                    
                    tokens = attention_data.get('tokens', [])
                    attention_weights = attention_data['attention_weights']
                    
                    # Create the attention heatmap
                    fig = create_attention_heatmap_matrix(tokens, attention_weights, layer_idx=0, head_idx=0)
                    
                    # Store the analysis data for persistence
                    analysis_store_data = {
                        "text": sentence_text,
                        "sentiment_result": sentiment_result,
                        "attention_data": attention_data,
                        "task": task,
                        "timestamp": n_clicks  # Use n_clicks as a simple timestamp
                    }
                    
                    # Create the visualization content with the actual attention data
                    visualization = dbc.Row([
                        # Left side: Controls
                        dbc.Col([
                            html.Div([
                                # View Type Buttons
                                html.Label("View Type:", className="fw-bold mb-2"),
                                dbc.ButtonGroup([
                                    dbc.Button("Matrix", id="view-matrix-btn", size="sm", outline=True, color="primary", active=True),
                                    dbc.Button("Line Graph", id="view-line-btn", size="sm", outline=True, color="primary")
                                ], className="mb-3 d-block"),
                                
                                # Layer Selection
                                html.Label("Layer:", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id="layer-dropdown",
                                    options=[{"label": f"Layer {i}", "value": i} for i in range(12)],
                                    value=0,
                                    className="mb-3"
                                ),
                                
                                # Head Selection
                                html.Label("Head:", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id="head-dropdown", 
                                    options=[{"label": f"Head {i}", "value": i} for i in range(12)],
                                    value=0,
                                    className="mb-3"
                                ),
                                
                                # Grid View Buttons
                                html.Label("Grid Views:", className="fw-bold mb-2"),
                                dbc.ButtonGroup([
                                    dbc.Button("All Heads Grid", id="view-all-heads-btn", size="sm", outline=True, color="secondary"),
                                    dbc.Button("All Matrices Grid", id="view-all-matrices-btn", size="sm", outline=True, color="secondary")
                                ], className="d-block", vertical=True)
                            ])
                        ], width=3),
                        
                        # Right side: Visualization with actual attention data
                        dbc.Col([
                            html.Div([
                                html.H6("Attention Patterns", className="mb-3"),
                                dcc.Graph(
                                    figure=fig,
                                    id="attention-visualization-content",
                                    config={'displayModeBar': False},
                                    style={"height": "300px", "width": "100%"}
                                ),
                            ], style={"width": "100%"})
                        ], width=9)
                    ], style={"width": "100%"})
                    
                    return result_card, visualization, analysis_store_data
                else:
                    # Fallback to default visualization if attention data not available
                    visualization = create_attention_visualization_content()
                    return result_card, visualization, {"text": sentence_text, "sentiment_result": sentiment_result, "task": task, "timestamp": n_clicks}
            except Exception as e:
                logger.error(f"Error creating attention visualization: {str(e)}")
                # Fallback to default visualization on error
                visualization = create_attention_visualization_content()
                return result_card, visualization, {"text": sentence_text, "sentiment_result": sentiment_result, "task": task, "timestamp": n_clicks}
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return error_msg, dash.no_update, dash.no_update

# Callback for QA analysis button
@callback(
    [Output("qa-results", "children", allow_duplicate=True),
     Output("analysis-area-content", "children", allow_duplicate=True)],
    [Input("perform-qa-btn", "n_clicks")],
    [State("qa-context-input", "value"),
     State("qa-question-input", "value"),
     State("selected-model-store", "data"),
     State("level-toggle", "value"),
     State("task-dropdown", "value")],
    prevent_initial_call=True
)
def perform_qa_analysis(n_clicks, context_text, question_text, selected_model, level, task):
    """Perform QA and show the answer in both the small results area and the main analysis area."""
    if not n_clicks or not selected_model or not context_text or not question_text:
        return dash.no_update, dash.no_update

    if task != "qa" or level != "sentence":
        return dash.no_update, dash.no_update

    try:
        # Normalize inputs (use same normalized context for inference and highlighting)
        context = (context_text or "").strip()
        question = (question_text or "").strip()

        # Run QA
        qa_result = model_api.answer_question(context, question)

        if not qa_result or "answer" not in qa_result or qa_result.get("error"):
            error_msg = qa_result.get("error", "Could not generate answer.") if isinstance(qa_result, dict) else "Could not generate answer."
            error_alert = dbc.Alert(f"Error: {error_msg}", color="danger")
            return error_alert, dash.no_update

        answer = qa_result.get("answer", "")
        score = qa_result.get("score", 0.0)
        start = qa_result.get("start", -1)
        end = qa_result.get("end", -1)

        # Fallback: if start/end invalid, try string search for answer span (case-insensitive)
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context)):
            if answer:
                idx = context.lower().find(answer.lower())
                if idx != -1:
                    start, end = idx, idx + len(answer)

        # Do not show answer below inputs; only use the main analysis area for a clean UI
        mini = html.Div()

        # Build compact, styled answer view (vertical stacking in CSS)
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context):
            highlighted_context_children = [
                html.Span(context[:start]),
                html.Span(context[start:end], className="qa-context-highlight", title="Predicted Answer Span"),
                html.Span(context[end:])
            ]
        else:
            highlighted_context_children = [html.Span(context)]

        analysis = html.Div(className="qa-answer-container", children=[
            html.Div(className="qa-answer-card", children=[
                html.Div(className="qa-answer-header", children=[
                    html.I(className="fas fa-comment-dots qa-answer-icon"),
                    html.Span("Answer", className="qa-answer-title")
                ]),
                html.Div(className="qa-answer-text", children=answer),
                html.Span(f"{score:.1%}", className="qa-confidence-badge")
            ]),
            html.Div(className="qa-context-card", children=[
                html.Div(className="qa-context-header", children=[
                    html.I(className="fas fa-book-open qa-context-icon"),
                    html.Span("Context", className="qa-context-title")
                ]),
                html.Div(className="qa-context-body", children=highlighted_context_children)
            ])
        ])

        return mini, analysis

    except Exception as e:
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return error_msg, dash.no_update

# Callback for NER analysis button
@callback(
    [Output("ner-results", "children", allow_duplicate=True),
     Output("analysis-area-content", "children", allow_duplicate=True)],
    [Input("perform-ner-btn", "n_clicks")],
    [State("ner-sentence-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def perform_ner_analysis(n_clicks, text, selected_model):
    """Perform NER analysis and show highlighted results - NO POPUP should appear."""
    if not n_clicks or not text or not selected_model:
        return dash.no_update, dash.no_update
    
    try:
        # Perform NER analysis
        ner_result = model_api.get_ner_prediction(text)
        
        if not ner_result or 'entities' not in ner_result:
            error_msg = dbc.Alert("Error: Could not perform NER analysis", color="danger")
            return error_msg, dash.no_update
        
        entities = ner_result['entities']
        
        # Create highlighted text with entities
        highlighted_text = create_highlighted_text(text, entities)
        
        # Create detailed results table
        entity_table = create_entity_results_table(entities)
        
        # Simple entity count display for NER results area
        entity_count = len(entities)
        ner_summary = html.Div([
            html.P(f"{entity_count} entities found", className="mb-0", 
                   style={"fontSize": "1.1rem", "fontWeight": "500"})
        ])
        
        # Highlighted text for the analysis area (Results tab)
        results_content = html.Div([
            html.H5("Highlighted Text", className="mb-3"),
            
            # Highlighted text display
            html.Div([
                html.Div(highlighted_text, className="p-3", 
                        style={"backgroundColor": "#f8f9fa", "borderRadius": "5px", "border": "1px solid #dee2e6"})
            ], className="mb-3"),
            
            # Color legend for entity types
            html.Div([
                html.H6("Entity Highlighted Color:", className="mb-2"),
                html.Div([
                    html.Span([
                        html.Span("PER", style={
                            "backgroundColor": "#FFE6E6", 
                            "padding": "2px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #FFE6E6",
                            "marginRight": "5px",
                            "fontSize": "0.9rem"
                        }),
                        html.Span(" - Person names", style={"fontSize": "0.9rem"})
                    ], className="me-3"),
                    
                    html.Span([
                        html.Span("ORG", style={
                            "backgroundColor": "#E6F3FF", 
                            "padding": "2px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #E6F3FF",
                            "marginRight": "5px",
                            "fontSize": "0.9rem"
                        }),
                        html.Span(" - Organizations", style={"fontSize": "0.9rem"})
                    ], className="me-3"),
                    
                    html.Span([
                        html.Span("LOC", style={
                            "backgroundColor": "#E6FFE6", 
                            "padding": "2px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #E6FFE6",
                            "marginRight": "5px",
                            "fontSize": "0.9rem"
                        }),
                        html.Span(" - Locations", style={"fontSize": "0.9rem"})
                    ], className="me-3"),
                    
                    html.Span([
                        html.Span("MISC", style={
                            "backgroundColor": "#FFF0E6", 
                            "padding": "2px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #FFF0E6",
                            "marginRight": "5px",
                            "fontSize": "0.9rem"
                        }),
                        html.Span(" - Miscellaneous entities", style={"fontSize": "0.9rem"})
                    ])
                ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"})
            ], className="mb-3", style={"padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "5px", "border": "1px solid #dee2e6"}),
            
            # Detailed Results button
            dbc.Button(
                "Detailed Results",
                id="ner-detailed-results-btn",
                color="primary",
                outline=True,
                className="mb-3"
            ),
            
            # Modal for detailed results
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Detailed NER Results")),
                dbc.ModalBody([
                    # Highlighted text section
                    html.H6("Highlighted Text", className="mb-3"),
                    html.Div([
                        html.Div(highlighted_text, className="p-3", 
                                style={"backgroundColor": "#f8f9fa", "borderRadius": "5px", "border": "1px solid #dee2e6"})
                    ], className="mb-3"),
                    
                    # Color legend for entity types in modal
                    html.Div([
                        html.H6("Entity Highlighted Color:", className="mb-2"),
                        html.Div([
                            html.Span([
                                html.Span("PER", style={
                                    "backgroundColor": "#FFE6E6", 
                                    "padding": "2px 6px", 
                                    "borderRadius": "3px", 
                                    "border": "1px solid #FFE6E6",
                                    "marginRight": "5px",
                                    "fontSize": "0.9rem"
                                }),
                                html.Span(" - Person names", style={"fontSize": "0.9rem"})
                            ], className="me-3"),
                            
                            html.Span([
                                html.Span("ORG", style={
                                    "backgroundColor": "#E6F3FF", 
                                    "padding": "2px 6px", 
                                    "borderRadius": "3px", 
                                    "border": "1px solid #E6F3FF",
                                    "marginRight": "5px",
                                    "fontSize": "0.9rem"
                                }),
                                html.Span(" - Organizations", style={"fontSize": "0.9rem"})
                            ], className="me-3"),
                            
                            html.Span([
                                html.Span("LOC", style={
                                    "backgroundColor": "#E6FFE6", 
                                    "padding": "2px 6px", 
                                    "borderRadius": "3px", 
                                    "border": "1px solid #E6FFE6",
                                    "marginRight": "5px",
                                    "fontSize": "0.9rem"
                                }),
                                html.Span(" - Locations", style={"fontSize": "0.9rem"})
                            ], className="me-3"),
                            
                            html.Span([
                                html.Span("MISC", style={
                                    "backgroundColor": "#FFF0E6", 
                                    "padding": "2px 6px", 
                                    "borderRadius": "3px", 
                                    "border": "1px solid #FFF0E6",
                                    "marginRight": "5px",
                                    "fontSize": "0.9rem"
                                }),
                                html.Span(" - Miscellaneous entities", style={"fontSize": "0.9rem"})
                            ])
                        ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"})
                    ], className="mb-4", style={"padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "5px", "border": "1px solid #dee2e6"}),
                    
                    # Detailed results table
                    html.H6("Entity Details", className="mb-3"),
                    entity_table if entities else html.P("No entities found in the text.", className="text-muted"),
                    
                    html.Hr(),
                    
                    # Implementation details
                    html.H6("Implementation Details", className="mb-3"),
                    html.Div([
                        html.H6("Model Information", className="mb-2", style={"fontSize": "1rem"}),
                        html.P(f"Model: {selected_model.get('display_name', 'Unknown')}", className="mb-1"),
                        html.P(f"Model Type: {selected_model.get('model_type', 'Unknown')}", className="mb-3"),
                        
                        html.H6("Analysis Process", className="mb-2", style={"fontSize": "1rem"}),
                        html.Ul([
                            html.Li("Text tokenization using model-specific tokenizer"),
                            html.Li("Forward pass through the NER model"),
                            html.Li("Entity extraction with confidence scoring"),
                            html.Li("Post-processing and aggregation of sub-word tokens")
                        ], className="mb-3"),
                        
                        html.H6("Entity Categories", className="mb-2", style={"fontSize": "1rem"}),
                        html.Ul([
                            html.Li([html.Strong("PER"), " - Person names"]),
                            html.Li([html.Strong("ORG"), " - Organizations"]),
                            html.Li([html.Strong("LOC"), " - Locations"]),
                            html.Li([html.Strong("MISC"), " - Miscellaneous entities"])
                        ])
                    ])
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-ner-detailed-modal", className="ms-auto", n_clicks=0)
                ),
            ], id="ner-detailed-modal", size="lg", is_open=False)
        ])
        
        return ner_summary, results_content
        
    except Exception as e:
        logger.error(f"Error in NER analysis: {str(e)}")
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return error_msg, dash.no_update

def create_highlighted_text(text, entities):
    """Create highlighted text with entity colors."""
    if not entities:
        return html.Span(text)
    
    # Define colors for different entity types
    entity_colors = {
        'PER': '#FFE6E6',    # Light red for persons
        'ORG': '#E6F3FF',    # Light blue for organizations  
        'LOC': '#E6FFE6',    # Light green for locations
        'MISC': '#FFF0E6'    # Light orange for miscellaneous
    }
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    
    # Create highlighted spans
    result_spans = []
    last_end = 0
    
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        entity_type = entity['entity']
        word = entity['word']
        score = entity['score']
        
        # Add text before entity
        if start > last_end:
            result_spans.append(html.Span(text[last_end:start]))
        
        # Add highlighted entity
        color = entity_colors.get(entity_type, '#F0F0F0')
        result_spans.append(
            html.Span(
                word,
                style={
                    'backgroundColor': color,
                    'padding': '2px 4px',
                    'borderRadius': '3px',
                    'border': f'1px solid {color}',
                    'margin': '0 1px'
                },
                title=f"{entity_type} (confidence: {score:.2f})"
            )
        )
        
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        result_spans.append(html.Span(text[last_end:]))
    
    return result_spans

def create_entity_results_table(entities):
    """Create a table showing detailed entity results."""
    if not entities:
        return html.P("No entities found.", className="text-muted")
    
    rows = []
    for i, entity in enumerate(entities, 1):
        rows.append(
            html.Tr([
                html.Td(i),
                html.Td(entity['word']),
                html.Td(entity['entity']),
                html.Td(f"{entity['score']:.3f}"),
                html.Td(f"{entity['start']}-{entity['end']}")
            ])
        )
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("#"),
                html.Th("Entity"),
                html.Th("Type"),
                html.Th("Confidence"),
                html.Th("Position")
            ])
        ]),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True)
    
    try:
        # Perform analysis based on task type
        if task == "sentiment":
            sentiment_result = model_api.analyze_sentiment(sentence_text)
        elif task == "ner":
            # For NER, we don't show results in this callback - results are shown via feature buttons
            # Just return empty visualization
            visualization = html.Div([
                html.P("Use the feature buttons below to explore NER analysis.", 
                       className="text-muted text-center py-3")
            ])
            return html.Div(), visualization
        else:
            # Default to sentiment for backward compatibility
            sentiment_result = model_api.analyze_sentiment(sentence_text)
        
        # Create result display - only for sentiment analysis
        if sentiment_result:
            label = sentiment_result.get('label', 'Unknown')
            confidence = sentiment_result.get('score', 0.0)
            
            if label.upper() in ['POSITIVE', 'POS', '1']:
                color = "success"
                icon = "fas fa-smile"
                result_text = "Positive"
            else:
                color = "danger" 
                icon = "fas fa-frown"
                result_text = "Negative"
            
            result_display = dbc.Alert([
                html.I(className=f"{icon} me-2"),
                html.Strong(f"{result_text} ({confidence:.2%} confidence)")
            ], color=color, className="mb-0")
        else:
            result_display = dbc.Alert("Analysis failed", color="warning")
        
        # Create visualization based on level and task
        if level == "sentence":
            if task == "ner":
                # For NER, don't show attention visualization by default
                visualization = html.Div([
                    html.P("Select a visualization option from the buttons above.", className="text-muted text-center py-3")
                ])
            else:
                visualization = create_attention_visualization_content()
        else:
            visualization = create_model_performance_content()
        
        return result_display, visualization
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        error_display = dbc.Alert(f"Analysis error: {str(e)}", color="danger")
        return error_display, dash.no_update

# Callback to update attention visualization when layer/head changes
@callback(
    Output("attention-visualization-content", "figure", allow_duplicate=True),
    [Input("layer-dropdown", "value"),
     Input("head-dropdown", "value")],
    [State("attention-analysis-store", "data"),
     State("selected-model-store", "data"),
     State("task-dropdown", "value")],
    prevent_initial_call=True
)
def update_attention_visualization(layer_idx, head_idx, stored_analysis, selected_model, task):
    """Update attention visualization when layer or head selection changes."""
    if not stored_analysis or not selected_model or task != "sentiment":
        return dash.no_update
    
    # Get the text from stored analysis
    sentence_text = stored_analysis.get("text")
    if not sentence_text:
        return dash.no_update
    
    try:
        # Get attention weights for the selected layer and head
        attention_data = model_api.get_attention_weights(sentence_text, layer_idx=layer_idx, head_idx=head_idx)
        
        if attention_data and 'attention_weights' in attention_data:
            from components.visualizations import create_attention_heatmap_matrix
            
            tokens = attention_data.get('tokens', [])
            attention_weights = attention_data['attention_weights']
            
            # Create the attention heatmap
            fig = create_attention_heatmap_matrix(tokens, attention_weights, layer_idx=layer_idx, head_idx=head_idx)
            return fig
        else:
            return dash.no_update
            
    except Exception as e:
        logger.error(f"Error updating attention visualization: {str(e)}")
        return dash.no_update

# Callback to toggle between matrix and line view
@callback(
    [Output("view-matrix-btn", "active"),
     Output("view-line-btn", "active"),
     Output("attention-visualization-content", "figure", allow_duplicate=True)],
    [Input("view-matrix-btn", "n_clicks"),
     Input("view-line-btn", "n_clicks")],
    [State("attention-analysis-store", "data"),
     State("layer-dropdown", "value"),
     State("head-dropdown", "value")],
    prevent_initial_call=True
)
def toggle_attention_view(matrix_clicks, line_clicks, stored_analysis, layer_idx, head_idx):
    """Toggle between matrix and line graph views for attention visualization."""
    if not stored_analysis:
        return dash.no_update, dash.no_update, dash.no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Get the text from stored analysis
    sentence_text = stored_analysis.get("text")
    if not sentence_text:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Get attention weights for the selected layer and head
        attention_data = model_api.get_attention_weights(sentence_text, layer_idx=layer_idx, head_idx=head_idx)
        
        if attention_data and 'attention_weights' in attention_data:
            tokens = attention_data.get('tokens', [])
            attention_weights = attention_data['attention_weights']
            
            if button_id == "view-matrix-btn":
                # Show matrix view
                from components.visualizations import create_attention_heatmap_matrix
                fig = create_attention_heatmap_matrix(tokens, attention_weights, layer_idx=layer_idx, head_idx=head_idx)
                return True, False, fig
            elif button_id == "view-line-btn":
                # Show line view
                from components.visualizations import create_attention_heatmap_lines
                fig = create_attention_heatmap_lines(tokens, attention_weights, layer_idx=layer_idx, head_idx=head_idx)
                return False, True, fig
        
        return dash.no_update, dash.no_update, dash.no_update
        
    except Exception as e:
        logger.error(f"Error toggling attention view: {str(e)}")
        return dash.no_update, dash.no_update, dash.no_update

# This callback was removed - duplicate of analyze_dataset_final

# This callback was removed - functionality moved to analyze_dataset_final

# Callback to update button text based on task
@callback(
    Output("analyze-sentence-btn", "children"),
    Input("task-dropdown", "value"),
    prevent_initial_call=True
)
def update_button_text(task):
    """Update button text based on selected task."""
    if task == "ner":
        return "Perform NER"
    elif task == "qa":
        return "Get Answer"
    else:
        return "Analyze Sentiment"

# Callback to update default text based on task
@callback(
    Output("sentence-input", "value"),
    Input("task-dropdown", "value"),
    prevent_initial_call=True
)
def update_default_text(task):
    """Update default text based on selected task."""
    if task == "ner":
        return "Apple Inc. is planning to open a new store in New York City next month. CEO Tim Cook announced this during his visit to Berlin, Germany."
    else:
        return "I really enjoyed this movie. The acting was superb and the story was engaging."

# Callback for feature button clicks
@callback(
    [Output("modal-title", "children"),
     Output("modal-body", "children"),
     Output("visualization-modal", "is_open")],
    [Input({"type": "feature-btn", "index": dash.dependencies.ALL}, "n_clicks")],
    [State("sentence-input", "value"),
     State("ner-sentence-input", "value"),
     State("selected-model-store", "data"),
     State("selected-dataset-store", "data"),
     State("task-dropdown", "value"),
     State("level-toggle", "value")],
    prevent_initial_call=True
)
def handle_feature_button_clicks(n_clicks_list, sentence_text, ner_sentence_text, selected_model, selected_dataset, task, level):
    """Handle feature button clicks and show appropriate visualizations."""
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_data = eval(button_id)  # Convert string back to dict
    feature_id = button_data["index"]
    
    # Determine which text to use based on task
    text_to_analyze = ner_sentence_text if task == "ner" else sentence_text
    
    try:
        if task == "sentiment" and level == "sentence":
            title, content = handle_sentiment_sentence_analysis(f"feature-btn-{feature_id}", text_to_analyze, selected_model)
        elif task == "sentiment" and level == "model":
            title, content = handle_sentiment_model_analysis(f"feature-btn-{feature_id}", selected_dataset, selected_model)
        elif task == "ner" and level == "sentence":
            title, content = handle_ner_sentence_analysis(f"feature-btn-{feature_id}", text_to_analyze, selected_model)
        elif task == "ner" and level == "model":
            title, content = handle_ner_model_analysis(f"feature-btn-{feature_id}", selected_dataset, selected_model)
        else:
            title = "Feature Not Available"
            content = html.Div("This feature is not available for the current task/level combination.", className="text-muted text-center py-3")
        
        return title, content, True
        
    except Exception as e:
        logger.error(f"Error in feature button handler: {str(e)}")
        return "Error", html.Div(f"Error: {str(e)}", className="text-danger"), True

def handle_sentiment_sentence_analysis(button_id, sentence_text, selected_model):
    """Handle sentiment analysis for sentence level."""
    if not sentence_text or not sentence_text.strip():
        return "Input Required", html.Div("Please enter text to analyze.", className="text-muted text-center py-3")

    try:
        if button_id == "feature-btn-lime":
            # LIME analysis
            result = model_api.get_lime_explanation(sentence_text)
            if result and "explanation" in result:
                from components.visualizations import create_lime_bar_chart
                content = create_lime_bar_chart(result["explanation"])
                title = "LIME Explanation"
                return title, content
            else:
                return "LIME Error", html.Div("Error: Could not generate LIME explanation.", className="text-danger")
                
        elif button_id == "feature-btn-attention_entropy":
            # Attention entropy analysis
            result = model_api.get_attention_entropy(sentence_text)
            if result and "entropy" in result:
                from components.visualizations import create_clickable_entropy_heatmap
                entropy_matrix = result["entropy"]
                fig = create_clickable_entropy_heatmap(entropy_matrix)
                content = dcc.Graph(figure=fig, config={'displayModeBar': False})
                title = "Attention Entropy"
                return title, content
            else:
                return "Attention Error", html.Div("Error: Could not retrieve attention data.", className="text-danger")
                
        elif button_id == "feature-btn-token_embeddings":
            # Token embeddings analysis
            result = model_api.get_sentence_embedding(sentence_text)
            if result and "token_embeddings" in result:
                from components.visualizations import create_embedding_plot
                tokens = sentence_text.split()
                embeddings = result["token_embeddings"]
                fig = create_embedding_plot(tokens, embeddings)
                content = dcc.Graph(figure=fig, config={'displayModeBar': False})
                title = "Token Embeddings"
                return title, content
            else:
                return "Embeddings Error", html.Div("Error: Could not retrieve token embeddings.", className="text-danger")
        
        return f"Feature: {button_id}", html.Div(f"Feature {button_id} not implemented yet.", className="text-muted text-center py-3")

    except Exception as e:
        logger.error(f"Error in sentiment sentence analysis: {str(e)}")
        return "Analysis Error", html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")

def handle_sentiment_model_analysis(button_id, selected_dataset, selected_model):
    """Handle sentiment analysis for model level."""
    if not selected_dataset or not selected_model:
        return "Configuration Required", html.Div("Please select a dataset and model first.", className="text-muted text-center py-3")

    try:
        if button_id == "feature-btn-error_analysis":
            # Error analysis - this should NOT automatically open the modal
            # Instead, show instructions or a summary
            content = html.Div([
                html.H5("Error Analysis Instructions"),
                html.P("1. Click 'Analyze Dataset' to run model evaluation"),
                html.P("2. View the confusion matrix and performance metrics"),
                html.P("3. Click on data points in the scatter plot for detailed analysis"),
                html.Hr(),
                html.P("This feature analyzes model errors and provides insights into failure patterns.", 
                       className="text-muted")
            ])
            title = "Error Analysis"
            return title, content
            
        elif button_id == "feature-btn-error_patterns":
            # Error pattern analysis
            content = html.Div([
                html.H5("Error Pattern Analysis"),
                html.P("This feature will analyze systematic error patterns in the model's predictions."),
                html.P("Run dataset analysis first to see error patterns.", className="text-muted")
            ])
            title = "Error Pattern Analysis"
            return title, content
            
        elif button_id == "feature-btn-similarity_analysis":
            # Similarity analysis
            content = html.Div([
                html.H5("Similarity Analysis"),
                html.P("This feature will find similar examples in the dataset based on model representations."),
                html.P("Run dataset analysis first to enable similarity search.", className="text-muted")
            ])
            title = "Similarity Analysis"
            return title, content
        
        return f"Feature: {button_id}", html.Div(f"Feature {button_id} not implemented yet.", className="text-muted text-center py-3")

    except Exception as e:
        logger.error(f"Error in sentiment model analysis: {str(e)}")
        return "Analysis Error", html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")

# Callback for analyze dataset button (model level)
@callback(
    [Output("model-summary-stats", "children", allow_duplicate=True),
     Output("model-confusion-matrix", "children", allow_duplicate=True)],
    [Input("analyze-dataset-btn", "n_clicks")],
    [State("dataset-dropdown", "value"),
     State("sample-size-input", "value"),
     State("confidence-threshold-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def analyze_dataset_final(n_clicks_analyze, selected_dataset, sample_size, confidence_threshold, selected_model):
    """Analyze dataset and show model performance results."""
    logger.info(f"analyze_dataset_final called: clicks={n_clicks_analyze}, dataset={selected_dataset}, model={selected_model}")
    
    if not n_clicks_analyze or not selected_dataset or not selected_model:
        logger.info(f"Early return: clicks={n_clicks_analyze}, dataset={selected_dataset}, model={selected_model}")
        return no_update, no_update
    
    try:
        # Validate dataset selection
        datasets = scan_datasets()
        dataset_info = None
        for task_datasets in datasets.values():
            if selected_dataset in task_datasets:
                dataset_info = task_datasets[selected_dataset]
                break
        
        if not dataset_info:
            error_content = html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.P(f"Dataset {selected_dataset} not found or not properly configured")
            ], className="text-center")
            return error_content, error_content
        
        # Load dataset samples with proper split
        split = 'test' if selected_dataset == 'IMDb' else 'dev'
        samples = load_dataset_samples(selected_dataset, 'sentiment', split=split, max_samples=sample_size)
        
        if not samples:
            raise ValueError(f"No samples found in dataset {selected_dataset}")
        
        # Run sentiment analysis on all samples
        results = []
        for sample in samples:
            text = sample['text']
            true_label = sample['label']
            
            prediction = model_api.get_sentiment(text)
            predicted_label = prediction.get("label", "Unknown")
            confidence = prediction.get("score", 0)
            
            # Convert labels to consistent format
            if str(predicted_label).lower() in ["positive", "1", "1.0"]:
                predicted_label = "1"
            elif str(predicted_label).lower() in ["negative", "0", "0.0"]:
                predicted_label = "0"
            
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": str(predicted_label) == str(true_label)
            })
        
        # Calculate metrics
        y_true = [int(r["true_label"]) for r in results]
        y_pred = [int(r["predicted_label"]) for r in results]
        
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Create summary statistics
        summary_stats = html.Div([
            html.H4(f"Accuracy: {accuracy:.2%}", className="text-center mb-3"),
            html.P([
                f"Total samples: {total}",
                html.Br(),
                f"Correct predictions: {correct}",
                html.Br(),
                f"Incorrect predictions: {total - correct}"
            ], className="text-center")
        ])
        
        # Create confusion matrix
        df = pd.DataFrame(results)
        df['true_label_int'] = df['true_label'].astype(str).map({'0': 0, '1': 1})
        df['predicted_label_int'] = df['predicted_label'].astype(str).map({'0': 0, '1': 1})
        
        confusion_data = df.groupby(['true_label_int', 'predicted_label_int']).size().reset_index(name='count')
        pivot_df = confusion_data.pivot(index='true_label_int', columns='predicted_label_int', values='count')
        
        confusion_fig = px.imshow(
            pivot_df.values,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=['Negative (0)', 'Positive (1)'],
            y=['Negative (0)', 'Positive (1)'],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        confusion_fig.update_layout(
            coloraxis_showscale=False,
            width=400,
            height=400
        )
        
        confusion_matrix_graph = dcc.Graph(figure=confusion_fig)
        
        # Store results for scatter plot access and error pattern analysis
        # Filter high confidence errors (confidence > threshold and incorrect predictions)
        high_conf_errors = []
        for result in results:
            if (not result["correct"] and 
                result["confidence"] > confidence_threshold):
                high_conf_errors.append(result)
        
        # Store analysis results in the analysis store
        analysis_results = {
            "results": results,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "total_samples": len(results),
            "correct_predictions": correct,
            "high_confidence_errors": high_conf_errors
        }
        
        # Clear previous analysis and store new results
        analysis_store.clear_analysis(selected_dataset, selected_model["model_path"])
        analysis_store.store_dataset_analysis(selected_dataset, selected_model["model_path"], analysis_results)
        
        # Create backup file
        try:
            import json
            import os
            
            # Ensure TempFiles directory exists
            temp_dir = "TempFiles"
            os.makedirs(temp_dir, exist_ok=True)
            
            backup_data = {
                "dataset": selected_dataset,
                "model_path": selected_model["model_path"],
                "results": analysis_results,
                "timestamp": time.time()
            }
            backup_file_path = os.path.join(temp_dir, "analysis_backup.json")
            with open(backup_file_path, "w") as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Analysis backup saved to: {backup_file_path}")
            print(f"BACKUP SUCCESS: Analysis backup saved to: {backup_file_path}")
            
        except Exception as backup_error:
            logger.error(f"Failed to create backup file: {str(backup_error)}")
            print(f"BACKUP FAILED: {str(backup_error)}")
        
        logger.info(f"Stored analysis results for {selected_dataset} with {len(high_conf_errors)} high-confidence errors")
        
        return summary_stats, confusion_matrix_graph
        
    except Exception as e:
        logger.error(f"Error in dataset analysis: {str(e)}")
        error_content = html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")
        return error_content, error_content

# Callback to update scatter plot based on toggle (for popup) - FILTER ONLY
@callback(
    Output("error-scatter-plot", "figure", allow_duplicate=True),
    Input("error-display-toggle", "value"),
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def update_scatter_plot(show_type, selected_dataset, selected_model):
    """Update scatter plot based on toggle selection - ONLY FILTER EXISTING DATA."""
    print(f"DEBUG: Scatter plot filter triggered - show_type={show_type}")
    
    if not selected_dataset or not selected_model:
        return no_update
    
    try:
        # Load from backup file (same as Error Analysis popup)
        import json
        import os
        backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
        
        if not os.path.exists(backup_file_path):
            print("DEBUG: No backup file found for filtering")
            placeholder_fig = go.Figure()
            placeholder_fig.add_annotation(
                text="No analysis data found. Please run 'Analyze Dataset' first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            placeholder_fig.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return placeholder_fig
            
        with open(backup_file_path, "r") as f:
            backup_data = json.load(f)
            
        # Get the analysis results
        analysis_results = backup_data.get("results", {})
        results = analysis_results.get("results", [])
        
        print(f"DEBUG: Filtering {len(results)} results, show_type={show_type}")
        
        # Filter results based on display mode
        if show_type == "wrong":
            filtered_results = [r for r in results if not r.get("correct", True)]
        else:
            filtered_results = results
        
        print(f"DEBUG: After filtering: {len(filtered_results)} results")
        
        if not filtered_results:
            placeholder_fig = go.Figure()
            if show_type == "wrong":
                placeholder_fig.add_annotation(
                    text="No wrong predictions found! All predictions were correct.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="green")
                )
            else:
                placeholder_fig.add_annotation(
                    text="No results to display.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
            placeholder_fig.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return placeholder_fig
        
        # Create scatter plot
        df = pd.DataFrame(filtered_results)
        
        # Create hover text
        df['hover_text'] = df.apply(lambda row: f"Text: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}<br>True: {'Positive' if str(row['true_label']) == '1' else 'Negative'}<br>Predicted: {'Positive' if str(row['predicted_label']) == '1' else 'Negative'}<br>Confidence: {row['confidence']:.2f}", axis=1)
        
        # Create the scatter plot
        fig = px.scatter(
            df, 
            x=df.index,  # Use index as x-axis (sample number)
            y="confidence", 
            color="correct",
            hover_data={"hover_text": True},
            labels={"x": "Sample Index", "confidence": "Prediction Confidence", "correct": "Correct"},
            title=f"Prediction Analysis - {show_type.title()} Predictions ({len(filtered_results)} samples)",
            color_discrete_map={True: "blue", False: "red"}
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>Sample %{x}</b><br>Confidence: %{y:.2f}<br>%{customdata[0]}<extra></extra>",
            customdata=df[['hover_text']].values
        )
        
        fig.update_layout(
            height=450,
            showlegend=True,
            legend=dict(title="Prediction", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error filtering scatter plot: {str(e)}")
        print(f"DEBUG: Error filtering scatter plot: {str(e)}")
        placeholder_fig = go.Figure()
        placeholder_fig.add_annotation(
            text=f"Error loading plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        placeholder_fig.update_layout(
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return placeholder_fig

# COMBINED CALLBACK: Handle both scatter plot clicks and other point analysis triggers
@callback(
    [Output("selected-point-details", "children"),
     Output("point-analysis-buttons", "children"),
     Output("point-analysis-buttons", "style")],
    [Input("error-scatter-plot", "clickData"),
     Input("dummy-trigger-1", "value"),  # Add other inputs that might trigger this
     Input("dummy-trigger-2", "n_clicks")],  # Add more inputs as needed
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def handle_point_analysis_combined(click_data, dummy1, dummy2, selected_dataset, selected_model):
    """Combined callback to handle scatter plot clicks and other point analysis triggers."""
    
    # Use callback_context to determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div(), [], {"display": "none"}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle different triggers
    if trigger_id == "error-scatter-plot":
        # Handle scatter plot click
        if not click_data or not selected_dataset or not selected_model:
            return html.Div(), [], {"display": "none"}
        
        return handle_scatter_plot_click_logic(click_data, selected_dataset, selected_model)
    
    elif trigger_id == "dummy-trigger-1":
        # Handle other trigger type 1 (if needed in future)
        return html.Div("Trigger 1 activated"), [], {"display": "none"}
    
    elif trigger_id == "dummy-trigger-2":
        # Handle other trigger type 2 (if needed in future)
        return html.Div("Trigger 2 activated"), [], {"display": "none"}
    
    else:
        # Default case
        return html.Div(), [], {"display": "none"}

def handle_scatter_plot_click_logic(click_data, selected_dataset, selected_model):
    """Logic for handling scatter plot clicks."""
    try:
        # Get the clicked point data
        point_index = click_data['points'][0]['x']
        
        # Load from backup file to get the point data
        import json
        import os
        backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
        
        if not os.path.exists(backup_file_path):
            return html.Div("No analysis data found.", className="text-warning"), [], {"display": "none"}
            
        with open(backup_file_path, "r") as f:
            backup_data = json.load(f)
            
        # Get the analysis results
        analysis_results = backup_data.get("results", {})
        results = analysis_results.get("results", [])
        
        if point_index >= len(results):
            return html.Div("Point data not found.", className="text-warning"), [], {"display": "none"}
        
        # Get the specific point data
        point_data = results[point_index]
        text = point_data.get("text", "")
        true_label = point_data.get("true_label", "")
        predicted_label = point_data.get("predicted_label", "")
        confidence = point_data.get("confidence", 0)
        is_correct = point_data.get("correct", False)
        
        # Create point details display
        point_details = html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-crosshairs me-2"),
                        f"Selected Point Analysis (Sample {point_index})"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Strong("Text Sample:"),
                            html.P(text[:200] + "..." if len(text) > 200 else text, className="text-muted mb-2")
                        ], width=12),
                        dbc.Col([
                            html.Strong("True Label:"),
                            html.Span(f" {'Positive' if str(true_label) == '1' else 'Negative'}", 
                                     className="badge bg-info ms-2")
                        ], width=6),
                        dbc.Col([
                            html.Strong("Predicted:"),
                            html.Span(f" {'Positive' if str(predicted_label) == '1' else 'Negative'}", 
                                     className=f"badge {'bg-success' if is_correct else 'bg-danger'} ms-2")
                        ], width=6),
                        dbc.Col([
                            html.Strong("Confidence:"),
                            html.Span(f" {confidence:.2%}", className="badge bg-secondary ms-2")
                        ], width=6),
                        dbc.Col([
                            html.Strong("Result:"),
                            html.Span(f" {'Correct' if is_correct else 'Incorrect'}", 
                                     className=f"badge {'bg-success' if is_correct else 'bg-danger'} ms-2")
                        ], width=6)
                    ])
                ])
            ], className="mb-3")
        ])
        
        # Create analysis buttons
        analysis_buttons = html.Div([
            html.H6("Detailed Analysis Options:", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-search-plus me-2"),
                        "LIME Analysis"
                    ], 
                    id={"type": "point-analysis-btn", "analysis": "lime", "index": point_index},
                    color="primary", className="w-100 mb-2")
                ], width=6),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-eye me-2"),
                        "Attention Entropy"
                    ], 
                    id={"type": "point-analysis-btn", "analysis": "attention", "index": point_index},
                    color="info", className="w-100 mb-2")
                ], width=6),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-vector-square me-2"),
                        "Token Embeddings"
                    ], 
                    id={"type": "point-analysis-btn", "analysis": "embeddings", "index": point_index},
                    color="success", className="w-100 mb-2")
                ], width=6),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-clone me-2"),
                        "Similarity Analysis"
                    ], 
                    id={"type": "point-analysis-btn", "analysis": "similarity", "index": point_index},
                    color="warning", className="w-100 mb-2")
                ], width=6)
            ])
        ])
        
        return point_details, analysis_buttons, {"display": "block"}
        
    except Exception as e:
        logger.error(f"Error handling scatter plot click: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="text-danger"), [], {"display": "none"}

# Callback to handle point analysis button clicks
@callback(
    [Output("feature-modal", "is_open", allow_duplicate=True),
     Output("feature-modal-title", "children", allow_duplicate=True),
     Output("feature-modal-body", "children", allow_duplicate=True)],
    [Input({"type": "point-analysis-btn", "analysis": ALL, "index": ALL}, "n_clicks")],
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def handle_point_analysis_click(n_clicks_list, selected_dataset, selected_model):
    """Handle point analysis button clicks to show detailed analysis."""
    if not any(n_clicks_list) or not selected_dataset or not selected_model:
        return False, "", html.Div()
    
    # Get the triggered button
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, "", html.Div()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    button_data = eval(button_id)  # Convert string back to dict
    
    analysis_type = button_data["analysis"]
    point_index = button_data["index"]
    
    try:
        # Load from backup file to get the point data
        import json
        import os
        backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
        
        if not os.path.exists(backup_file_path):
            return True, "Error", html.Div("No analysis data found.", className="text-danger")
            
        with open(backup_file_path, "r") as f:
            backup_data = json.load(f)
            
        # Get the analysis results
        analysis_results = backup_data.get("results", {})
        results = analysis_results.get("results", [])
        
        if point_index >= len(results):
            return True, "Error", html.Div("Point data not found.", className="text-danger")
        
        # Get the specific point data
        point_data = results[point_index]
        text = point_data.get("text", "")
        
        # Set the model for analysis
        model_api.set_selected_model(selected_model["model_path"], selected_model["model_type"])
        
        # Handle different analysis types
        if analysis_type == "lime":
            title = f"LIME Analysis - Sample {point_index}"
            try:
                lime_result = model_api.get_lime_explanation(text, num_features=10, num_samples=1000)
                
                # Create LIME visualization
                content = html.Div([
                    html.H6("LIME Feature Importance", className="mb-3"),
                    html.P(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}", className="text-muted mb-3"),
                    
                    # Feature importance table
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Feature"),
                                html.Th("Importance"),
                                html.Th("Impact")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(feature),
                                html.Td(f"{importance:.3f}"),
                                html.Td(
                                    html.Span("Positive" if importance > 0 else "Negative",
                                             className=f"badge {'bg-success' if importance > 0 else 'bg-danger'}")
                                )
                            ]) for feature, importance in lime_result.get("feature_importance", [])[:10]
                        ])
                    ], striped=True, hover=True)
                ])
                
            except Exception as e:
                content = html.Div(f"Error generating LIME analysis: {str(e)}", className="text-danger")
                
        elif analysis_type == "attention":
            title = f"Attention Entropy - Sample {point_index}"
            try:
                attention_result = model_api.get_attention_entropy(text)
                
                content = html.Div([
                    html.H6("Attention Entropy Analysis", className="mb-3"),
                    html.P(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}", className="text-muted mb-3"),
                    html.P(f"Average Attention Entropy: {attention_result.get('avg_entropy', 'N/A')}", className="mb-2"),
                    html.P("Attention entropy measures how focused or diffuse the model's attention is across tokens.", className="text-info")
                ])
                
            except Exception as e:
                content = html.Div(f"Error generating attention analysis: {str(e)}", className="text-danger")
                
        elif analysis_type == "embeddings":
            title = f"Token Embeddings - Sample {point_index}"
            try:
                embedding_result = model_api.get_sentence_embedding(text)
                
                content = html.Div([
                    html.H6("Token Embeddings Analysis", className="mb-3"),
                    html.P(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}", className="text-muted mb-3"),
                    html.P(f"Sentence embedding shape: {embedding_result.get('sentence_embedding_shape', 'N/A')}", className="mb-2"),
                    html.P(f"Number of tokens: {len(embedding_result.get('token_embeddings', []))}", className="mb-2"),
                    html.P("Token embeddings represent the semantic meaning of individual words in the context.", className="text-info")
                ])
                
            except Exception as e:
                content = html.Div(f"Error generating embeddings analysis: {str(e)}", className="text-danger")
                
        elif analysis_type == "similarity":
            title = f"Similarity Analysis - Sample {point_index}"
            content = html.Div([
                html.H6("Similarity Analysis", className="mb-3"),
                html.P(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}", className="text-muted mb-3"),
                html.P("Finding similar examples in the dataset...", className="text-info"),
                html.P("This feature compares the selected sample with other examples in the dataset to find similar patterns.", className="text-muted")
            ])
            
        else:
            title = f"Analysis - Sample {point_index}"
            content = html.Div(f"Analysis type '{analysis_type}' not implemented yet.", className="text-warning")
        
        return True, title, content
        
    except Exception as e:
        logger.error(f"Error in point analysis: {str(e)}")
        return True, "Error", html.Div(f"Error: {str(e)}", className="text-danger")

# Helper function to create attention visualization
def create_attention_viz(text, layer=0, head=0):
    """Create attention visualization for a given text."""
    try:
        # Get attention weights from model
        attention_data = model_api.get_attention_weights(text, layer, head)
        
        if attention_data and 'tokens' in attention_data and 'attention_weights' in attention_data:
            tokens = attention_data['tokens']
            attentions = attention_data['attention_weights']
            
            fig = create_attention_heatmap_matrix(tokens, attentions, layer, head)
            
            # Update layout for better sizing
            fig.update_layout(
                height=400,  # Fixed height
                margin=dict(l=50, r=50, t=50, b=50),
                autosize=True
            )
            
            return fig
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error creating attention visualization: {str(e)}")
        return None

def create_grid_view_content(view_type, text, selected_model):
    """Create content for grid view popup with layer selector."""
    return html.Div([
        # Layer selector at top
        dbc.Row([
            dbc.Col([
                html.Label("Select Layer:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="grid-layer-dropdown",
                    options=[{"label": f"Layer {i}", "value": i} for i in range(12)],
                    value=0,
                    className="mb-3"
                )
            ], width=4)
        ]),
        
        # Grid visualization area
        html.Div(id="grid-visualization-area", children=[
            create_grid_visualization(view_type, text, selected_model, 0)
        ])
    ])

def create_grid_visualization(view_type, text, selected_model, layer):
    """Create the actual grid visualization."""
    try:
        # Get attention weights from model
        attention_data = model_api.get_attention_weights(text, layer, 0)  # Get all heads for layer
        
        if attention_data and 'tokens' in attention_data and 'attention_weights' in attention_data:
            tokens = attention_data['tokens']
            attentions = attention_data['attention_weights']
            
            if view_type == "heads":
                # Use our local function for all heads grid (line visualization)
                from components.visualizations import create_attention_all_heads_grid
                fig = create_attention_all_heads_grid(tokens, attentions, layer)
            else:  # matrices
                # Use our local function for all matrices grid (heatmap visualization)
                from components.visualizations import create_attention_all_matrices_grid
                fig = create_attention_all_matrices_grid(attentions, tokens, layer)
            
            # Update layout for proper modal sizing
            fig.update_layout(
                height=700,  # Larger height for grid views
                margin=dict(l=50, r=50, t=80, b=50),
                autosize=True,
                showlegend=True
            )
            
            from dash import dcc
            return dcc.Graph(
                figure=fig, 
                style={"height": "700px", "width": "100%", "maxWidth": "100%"},
                config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}
            )
        else:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                html.H5("No Attention Data Available", className="text-warning"),
                html.P("Please ensure the model is loaded and supports attention extraction.")
            ], className="text-center py-5")
            
    except Exception as e:
        logger.error(f"Error creating grid visualization: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Grid Visualization Error", className="text-danger"),
            html.P(f"Error: {str(e)}")
        ], className="text-center py-5")

# Callback for grid layer selection
@callback(
    Output("grid-visualization-area", "children", allow_duplicate=True),
    [Input("grid-layer-dropdown", "value")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data"),
     State("modal-title", "children")],
    prevent_initial_call=True
)
def update_grid_visualization(layer, sentence_text, selected_model, modal_title):
    """Update grid visualization when layer changes."""
    if not sentence_text or not selected_model or not modal_title:
        return dash.no_update
    
    # Determine view type from modal title
    view_type = "heads" if "Heads" in modal_title else "matrices"
    
    return create_grid_visualization(view_type, sentence_text, selected_model, layer)

# Callback for attention view buttons
@callback(
    Output("attention-visualization-content", "children", allow_duplicate=True),
    [Input("view-matrix-btn", "n_clicks"),
     Input("view-line-btn", "n_clicks"),
     Input("layer-dropdown", "value"),
     Input("head-dropdown", "value")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def update_attention_view(matrix_clicks, line_clicks, layer, head, sentence_text, selected_model):
    """Update attention visualization based on view selection."""
    ctx = dash.callback_context
    if not ctx.triggered or not sentence_text or not selected_model:
        return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "view-matrix-btn" or button_id == "layer-dropdown" or button_id == "head-dropdown":
        return create_attention_matrix_visualization(sentence_text, selected_model, layer, head)
    elif button_id == "view-line-btn":
        return create_attention_line_graph(sentence_text, selected_model, layer, head)
    
    return dash.no_update

def create_attention_matrix_visualization(text, selected_model, layer=0, head=0):
    """Create attention matrix visualization."""
    try:
        # Get attention weights from model
        attention_data = model_api.get_attention_weights(text, layer, head)
        
        if attention_data and 'tokens' in attention_data and 'attention_weights' in attention_data:
            tokens = attention_data['tokens']
            attentions = attention_data['attention_weights']
            
            # Use our local visualization function
            from components.visualizations import create_attention_heatmap_matrix
            fig = create_attention_heatmap_matrix(tokens, attentions, layer, head)
            
            # Update layout for better sizing
            fig.update_layout(
                height=400,  # Fixed height
                margin=dict(l=50, r=50, t=50, b=50),
                autosize=True
            )
            
            from dash import dcc
            return dcc.Graph(
                figure=fig,
                style={"height": "400px", "width": "100%"},
                config={'displayModeBar': True, 'displaylogo': False}
            )
        else:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                html.H5("No Attention Data Available", className="text-warning"),
                html.P("Please ensure the model is loaded and supports attention extraction.")
            ], className="text-center py-5")
        
    except Exception as e:
        logger.error(f"Error creating attention matrix: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Attention Visualization Error", className="text-danger"),
            html.P(f"Error: {str(e)}")
        ], className="text-center py-5")

def create_attention_line_graph(text, selected_model, layer=0, head=0):
    """Create attention line graph visualization."""
    try:
        # Get attention weights from model
        attention_data = model_api.get_attention_weights(text, layer, head)
        
        if attention_data and 'tokens' in attention_data and 'attention_weights' in attention_data:
            tokens = attention_data['tokens']
            attentions = attention_data['attention_weights']
            
            # Use our local visualization function
            from components.visualizations import create_attention_heatmap_lines
            fig = create_attention_heatmap_lines(tokens, attentions, layer, head)
            
            # Update layout for better sizing
            fig.update_layout(
                height=400,  # Fixed height
                margin=dict(l=50, r=50, t=50, b=50),
                autosize=True
            )
            
            from dash import dcc
            return dcc.Graph(
                figure=fig,
                style={"height": "400px", "width": "100%"},
                config={'displayModeBar': True, 'displaylogo': False}
            )
        else:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                html.H5("No Attention Data Available", className="text-warning"),
                html.P("Please ensure the model is loaded and supports attention extraction.")
            ], className="text-center py-5")
        
    except Exception as e:
        logger.error(f"Error creating attention line graph: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Attention Visualization Error", className="text-danger"),
            html.P(f"Error: {str(e)}")
        ], className="text-center py-5")

# Callback to manage dataset dropdown state and selection
@callback(
    [Output("dataset-dropdown", "options", allow_duplicate=True),
     Output("dataset-dropdown", "value", allow_duplicate=True),
     Output("dataset-dropdown", "disabled", allow_duplicate=True)],
    [Input("task-dropdown", "value"),
     Input("selected-model-store", "data")],
    [State("dataset-dropdown", "value"),
     State("selected-dataset-store", "data")],
    prevent_initial_call=True
)
def update_dataset_dropdown(task, selected_model, current_dataset, stored_dataset):
    """Update dataset dropdown and maintain selection if valid."""
    # Disable dropdown if no model is selected
    if not selected_model:
        return [], None, True

    try:
        datasets = scan_datasets()
        task_datasets = datasets.get(task, {})
        
        if not task_datasets:
            logger.warning(f"No datasets found for task: {task}")
            return [], None, True

        # Build options list
        options = []
        for dataset_key, dataset_info in task_datasets.items():
            options.append({
                "label": dataset_info.get("display_name", dataset_key),
                "value": dataset_key
            })
        
        if not options:
            logger.warning("No dataset options created")
            return [], None, True

        # Determine the value to use (prioritize stored > current > first)
        if stored_dataset and isinstance(stored_dataset, dict) and stored_dataset.get("key"):
            if any(opt["value"] == stored_dataset["key"] for opt in options):
                value = stored_dataset["key"]
            else:
                value = options[0]["value"]
        elif current_dataset and any(opt["value"] == current_dataset for opt in options):
            value = current_dataset
        else:
            value = options[0]["value"]

        logger.info(f"Dataset dropdown updated: {len(options)} options, selected: {value}")
        return options, value, False

    except Exception as e:
        logger.exception(f"Error updating dataset dropdown: {str(e)}")
        return [], None, True

# Generate feature button IDs dynamically based on current layout
def get_current_feature_button_ids():
    """Get feature button IDs that actually exist in the current layout."""
    # Only include buttons that are actually rendered
    current_ids = []
    for task_features in FEATURE_CONFIG.values():
        for level_features in task_features.values():
            for feature in level_features:
                current_ids.append(f"feature-btn-{feature['id']}")
    return current_ids

# Callback for visualization modal (grid view buttons only)
@callback(
    [Output("visualization-modal", "is_open", allow_duplicate=True),
     Output("modal-title", "children", allow_duplicate=True),
     Output("modal-body", "children", allow_duplicate=True),
     Output("current-analysis-store", "data", allow_duplicate=True)],
    [Input("view-all-heads-btn", "n_clicks"),
     Input("view-all-matrices-btn", "n_clicks")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def handle_grid_view_popups(heads_clicks, matrices_clicks, sentence_text, selected_model):
    """Handle grid view button clicks and open the modal with a loading spinner."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Check if any button was actually clicked (not just None values)
    if not heads_clicks and not matrices_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    logger.info(f"Grid view button clicked: {button_id}")

    if not sentence_text or not selected_model:
        return True, "Input Required", html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
            html.H5("Input Required", className="text-warning"),
            html.P("Please enter text and select a model first.")
        ], className="text-center py-5"), dash.no_update

    loading_content = html.Div([
        dbc.Spinner(size="lg", color="primary"),
        html.H5("Loading visualization...", className="mt-3")
    ], className="text-center py-5")

    analysis_data = {
        "type": "heads" if button_id == "view-all-heads-btn" else "matrices",
        "timestamp": time.time()  # Trigger the background callback
    }

    if button_id == "view-all-heads-btn":
        return True, "All Attention Heads Grid", loading_content, analysis_data
    
    if button_id == "view-all-matrices-btn":
        return True, "All Attention Matrices Grid", loading_content, analysis_data

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback for model-level feature buttons (excluding error_patterns which has its own modal)
@callback(
    [Output("visualization-modal", "is_open", allow_duplicate=True),
     Output("modal-title", "children", allow_duplicate=True),
     Output("modal-body", "children", allow_duplicate=True)],
    [Input("feature-btn-error_analysis", "n_clicks")],
    [State("selected-model-store", "data"),
     State("task-dropdown", "value"),
     State("level-toggle", "value"),
     State("dataset-dropdown", "value")],
    prevent_initial_call=True
)
def handle_model_feature_analysis(error_analysis_clicks, selected_model, task, level, selected_dataset):
    """Handle model-level feature button clicks (excluding error_patterns)."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    logger.info(f"Model feature button clicked: {button_id}")
    
    # Only handle if we're in model level and button was actually clicked
    if level != "model" or not error_analysis_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    if task == "sentiment":
        title, content = handle_sentiment_model_analysis(button_id, selected_dataset, selected_model)
        return True, title, content
    
    return dash.no_update, dash.no_update, dash.no_update

# Callback for analyze sentence button
@callback(
    Output("attention-visualization-content", "children", allow_duplicate=True),
    [Input("analyze-sentence-btn", "n_clicks")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data"),
     State("task-dropdown", "value")],
    prevent_initial_call=True
)
def handle_analyze_sentence(n_clicks, sentence_text, selected_model, task):
    """Handle analyze sentence button click."""
    if not n_clicks or not sentence_text or not selected_model:
        return dash.no_update
    
    logger.info(f"Analyze sentence button clicked for task: {task}")
    
    try:
        if task == "sentiment":
            # Get attention weights and create matrix visualization
            return create_attention_matrix_visualization(sentence_text, selected_model, 0, 0)
        elif task == "ner":
            # For NER, don't do anything in this callback - NER has its own button
            logger.info("NER task detected in analyze sentence - doing nothing")
            return dash.no_update
    except Exception as e:
        logger.error(f"Error in analyze sentence: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")
    
    return dash.no_update

# Background callback to generate the grid visualization
@callback(
    Output("modal-body", "children", allow_duplicate=True),
    Input("current-analysis-store", "data"),
    [State("sentence-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def generate_grid_visualization(analysis_data, sentence_text, selected_model):
    """Generate the grid visualization in the background."""
    if not analysis_data or not sentence_text or not selected_model:
        return dash.no_update

    view_type = analysis_data.get("type")
    logger.info(f"Generating visualization for type: {view_type}")

    try:
        # Handle grid visualizations only
        if view_type in ["heads", "matrices"]:
            content = create_grid_visualization(view_type, sentence_text, selected_model, 0)
            return content
        else:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                html.H5("Unknown Visualization Type", className="text-warning"),
                html.P(f"Unknown type: {view_type}")
            ], className="text-center py-5")
            
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Visualization Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")

# Callback for all feature buttons (using pattern-matching to handle dynamic buttons)
@callback(
    [Output("visualization-modal", "is_open", allow_duplicate=True),
     Output("modal-title", "children", allow_duplicate=True),
     Output("modal-body", "children", allow_duplicate=True)],
    [Input("close-modal", "n_clicks"),
     Input({"type": "feature-btn", "index": dash.dependencies.ALL}, "n_clicks")],
    [State("sentence-input", "value"),
     State("ner-sentence-input", "value"),
     State("selected-model-store", "data"),
     State("task-dropdown", "value"),
     State("level-toggle", "value"),
     State("visualization-modal", "is_open"),
     State("dataset-dropdown", "value")],
    prevent_initial_call=True
)
def handle_feature_analysis(*args):
    """Handle feature button clicks and open popup modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Debug logging
    triggered_prop = ctx.triggered[0]["prop_id"]
    triggered_value = ctx.triggered[0]["value"]
    logger.info(f"Feature button clicked: {triggered_prop}, value: {triggered_value}")
    
    # With pattern-matching inputs, we know the structure:
    # args[0] = close-modal n_clicks
    # args[1] = list of feature button n_clicks (pattern-matching)
    # args[2-8] = States (7 states total)
    
    close_modal_n_clicks = args[0]
    feature_button_clicks = args[1]  # This is a list
    
    # Extract states (last 7 arguments)
    sentence_text = args[-7]
    ner_sentence_text = args[-6]
    selected_model = args[-5]
    task = args[-4]
    level = args[-3]
    is_open = args[-2]
    selected_dataset = args[-1]
    
    # Determine which button was clicked
    button_id = None
    if "close-modal" in triggered_prop:
        button_id = "close-modal"
    else:
        # Parse the pattern-matching ID to get the feature type
        import json
        try:
            parsed_id = json.loads(triggered_prop.split(".")[0])
            if parsed_id.get("type") == "feature-btn":
                button_id = f"feature-btn-{parsed_id.get('index')}"
                # Check if this button was actually clicked (value > 0) and not just initialized
                # Check if button was actually clicked (not just initialized)
                if triggered_value is None or triggered_value == 0:
                    logger.info(f"Button {button_id} was not actually clicked (value: {triggered_value})")
                    return dash.no_update, dash.no_update, dash.no_update
        except:
            logger.error(f"Could not parse button ID: {triggered_prop}")
            return dash.no_update, dash.no_update, dash.no_update
    
    logger.info(f"Processing button: {button_id}")
    logger.info(f"Task: {task}, Level: {level}, Model: {selected_model is not None}")
    logger.info(f"Sentence text: {sentence_text[:50] if sentence_text else 'None'}...")
    
    # Handle close modal
    if button_id == "close-modal":
        return False, dash.no_update, dash.no_update
    
    # Only open modal if a feature button was actually clicked (not just on page load)
    if not button_id or not button_id.startswith("feature-btn-"):
        logger.info("No feature button clicked, keeping modal closed")
        return dash.no_update, dash.no_update, dash.no_update
    
    # Handle feature buttons for sentence-level analysis
    # For NER, use ner_sentence_text; for sentiment, use sentence_text
    current_text = ner_sentence_text if task == "ner" else sentence_text
    
    logger.info(f"DEBUG: Checking conditions - task={task}, level={level}")
    logger.info(f"DEBUG: current_text={current_text is not None}, selected_model={selected_model is not None}")
    print(f"DEBUG: Checking conditions - task={task}, level={level}")
    print(f"DEBUG: current_text={current_text is not None}, selected_model={selected_model is not None}")
    
    # For model-level analysis, we don't need current_text
    if level == "model":
        logger.info(f"DEBUG: Model-level analysis, skipping text check")
        print(f"DEBUG: Model-level analysis, skipping text check")
    elif not current_text or not selected_model:
        logger.info(f"DEBUG: Returning Input Required - current_text={current_text}, selected_model={selected_model}")
        print(f"DEBUG: Returning Input Required - current_text={current_text}, selected_model={selected_model}")
        return True, "Input Required", html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
            html.H5("Input Required", className="text-warning"),
            html.P("Please enter text and select a model first.")
        ], className="text-center py-5")
    
    # Check if model is selected for other features
    if not selected_model:
        return True, "No Model Selected", html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
            html.H5("No Model Selected", className="text-warning"),
            html.P("Please select a model from the dropdown above.")
        ], className="text-center py-5")
    
    # Handle different feature buttons and analysis types
    try:
        logger.info(f"DEBUG: Entering button handling logic - task={task}, level={level}")
        print(f"DEBUG: Entering button handling logic - task={task}, level={level}")
        
        # Handle buttons based on task and level
        if task == "sentiment" and level == "sentence":
            # Special handling for logit matrix - use regular modal with logit content
            if button_id == "feature-btn-logit_matrix":
                title = "Logit Matrix Heatmap"
                content = create_logit_matrix_content(sentence_text, selected_model, "sentiment")
                return True, title, content
            title, content = handle_sentiment_sentence_analysis(button_id, sentence_text, selected_model)
        elif task == "sentiment" and level == "model":
            if button_id == "feature-btn-error_analysis":
                logger.info(f"DEBUG: Calling handle_sentiment_model_analysis with dataset={selected_dataset}, model={selected_model}")
                print(f"DEBUG: Calling handle_sentiment_model_analysis with dataset={selected_dataset}, model={selected_model}")
                
                # INLINE ERROR ANALYSIS FUNCTION - NO IMPORTS NEEDED
                print(f"DEBUG Error Analysis: Function called with button_id={button_id}")
                
                try:
                    import json
                    import os
                    backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
                    print(f"DEBUG Error Analysis: Looking for backup at: {backup_file_path}")
                    
                    if os.path.exists(backup_file_path):
                        print("DEBUG Error Analysis: Backup file exists! Loading...")
                        with open(backup_file_path, "r") as f:
                            backup_data = json.load(f)
                            
                        # Get the analysis results
                        analysis_results = backup_data.get("results", {})
                        results = analysis_results.get("results", [])
                        accuracy = analysis_results.get("accuracy", 0)
                        total_samples = analysis_results.get("total_samples", 0)
                        
                        print(f"DEBUG Error Analysis: Found {len(results)} results, accuracy: {accuracy}")
                        
                        # Create proper Error Analysis content like error_analysis_old.py
                        correct_predictions = analysis_results.get("correct_predictions", 0)
                        
                        # Filter wrong predictions for scatter plot
                        wrong_results = [r for r in results if not r.get("correct", True)]
                        
                        # Create scatter plot data
                        if wrong_results:
                            df = pd.DataFrame(wrong_results)
                            df['hover_text'] = df.apply(lambda row: f"Text: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}<br>True: {'Positive' if str(row['true_label']) == '1' else 'Negative'}<br>Predicted: {'Positive' if str(row['predicted_label']) == '1' else 'Negative'}<br>Confidence: {row['confidence']:.2f}", axis=1)
                            
                            # Create the scatter plot
                            fig = px.scatter(
                                df, 
                                x=df.index,
                                y="confidence", 
                                color="correct",
                                hover_data={"hover_text": True},
                                labels={"x": "Sample Index", "confidence": "Prediction Confidence", "correct": "Correct"},
                                title=f"Error Analysis - Wrong Predictions ({len(wrong_results)} errors out of {total_samples} samples)",
                                color_discrete_map={True: "blue", False: "red"}
                            )
                            
                            fig.update_traces(
                                hovertemplate="<b>Sample %{x}</b><br>Confidence: %{y:.2f}<br>%{customdata[0]}<extra></extra>",
                                customdata=df[['hover_text']].values
                            )
                            
                            fig.update_layout(
                                height=450,
                                showlegend=True,
                                legend=dict(title="Prediction", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            scatter_plot = dcc.Graph(
                                id="error-scatter-plot", 
                                figure=fig,
                                style={"width": "100%", "height": "450px"},
                                config={'responsive': True, 'displayModeBar': False}
                            )
                        else:
                            scatter_plot = html.Div([
                                html.I(className="fas fa-check-circle fa-3x text-success mb-3"),
                                html.H5("No Errors Found", className="text-success"),
                                html.P("All predictions were correct!", className="text-muted")
                            ], className="text-center py-5")
                        
                        # Create the main Error Analysis layout with custom styling
                        content = html.Div([
                            # Header Section with custom styling
                            html.Div([
                                html.H4([
                                    html.I(className="fas fa-exclamation-triangle me-2"),
                                    "Error Analysis Dashboard"
                                ], className="mb-2"),
                                html.P("Interactive analysis of model prediction errors with detailed insights", 
                                       className="error-analysis-header-subtitle")
                            ], className="error-analysis-header"),
                            
                            # Stats Section with custom styling
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H4(f"{accuracy:.1%}", className="error-analysis-stat-number text-info"),
                                            html.P("Accuracy", className="error-analysis-stat-label")
                                        ], className="error-analysis-stat-card")
                                    ], width=3),
                                    dbc.Col([
                                        html.Div([
                                            html.H4(f"{total_samples}", className="error-analysis-stat-number text-primary"),
                                            html.P("Total Samples", className="error-analysis-stat-label")
                                        ], className="error-analysis-stat-card")
                                    ], width=3),
                                    dbc.Col([
                                        html.Div([
                                            html.H4(f"{correct_predictions}", className="error-analysis-stat-number text-success"),
                                            html.P("Correct", className="error-analysis-stat-label")
                                        ], className="error-analysis-stat-card")
                                    ], width=3),
                                    dbc.Col([
                                        html.Div([
                                            html.H4(f"{total_samples - correct_predictions}", className="error-analysis-stat-number text-danger"),
                                            html.P("Errors", className="error-analysis-stat-label")
                                        ], className="error-analysis-stat-card")
                                    ], width=3)
                                ])
                            ], className="error-analysis-stats-row"),
                            
                            # Controls Section with custom styling
                            html.Div([
                                html.Div([
                                    html.Label("Display Mode:", className="error-analysis-toggle-label"),
                                    dcc.RadioItems(
                                        id="error-display-toggle",
                                        options=[
                                            {"label": "Show All Predictions", "value": "all"},
                                            {"label": "Show Only Errors", "value": "wrong"}
                                        ],
                                        value="wrong",
                                        inline=True,
                                        className="error-analysis-toggle-group"
                                    )
                                ])
                            ], className="error-analysis-controls"),
                            
                            # Scatter Plot
                            html.Div([scatter_plot], className="mb-4"),
                            
                            # Selected Point Details (for point analysis)
                            html.Div(id="selected-point-details", className="mb-3"),
                            
                            # Analysis Buttons Section (initially hidden)
                            html.Div(id="point-analysis-buttons", children=[], style={"display": "none"}),
                            
                            # Loading and Results containers for detailed analysis
                            html.Div(id="analysis-loading-container", className="mt-3"),
                            html.Div(id="analysis-results-container", className="mt-3"),
                            
                            # Hidden dummy components for combined callback
                            html.Div(id="dummy-trigger-1", style={"display": "none"}),
                            html.Div(id="dummy-trigger-2", style={"display": "none"}),
                            
                            # Instructions Section
                            html.Div([
                                dbc.Accordion([
                                    dbc.AccordionItem([
                                        html.Div([
                                            html.H6([
                                                html.I(className="fas fa-info-circle me-2"),
                                                "How to Use Error Analysis"
                                            ]),
                                            html.P([
                                                html.Strong("Interactive Analysis: "),
                                                "Click on any point in the scatter plot to open detailed analysis with LIME, Attention Entropy, Token Embeddings, and Similarity Analysis."
                                            ]),
                                            html.Ul([
                                                html.Li([html.Strong("Red Points:"), " Incorrect predictions - focus areas for improvement"]),
                                                html.Li([html.Strong("Blue Points:"), " Correct predictions - model confidence validation"]),
                                                html.Li([html.Strong("Position:"), " X-axis shows sample index, Y-axis shows prediction confidence"]),
                                            ]),
                                            html.P([
                                                "Use the display mode toggle to switch between viewing only errors or all predictions. ",
                                                "This helps identify patterns in model behavior and areas for improvement."
                                            ])
                                        ])
                                    ], title="Instructions", item_id="instructions")
                                ], start_collapsed=True, className="mt-3")
                            ])
                        ], className="error-analysis-modal-container")
                        
                        print(f"DEBUG Error Analysis: Returning content successfully")
                        title = "Error Analysis"
                        
                    else:
                        print("DEBUG Error Analysis: Backup file does not exist")
                        title = "Error Analysis"
                        content = html.Div("Backup file not found. Please run analysis first.", className="text-warning")
                        
                except Exception as e:
                    print(f"DEBUG Error Analysis: Exception: {e}")
                    import traceback
                    traceback.print_exc()
                    title = "Error Analysis"
                    content = html.Div(f"Error loading analysis: {str(e)}", className="text-danger")
                
                logger.info(f"DEBUG: handle_sentiment_model_analysis returned title={title}")
                print(f"DEBUG: handle_sentiment_model_analysis returned title={title}")
            elif button_id == "feature-btn-similarity_analysis":
                title, original_content = handle_similarity_analysis(button_id, selected_dataset, selected_model)
                # Create better UI wrapper with Bootstrap classes
                content = html.Div([
                    # Header with gradient background
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-project-diagram me-2"),
                            title
                        ], className="text-white mb-2"),
                        html.P("Find similar examples and analyze clustering patterns in your dataset", 
                               className="text-white-50 mb-0")
                    ], className="p-4 mb-4", style={
                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        "borderRadius": "15px 15px 0 0"
                    }),
                    
                    # Content with better styling
                    html.Div([
                        original_content
                    ], className="p-4", style={
                        "background": "rgba(255, 255, 255, 0.98)",
                        "borderRadius": "0 0 15px 15px",
                        "minHeight": "400px"
                    })
                ], style={
                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    "borderRadius": "15px",
                    "overflow": "hidden",
                    "boxShadow": "0 10px 30px rgba(0,0,0,0.2)"
                })
            elif button_id == "feature-btn-error_patterns":
                title, original_content = handle_error_patterns_analysis(button_id, selected_dataset, selected_model)
                # Create better UI wrapper with Bootstrap classes
                content = html.Div([
                    # Header with gradient background
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-search me-2"),
                            title
                        ], className="text-white mb-2"),
                        html.P("Systematic analysis of error patterns to identify common failure modes", 
                               className="text-white-50 mb-0")
                    ], className="p-4 mb-4", style={
                        "background": "linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%)",
                        "borderRadius": "15px 15px 0 0"
                    }),
                    
                    # Content with better styling
                    html.Div([
                        original_content
                    ], className="p-4", style={
                        "background": "rgba(255, 255, 255, 0.98)",
                        "borderRadius": "0 0 15px 15px",
                        "minHeight": "400px"
                    })
                ], style={
                    "background": "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
                    "borderRadius": "15px",
                    "overflow": "hidden",
                    "boxShadow": "0 10px 30px rgba(0,0,0,0.2)"
                })
            else:
                title, content = handle_sentiment_model_analysis(button_id, selected_dataset, selected_model)
        elif task == "ner" and level == "sentence":
            # Special handling for logit matrix - use regular modal with logit content
            if button_id == "feature-btn-logit_matrix":
                title = "Logit Matrix Heatmap"
                content = create_logit_matrix_content(ner_sentence_text, selected_model, "ner")
                return True, title, content
            title, content = handle_ner_sentence_analysis(button_id, ner_sentence_text, selected_model)
        elif task == "ner" and level == "model":
            title, content = handle_ner_model_analysis(button_id, selected_dataset, selected_model)
        elif task == "qa" and level == "sentence":
            # Special handling for QA knowledge assessment
            if button_id == "feature-btn-knowledge_assessment":
                title = "QA Knowledge Assessment"
                content = handle_qa_knowledge_assessment(current_text, selected_model)
                return True, title, content
            elif button_id == "feature-btn-knowledge_competition":
                title = "Knowledge Competition Analysis"
                content = handle_knowledge_competition_analysis(current_text, selected_model)
                return True, title, content
            elif button_id == "feature-btn-model_viz":
                title = "Model Visualization"
                content = handle_qa_model_viz(current_text, selected_model)
                return True, title, content
            elif button_id == "feature-btn-counterfactual_flow":
                title = "Counterfactual Data Flow"
                content = handle_qa_counterfactual_flow(current_text, selected_model)
                return True, title, content
            title, content = handle_qa_sentence_analysis(button_id, current_text, selected_model)
        elif task == "qa" and level == "model":
            title, content = handle_qa_model_analysis(button_id, selected_dataset, selected_model)
        else:
            title = "Feature Not Implemented"
            content = html.Div("This feature is not implemented yet.", className="text-muted text-center py-3")
        
        return True, title, content
            
    except Exception as e:
        logger.error(f"Error in feature analysis: {str(e)}")
        return True, "Analysis Error", html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")


def handle_sentiment_sentence_analysis(button_id, sentence_text, selected_model):
    """Handle sentiment analysis for sentence level."""
    if not sentence_text or not sentence_text.strip():
        return "Input Required", html.Div("Please enter text to analyze.", className="text-muted text-center py-3")

    if button_id == "feature-btn-lime":
        title = "LIME Explanation"
        # Import and create the LIME layout directly
        from pages import sentiment_lime
        content = sentiment_lime.create_layout()
        return title, content
    elif button_id == "feature-btn-attention_entropy":
        title = "Attention Entropy"
        # Import and create the attention entropy layout directly
        from pages import sentiment_attention_entropy
        content = sentiment_attention_entropy.create_layout()
        return title, content
    elif button_id == "feature-btn-token_embeddings":
        title = "Token Embeddings"
        # Import and create the token embeddings layout directly
        from pages import sentiment_token_embeddings
        content = sentiment_token_embeddings.create_layout()
        return title, content
    elif button_id == "feature-btn-logit_matrix":
        # This should open the logit matrix modal, not the regular modal
        # Return a special indicator that will be handled by a separate callback
        title = "Logit Matrix Heatmap"
        content = html.Div([
            html.I(className="fas fa-chart-area me-2"),
            html.P("Opening Logit Matrix Analysis...")
        ], className="text-center py-3")
        return title, content
    elif button_id == "feature-btn-attribution":
        title = "Token Attribution"
        try:
            if not sentence_text.strip():
                return "Input Required", html.Div("Please enter text to analyze.", className="text-muted text-center py-3")
            
            result = model_api.get_token_attributions(sentence_text)
            tokens = result["tokens"]
            attributions = result["attributions"]
            predicted_class = result["predicted_class"]

            # Create bar chart for token attributions
            fig = go.Figure()
            positive_attr = [max(0, attr) for attr in attributions]
            negative_attr = [min(0, attr) for attr in attributions]
            
            fig.add_trace(go.Bar(
                x=tokens,
                y=positive_attr,
                name='Positive contribution',
                marker_color='green',
                text=[f"{attr:.3f}" if attr > 0 else "" for attr in attributions],
                textposition='outside',
            ))
            
            fig.add_trace(go.Bar(
                x=tokens,
                y=negative_attr,
                name='Negative contribution',
                marker_color='red',
                text=[f"{attr:.3f}" if attr < 0 else "" for attr in attributions],
                textposition='outside',
            ))
            
            fig.update_layout(
                title="Token Contributions to Sentiment",
                barmode='relative',
                height=400,
                margin=dict(l=20, r=20, t=60, b=80),
                xaxis=dict(title='Tokens', tickangle=-45),
                yaxis=dict(title='Attribution Score'),
            )

            sentiment_class = "success" if predicted_class.lower() in ["positive", "1"] else "danger"
            sentiment_icon = "✓" if predicted_class.lower() in ["positive", "1"] else "✗"
            
            sentiment_card = dbc.Card([
                dbc.CardHeader("Sentiment Prediction"),
                dbc.CardBody([
                    html.H4([f"{sentiment_icon} ", f"Sentiment: {predicted_class}"], 
                            className=f"text-{sentiment_class}"),
                ]),
            ], className="mb-4 shadow-sm")

            text_tokens = []
            for i, (token, attr) in enumerate(zip(tokens, attributions)):
                if attr > 0.1:
                    text_tokens.append(html.Span(token, className="token-highlight token-highlight-positive", 
                                              title=f"Score: {attr:.3f}"))
                elif attr < -0.1:
                    text_tokens.append(html.Span(token, className="token-highlight token-highlight-negative", 
                                              title=f"Score: {attr:.3f}"))
                else:
                    text_tokens.append(html.Span(token))
                
                if i < len(tokens) - 1:
                    text_tokens.append(" ")
            
            text_card = dbc.Card([
                dbc.CardHeader("Text with Highlighted Tokens"),
                dbc.CardBody([
                    html.P(text_tokens),
                    html.Small([
                        "Green: positive contribution, Red: negative contribution. ",
                        "Hover over highlighted tokens to see attribution scores."
                    ], className="text-muted mt-2")
                ]),
            ], className="mb-4 shadow-sm")

            attribution_card = dbc.Card([
                dbc.CardHeader("Token Attribution Visualization"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    ),
                ]),
            ], className="shadow-sm")

            content = html.Div([
                sentiment_card,
                text_card,
                attribution_card
            ])
            return title, content
        except Exception as e:
            logger.error(f"Error generating token attribution: {str(e)}")
            return "Analysis Error", html.Div([
                html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
                html.H5("Analysis Error", className="text-danger"),
                html.P(f"An error occurred: {str(e)}")
            ], className="text-center py-5")
    
    return f"Feature: {button_id}", html.Div(f"Feature {button_id} not implemented yet.", className="text-muted text-center py-3")

def handle_ner_sentence_analysis(button_id, sentence_text, selected_model):
    """Handle NER analysis for sentence level."""
    if not sentence_text or not sentence_text.strip():
        return "Input Required", html.Div("Please enter text to analyze.", className="text-muted text-center py-3")

    if button_id == "feature-btn-entity_viz":
        title = "Entity Visualization"
        # Use the dedicated NER entity visualization page
        from pages import ner_entity_visualization
        content = ner_entity_visualization.create_layout()
        return title, content
    elif button_id == "feature-btn-attention_entropy":
        title = "NER Attention Entropy"
        # Use the sentiment attention entropy layout with pre-filled text
        content = create_ner_attention_entropy_for_point(sentence_text, selected_model)
        return title, content
    
    return f"Feature: {button_id}", html.Div(f"Feature {button_id} not implemented yet.", className="text-muted text-center py-3")

def handle_ner_model_analysis(button_id, selected_dataset, selected_model):
    """Handle NER analysis for model level."""
    # NER model level is in development
    return "In Development", html.Div([
        html.Div([
            html.I(className="fas fa-tools fa-4x text-muted mb-4"),
            html.H3("In Development", className="text-muted mb-3"),
            html.P("This feature is currently under development and will be available soon.", 
                  className="text-muted text-center", style={"fontSize": "1.2rem"}),
            html.P("Please check back later for NER model-level analysis capabilities.", 
                  className="text-muted text-center")
        ], className="text-center", style={"minHeight": "60vh", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
    ])

# categorize_error_patterns function moved to models/error_analysis.py

@callback(
    Output("error-pattern-content", "children", allow_duplicate=True),
    [Input("analyze-error-patterns-btn", "n_clicks")],
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def analyze_error_patterns(n_clicks, selected_dataset, selected_model):
    """Analyze error patterns in the dataset."""
    if not n_clicks or not selected_dataset or not selected_model:
        return dash.no_update

    try:
        # Load dataset samples
        if selected_dataset == 'IMDb':
            split = 'test'
        else:
            split = 'dev'
        samples = load_dataset_samples(selected_dataset, 'sentiment', split=split, max_samples=200)
        
        if not samples:
            raise ValueError(f"No samples found in dataset {selected_dataset}")
        
        # Run sentiment analysis and collect errors
        errors = []
        for sample in samples:
            text = sample['text']
            true_label = sample['label']
            
            prediction = model_api.analyze_sentiment(text)
            predicted_label = prediction.get("label", "Unknown")
            confidence = prediction.get("score", 0)
            
            if str(predicted_label) != str(true_label):
                errors.append({
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence
                })
        
        # Categorize errors
        error_patterns = categorize_error_patterns(errors)
        
        # Create visualization
        categories = list(error_patterns.keys())
        counts = [error_patterns[cat]["count"] for cat in categories]
        
        pattern_fig = px.bar(
            x=[cat.replace("_", " ").title() for cat in categories],
            y=counts,
            labels={"x": "Error Category", "y": "Count"},
            title="Error Categories Distribution"
        )
        
        # Create detailed breakdown
        pattern_cards = []
        for category, data in error_patterns.items():
            if data["count"] > 0:
                examples = data["examples"][:3]  # Show top 3 examples
                example_list = html.Ul([
                    html.Li([
                        html.Div(f"Text: {ex['text'][:100]}..."),
                        html.Small([
                            f"True: {'Positive' if ex['true_label'] == '1' else 'Negative'}, ",
                            f"Predicted: {'Positive' if ex['predicted_label'] == '1' else 'Negative'}, ",
                            f"Confidence: {ex['confidence']:.2f}"
                        ], className="text-muted")
                    ]) for ex in examples
                ])
                
                card = dbc.Card([
                    dbc.CardHeader(category.replace("_", " ").title()),
                    dbc.CardBody([
                        html.P(data["description"]),
                        html.P(f"Count: {data['count']} errors"),
                        html.H6("Example Cases:", className="mt-3"),
                        example_list
                    ])
                ], className="mb-3")
                pattern_cards.append(card)
        
        return html.Div([
            dcc.Graph(figure=pattern_fig, className="mb-4"),
            html.H4("Detailed Error Pattern Analysis", className="mb-3"),
            html.Div(pattern_cards)
        ])
        
    except Exception as e:
        logger.error(f"Error in error pattern analysis: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger"),
            html.P(f"Error: {str(e)}")
        ])

# Function moved to pages/point_analysis.py

# Callback for grid layer selection in modal popups
@callback(
    Output("grid-heads-content", "children", allow_duplicate=True),
    [Input("grid-layer-dropdown-heads", "value")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def update_heads_grid_content(layer, sentence_text, selected_model):
    """Update heads grid content based on layer selection."""
    if not sentence_text or not selected_model or layer is None:
        return dash.no_update
    
    return create_grid_visualization("heads", sentence_text, selected_model, layer)

@callback(
    Output("grid-matrices-content", "children", allow_duplicate=True),
    [Input("grid-layer-dropdown-matrices", "value")],
    [State("sentence-input", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def update_matrices_grid_content(layer, sentence_text, selected_model):
    """Update matrices grid content based on layer selection."""
    if not sentence_text or not selected_model or layer is None:
        return dash.no_update
    
    return create_grid_visualization("matrices", sentence_text, selected_model, layer)

# Callback to handle close error modal button
@callback(
    Output("error-analysis-modal", "is_open", allow_duplicate=True),
    Input("close-error-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_error_modal(n_clicks):
    """Close the error analysis modal."""
    if n_clicks:
        return False
    return no_update

# Callback to handle close visualization modal button
@callback(
    Output("visualization-modal", "is_open", allow_duplicate=True),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_visualization_modal(n_clicks):
    """Close the visualization modal."""
    if n_clicks:
        return False
    return no_update

# Note: Removed duplicate callback that was causing conflicts
# The scatter plot click handling is now done in handle_scatter_plot_click callback

# Callback to handle counterfactual analysis after point click
@callback(
    Output("counterfactual-tab", "children", allow_duplicate=True),
    [Input("selected-error-store", "data")],
    prevent_initial_call=True
)
def update_counterfactual_analysis(point_data):
    """Generate counterfactual examples for the selected point."""
    if not point_data:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-warning me-2"),
            html.P("No data point selected for analysis.")
        ])
        
    if not isinstance(point_data, dict) or 'text' not in point_data:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P("Invalid data point format.")
        ])
    
    try:
        text = point_data["text"]
        
        # Generate counterfactuals using different strategies
        counterfactuals = []
        
        # 1. Try negation
        if "not" not in text.lower():
            counterfactuals.append({
                "text": f"not {text}",
                "strategy": "Negation"
            })
        
        # 2. Try intensity modification
        intensity_words = ["very", "really", "extremely", "absolutely"]
        for word in intensity_words:
            if word in text.lower():
                modified = text.lower().replace(word, "slightly")
                counterfactuals.append({
                    "text": modified,
                    "strategy": "Intensity Modification"
                })
        
        # 3. Try sentiment word replacement
        sentiment_replacements = {
            "good": "bad",
            "great": "terrible",
            "excellent": "poor",
            "amazing": "awful"
        }
        for orig, repl in sentiment_replacements.items():
            if orig in text.lower():
                modified = text.lower().replace(orig, repl)
                counterfactuals.append({
                    "text": modified,
                    "strategy": "Sentiment Word Replacement"
                })
        
        # Analyze counterfactuals
        results = []
        for cf in counterfactuals:
            try:
                prediction = model_api.analyze_sentiment(cf["text"])
                results.append({
                    "original": text,
                    "counterfactual": cf["text"],
                    "strategy": cf["strategy"],
                    "new_prediction": prediction.get("label", "Unknown"),
                    "confidence": prediction.get("score", 0)
                })
            except Exception as e:
                logger.error(f"Error analyzing counterfactual: {str(e)}")
                continue
        
        # Create results display
        if results:
            rows = []
            for result in results:
                rows.append(
                    html.Tr([
                        html.Td(result["strategy"]),
                        html.Td(result["counterfactual"]),
                        html.Td(f"{'Positive' if result['new_prediction'] in ['1', 1] else 'Negative'}"),
                        html.Td(f"{result['confidence']:.2f}")
                    ])
                )
            
            return html.Div([
                html.H5("Counterfactual Analysis"),
                html.P(f"Original text: '{text}'"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Strategy"),
                                html.Th("Modified Text"),
                                html.Th("New Prediction"),
                                html.Th("Confidence")
                            ])
                        ),
                        html.Tbody(rows)
                    ],
                    bordered=True,
                    hover=True
                )
            ])
        else:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("No suitable counterfactuals generated for this text.")
            ])
            
    except Exception as e:
        logger.exception("Error in counterfactual analysis")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error generating counterfactuals: {str(e)}")
        ])

# Note: Callback moved to pages/similarity_analysis.py

# Callbacks for the enhanced point analysis buttons
@callback(
    Output("point-analysis-results", "children"),
    [Input("point-lime-btn", "n_clicks"),
     Input("point-attention-btn", "n_clicks"),
     Input("point-embeddings-btn", "n_clicks"),
     Input("point-counterfactual-btn", "n_clicks"),
     Input("point-similarity-btn", "n_clicks")],
    [State("selected-error-store", "data"),
     State("dataset-dropdown", "value")],
    prevent_initial_call=True
)
def handle_point_analysis_buttons(lime_clicks, attention_clicks, embeddings_clicks, 
                                counterfactual_clicks, similarity_clicks, point_data, selected_dataset):
    """Handle clicks on the enhanced point analysis buttons."""
    ctx = dash.callback_context
    if not ctx.triggered or not point_data:
        return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    text = point_data.get("text", "")
    
    if not text:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-warning me-2"),
            html.P("No text available for analysis.")
        ])
    
    try:
        if button_id == "point-lime-btn":
            return point_analysis.create_lime_analysis(text)
        elif button_id == "point-attention-btn":
            return point_analysis.create_attention_analysis(text)
        elif button_id == "point-embeddings-btn":
            return point_analysis.create_embeddings_analysis(text)
        elif button_id == "point-counterfactual-btn":
            return create_counterfactual_analysis_enhanced(text)
        elif button_id == "point-similarity-btn":
            return create_similarity_analysis_enhanced(text, selected_dataset)
        
    except Exception as e:
        logger.error(f"Error in point analysis: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error: {str(e)}")
        ])
    
    return dash.no_update

def create_counterfactual_analysis_enhanced(text):
    """Create counterfactual analysis for the selected text."""
    try:
        # Generate counterfactuals using different strategies
        counterfactuals = []
        
        # Strategy 1: Negation flipping
        if " not " in text.lower():
            modified = text.lower().replace(" not ", " ")
            counterfactuals.append({"text": modified, "strategy": "Remove negation"})
        else:
            # Add negation to the first verb or adjective
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ["is", "was", "are", "were", "good", "bad", "great", "terrible"]:
                    modified_words = words[:i] + ["not"] + words[i:]
                    counterfactuals.append({"text": " ".join(modified_words), "strategy": "Add negation"})
                    break
        
        # Strategy 2: Sentiment word replacement
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike", "poor", "worst"]
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in positive_words:
                modified_words = words[:]
                modified_words[i] = "bad"
                counterfactuals.append({"text": " ".join(modified_words), "strategy": "Sentiment flip (pos→neg)"})
                break
            elif word.lower() in negative_words:
                modified_words = words[:]
                modified_words[i] = "good"
                counterfactuals.append({"text": " ".join(modified_words), "strategy": "Sentiment flip (neg→pos)"})
                break
        
        # Analyze counterfactuals
        results = []
        for cf in counterfactuals:
            try:
                prediction = model_api.analyze_sentiment(cf["text"])
                results.append({
                    "original": text,
                    "counterfactual": cf["text"],
                    "strategy": cf["strategy"],
                    "new_prediction": prediction.get("label", "Unknown"),
                    "confidence": prediction.get("score", 0)
                })
            except Exception as e:
                logger.error(f"Error analyzing counterfactual: {str(e)}")
                continue
        
        # Create results display
        if results:
            rows = []
            for result in results:
                rows.append(
                    html.Tr([
                        html.Td(result["strategy"]),
                        html.Td(result["counterfactual"]),
                        html.Td(f"{'Positive' if result['new_prediction'] in ['1', 1] else 'Negative'}"),
                        html.Td(f"{result['confidence']:.2f}")
                    ])
                )
            
            return html.Div([
                html.H5("Counterfactual Testing Results"),
                html.P(f"Original text: '{text}'"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Strategy"),
                                html.Th("Modified Text"),
                                html.Th("New Prediction"),
                                html.Th("Confidence")
                            ])
                        ),
                        html.Tbody(rows)
                    ],
                    bordered=True,
                    hover=True
                )
            ])
        else:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("No suitable counterfactuals generated for this text.")
            ])
            
    except Exception as e:
        logger.exception("Error in counterfactual analysis")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error generating counterfactuals: {str(e)}")
        ])

def create_lime_analysis_enhanced(text):
    """Create LIME analysis display for the selected text."""
    try:
        # Get LIME explanation
        lime_result = model_api.get_lime_explanation(text)
        
        if lime_result and 'feature_importance' in lime_result:
            # Extract words and weights from feature_importance (same as sentence-level analysis)
            feature_importance = lime_result.get("feature_importance", [])
            words = [item[0] for item in feature_importance]
            weights = [item[1] for item in feature_importance]
            
            # Create LIME visualization
            from components.visualizations import create_lime_bar_chart
            fig = create_lime_bar_chart(words, weights)
            
            return html.Div([
                html.H5("LIME Analysis Results"),
                html.P(f"Original text: '{text}'"),
                dcc.Graph(figure=fig, config={'displayModeBar': False}),
                html.P("LIME shows which words contribute most to the model's prediction.", className="text-muted")
            ])
        else:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("LIME analysis not available for this text.")
            ])
            
    except Exception as e:
        logger.exception("Error in LIME analysis")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error in LIME analysis: {str(e)}")
        ])

def create_attention_entropy_analysis_enhanced(text, selected_model):
    """Create attention entropy analysis display for the selected text."""
    try:
        # Get attention entropy data
        entropy_result = model_api.get_attention_entropy(text)
        
        if entropy_result and 'entropy' in entropy_result:
            # Create entropy visualization
            from components.visualizations import create_clickable_entropy_heatmap
            fig = create_clickable_entropy_heatmap(entropy_result['entropy'])
            
            return html.Div([
                html.H5("Attention Entropy Analysis Results"),
                html.P(f"Original text: '{text}'"),
                dcc.Graph(
                    id="entropy-heatmap-graph", 
                    figure=fig, 
                    config={'displayModeBar': False}
                ),
                html.P("Click on any cell to see the detailed attention matrix for that layer and head.", className="text-info"),
                html.Div(id="entropy-attention-matrix", className="mt-3"),
                html.P("Attention entropy shows how focused the model's attention is across layers and heads.", className="text-muted")
            ])
        else:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("Attention entropy analysis not available for this text.")
            ])
            
    except Exception as e:
        logger.exception("Error in attention entropy analysis")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error in attention entropy analysis: {str(e)}")
        ])

def create_token_embeddings_analysis_enhanced(text, selected_model):
    """Create token embeddings analysis display for the selected text."""
    try:
        # Get token embeddings using the visualizer's method
        visualizer = model_api.get_bert_visualizer()
        embeddings_result = visualizer.get_token_embeddings(text)
        
        if embeddings_result and 'embeddings' in embeddings_result:
            # Filter out special tokens that cause issues
            tokens = embeddings_result['tokens']
            embeddings = embeddings_result['embeddings']
            
            # Remove special tokens like [CLS], [SEP], [PAD]
            filtered_tokens = []
            filtered_embeddings = []
            special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
            
            for i, token in enumerate(tokens):
                if token not in special_tokens:
                    filtered_tokens.append(token)
                    filtered_embeddings.append(embeddings[i])
            
            if filtered_tokens and filtered_embeddings:
                # Create embeddings visualization
                from components.visualizations import create_embedding_plot
                fig = create_embedding_plot(filtered_embeddings, filtered_tokens)
                
                return html.Div([
                    html.H5("Token Embeddings Analysis Results"),
                    html.P(f"Original text: '{text}'"),
                    dcc.Graph(figure=fig, config={'displayModeBar': False}),
                    html.P("Token embeddings show how the model represents each word in high-dimensional space.", className="text-muted")
                ])
            else:
                return html.Div([
                    html.I(className="fas fa-info-circle text-info me-2"),
                    html.P("No valid tokens found for embedding visualization.")
                ])
        else:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("Token embeddings analysis not available for this text.")
            ])
            
    except Exception as e:
        logger.exception("Error in token embeddings analysis")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error in token embeddings analysis: {str(e)}")
        ])

def create_similarity_analysis_enhanced(text, selected_dataset):
    """Create similarity analysis display for the selected text."""
    try:
        # Use the similarity analysis function from the new module
        similar_examples = find_similar_examples(text, selected_dataset)
        
        if similar_examples:
            rows = []
            for i, example in enumerate(similar_examples, 1):
                rows.append(
                    html.Tr([
                        html.Td(i),
                        html.Td(example["text"]),
                        html.Td(f"{example['similarity']:.2f}")
                    ])
                )
            
            return html.Div([
                html.H5("Similarity Analysis Results"),
                html.P(f"Original text: '{text}'"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("#"),
                                html.Th("Similar Text"),
                                html.Th("Similarity Score")
                            ])
                        ),
                        html.Tbody(rows)
                    ],
                    bordered=True,
                    hover=True
                )
            ])
            
        return html.Div([
            html.I(className="fas fa-info-circle text-info me-2"),
            html.P("No similar examples found in the dataset."),
            html.Small([
                "Try adjusting the similarity threshold or analyzing a different example.",
                html.Br(),
                f"Current text: '{text[:100]}...'"
            ], className="text-muted")
        ])
            
    except Exception as e:
        logger.exception("Error in similarity analysis")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.H6("Error in Similarity Analysis", className="mb-2"),
            html.P(f"An error occurred: {str(e)}"),
            html.Small(
                "Please try again with a different example or contact support if the issue persists.",
                className="text-muted"
            )
        ])

# Callback to handle Error Pattern Analysis feature button
@callback(
    Output("error-analysis-modal", "is_open", allow_duplicate=True),
    [Input("feature-btn-error_patterns", "n_clicks")],
    [State("error-analysis-modal", "is_open")],
    prevent_initial_call=True
)
def open_error_pattern_modal(n_clicks, is_open):
    """Open Error Pattern Analysis modal when feature button is clicked."""
    if n_clicks:
        return True
    return is_open

# Callback to populate error pattern content
@callback(
    Output("error-pattern-content", "children"),
    [Input("error-analysis-modal", "is_open")],
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def populate_error_pattern_content(modal_is_open, selected_dataset, selected_model):
    """Populate the error pattern analysis content when modal opens."""
    if not modal_is_open or not selected_dataset or not selected_model:
        return html.Div()
    
    try:
        # Get stored analysis results
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, selected_model["model_path"])
        
        if not stored_analysis:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.H5("No Analysis Data Available"),
                html.P("Please run 'Analyze Dataset' first to see error patterns.")
            ], className="text-center py-5")
        
        # Get the results from the stored analysis
        analysis_results = stored_analysis.get("results", {})
        high_conf_errors = analysis_results.get("high_confidence_errors", [])
        dataset_name = selected_dataset
        
        # Create the error pattern analysis content
        content = create_error_pattern_analysis_content(high_conf_errors, dataset_name)
        
        return content
        
    except Exception as e:
        logger.error(f"Error creating error pattern content: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.H5("Error Loading Analysis"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")

# REMOVED: Duplicate callback - now handled by handle_point_analysis_combined above
    """Handle clicking on scatter plot points and show analysis buttons."""

# Callback to handle entropy heatmap clicks and show attention matrix
@callback(
    Output("entropy-attention-matrix", "children"),
    [Input("entropy-heatmap-graph", "clickData")],
    [State("error-scatter-plot", "clickData"),
     State("selected-model-store", "data"),
     State("dataset-dropdown", "value")],
    prevent_initial_call=True
)
def show_attention_matrix_from_entropy(entropy_click_data, scatter_click_data, selected_model, selected_dataset):
    """Show attention matrix when entropy heatmap cell is clicked."""
    if not entropy_click_data or not scatter_click_data or not selected_model:
        return html.Div()
    
    try:
        # Get the clicked layer and head from entropy heatmap
        layer_idx = entropy_click_data["points"][0]["y"]
        head_idx = entropy_click_data["points"][0]["x"]
        
        # Get the original text from scatter plot data
        point_index = scatter_click_data["points"][0]["pointIndex"]
        stored_analysis = analysis_store.get_dataset_analysis(
            selected_dataset,  # Use the actual selected dataset
            selected_model["model_path"]
        )
        
        if stored_analysis:
            results = stored_analysis["results"]["results"]
            selected_point = results[point_index]
            text = selected_point["text"]
            
            # Get attention weights for this specific layer and head
            attention_data = model_api.get_attention_weights(text, layer_idx, head_idx)
            
            if attention_data and 'attention_matrix' in attention_data:
                # Create attention matrix visualization
                from components.visualizations import create_attention_heatmap_matrix
                tokens = attention_data['tokens']
                attention_matrix = attention_data['attention_matrix']
                
                fig = create_attention_heatmap_matrix(tokens, attention_matrix, layer_idx, head_idx)
                
                return html.Div([
                    html.H6(f"Attention Matrix - Layer {layer_idx}, Head {head_idx}"),
                    dcc.Graph(figure=fig, config={'displayModeBar': False}),
                    html.P(f"Text: {text}", className="text-muted mt-2")
                ])
            
        return html.Div("Could not load attention matrix.", className="text-muted")
        
    except Exception as e:
        logger.error(f"Error showing attention matrix: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

# CALLBACK 1: Handle display mode toggle and scatter plot clicks
@callback(
    [Output("error-scatter-plot", "figure", allow_duplicate=True),
     Output("point-analysis-buttons", "children", allow_duplicate=True),
     Output("point-analysis-buttons", "style", allow_duplicate=True),
     Output("selected-point-details", "children", allow_duplicate=True)],
    [Input("error-display-toggle", "value"),
     Input("error-scatter-plot", "clickData")],
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def handle_error_analysis_display_and_clicks(display_mode, click_data, selected_dataset, selected_model):
    """Handle display mode toggle and scatter plot clicks."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, [], {"display": "none"}, html.Div()
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Load analysis data from backup file
    import json
    import os
    backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
    
    if not os.path.exists(backup_file_path):
        return dash.no_update, [], {"display": "none"}, html.Div("Analysis data not found. Please run analysis first.", className="text-warning")
    
    try:
        with open(backup_file_path, "r") as f:
            backup_data = json.load(f)
        
        results = backup_data.get("results", {}).get("results", [])
        total_samples = backup_data.get("results", {}).get("total_samples", 0)
        
        # HANDLE DISPLAY MODE TOGGLE
        if triggered_id == "error-display-toggle":
            if not results:
                return dash.no_update, [], {"display": "none"}, html.Div()
            
            # Filter results based on display mode
            if display_mode == "wrong":
                filtered_results = [r for r in results if not r.get("correct", True)]
                title_suffix = f"Wrong Predictions ({len(filtered_results)} errors out of {total_samples} samples)"
            else:  # display_mode == "all"
                filtered_results = results
                title_suffix = f"All Predictions ({len(filtered_results)} samples)"
            
            if not filtered_results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data to display", xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16, color="gray")
                )
                fig.update_layout(title=f"Error Analysis - {title_suffix}", height=450, showlegend=False)
                return fig, [], {"display": "none"}, html.Div()
            
            # Create scatter plot
            df = pd.DataFrame(filtered_results)
            df['hover_text'] = df.apply(lambda row: f"Text: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}<br>True: {'Positive' if str(row['true_label']) == '1' else 'Negative'}<br>Predicted: {'Positive' if str(row['predicted_label']) == '1' else 'Negative'}<br>Confidence: {row['confidence']:.2f}", axis=1)
            
            fig = px.scatter(
                df, x=df.index, y="confidence", color="correct", hover_data={"hover_text": True},
                labels={"x": "Sample Index", "confidence": "Prediction Confidence", "correct": "Correct"},
                title=f"Error Analysis - {title_suffix}", color_discrete_map={True: "blue", False: "red"}
            )
            fig.update_traces(hovertemplate="<b>Sample %{x}</b><br>Confidence: %{y:.2f}<br>%{customdata[0]}<extra></extra>", customdata=df[['hover_text']].values)
            fig.update_layout(height=450, showlegend=True, legend=dict(title="Prediction", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            
            return fig, [], {"display": "none"}, html.Div()
        
        # HANDLE SCATTER PLOT CLICK
        elif triggered_id == "error-scatter-plot" and click_data:
            if not selected_dataset or not selected_model:
                return dash.no_update, [], {"display": "none"}, html.Div()
            
            point_index = click_data["points"][0]["pointIndex"]
            if point_index >= len(results):
                return dash.no_update, [], {"display": "none"}, html.Div("Invalid point selected.", className="text-danger")
            
            selected_point = results[point_index]
            text = selected_point["text"]
            
            # Create analysis buttons
            buttons = html.Div([
                html.H5("Detailed Analysis Options:", className="mb-3"),
                html.P("Click on any analysis type to explore this data point in detail:", className="text-muted mb-3"),
                dbc.Row([
                    dbc.Col([dbc.Button([html.I(className="fas fa-lightbulb me-2"), "LIME Explanation"], id="point-lime-btn", color="primary", size="sm", className="w-100 mb-2")], width=6),
                    dbc.Col([dbc.Button([html.I(className="fas fa-brain me-2"), "Attention Entropy"], id="point-attention-btn", color="info", size="sm", className="w-100 mb-2")], width=6),
                ]),
                dbc.Row([
                    dbc.Col([dbc.Button([html.I(className="fas fa-project-diagram me-2"), "Token Embeddings"], id="point-embeddings-btn", color="success", size="sm", className="w-100 mb-2")], width=6),
                    dbc.Col([dbc.Button([html.I(className="fas fa-flask me-2"), "Test Counterfactuals"], id="point-counterfactual-btn", color="warning", size="sm", className="w-100 mb-2")], width=6),
                ])
            ])
            
            # Show selected point info
            point_info = html.Div([
                html.H5("Selected Point Analysis", className="mb-3"),
                dbc.Card([dbc.CardBody([
                    html.P([html.Strong("Text: "), text[:200] + "..." if len(text) > 200 else text], className="mb-2"),
                    html.P([html.Strong("True Label: "), "Positive" if selected_point["true_label"] == "1" else "Negative"], className="mb-2"),
                    html.P([html.Strong("Predicted Label: "), "Positive" if selected_point["predicted_label"] == "1" else "Negative"], className="mb-2"),
                    html.P([html.Strong("Confidence: "), f"{selected_point['confidence']:.2%}"], className="mb-2"),
                    html.P([html.Strong("Correct: "), "Yes" if selected_point["correct"] else "No"], className="mb-0")
                ])], className="mb-3")
            ])
            
            return dash.no_update, buttons, {"display": "block"}, point_info
        
        return dash.no_update, [], {"display": "none"}, html.Div()
        
    except Exception as e:
        logger.error(f"Error in error analysis interactions: {str(e)}")
        return dash.no_update, [], {"display": "none"}, html.Div(f"Error: {str(e)}", className="text-danger")

# CALLBACK 2: Handle analysis button clicks using pattern-matching
@callback(
    [Output("analysis-loading-container", "children", allow_duplicate=True),
     Output("analysis-results-container", "children", allow_duplicate=True)],
    [Input("point-lime-btn", "n_clicks"),
     Input("point-attention-btn", "n_clicks"),
     Input("point-embeddings-btn", "n_clicks"),
     Input("point-counterfactual-btn", "n_clicks")],
    [State("error-scatter-plot", "clickData"),
     State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def handle_analysis_button_clicks(lime_clicks, attention_clicks, embeddings_clicks, counterfactual_clicks, 
                                click_data, selected_dataset, selected_model):
    """Handle clicks on the detailed analysis buttons."""
    ctx = dash.callback_context
    if not ctx.triggered or not click_data:
        return html.Div(), html.Div()
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Show loading animation first
    loading_content = html.Div([
        dbc.Spinner(size="lg", color="primary"),
        html.H5("Loading analysis...", className="mt-3 text-center"),
        html.P("Please wait while we analyze this data point.", className="text-muted text-center")
    ], className="text-center py-4")
    
    try:
        # Get the selected point data
        point_index = click_data["points"][0]["pointIndex"]
        
        # Load analysis data from backup file
        import json
        import os
        backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
        
        if not os.path.exists(backup_file_path):
            return html.Div(), html.Div("Analysis data not found. Please run analysis first.", className="text-warning")
        
        with open(backup_file_path, "r") as f:
            backup_data = json.load(f)
        
        results = backup_data.get("results", {}).get("results", [])
        if point_index >= len(results):
            return html.Div(), html.Div("Invalid point selected.", className="text-danger")
        
        selected_point = results[point_index]
        text = selected_point["text"]
        
        # Create analysis content based on button clicked
        if triggered_id == "point-lime-btn":
            analysis_title = "LIME Explanation"
            content = create_lime_analysis_for_point(text, selected_model)
        elif triggered_id == "point-attention-btn":
            analysis_title = "Attention Entropy Analysis"
            content = create_attention_analysis_for_point(text, selected_model)
        elif triggered_id == "point-embeddings-btn":
            analysis_title = "Token Embeddings Analysis"
            content = create_embeddings_analysis_for_point(text, selected_model)
        elif triggered_id == "point-counterfactual-btn":
            analysis_title = "Counterfactual Analysis"
            content = create_counterfactual_analysis_for_point(text, selected_model)
        else:
            return html.Div(), html.Div()
        
        # Create final analysis results
        analysis_results = html.Div([
            html.H5(analysis_title, className="mb-3 text-primary"),
            html.Hr(),
            content
        ], className="analysis-results-card p-3 border rounded")
        
        return html.Div(), analysis_results
        
    except Exception as e:
        logger.error(f"Error in analysis button click: {str(e)}")
        return html.Div(), html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.Span(f"Error: {str(e)}", className="text-danger")
        ])

def create_lime_analysis_for_point(text, selected_model):
    """Create LIME analysis for a specific point with pre-filled text."""
    try:
        # Create a modified LIME layout with the selected text pre-filled
        from pages import sentiment_lime
        layout = sentiment_lime.create_layout()
        
        # Find and replace the textarea with pre-filled text
        def update_textarea(component):
            if hasattr(component, 'id') and component.id == 'lime-input-text':
                component.value = text
                # Also update the label
                return component
            elif hasattr(component, 'children') and isinstance(component.children, list):
                for i, child in enumerate(component.children):
                    if hasattr(child, 'children') and isinstance(child.children, str):
                        if "Enter text to analyze:" in child.children:
                            child.children = f"Text: {text[:100]}{'...' if len(text) > 100 else text}"
                    else:
                        update_textarea(child)
            return component
        
        update_textarea(layout)
        return layout
    except Exception as e:
        return html.Div(f"Error creating LIME analysis: {str(e)}", className="text-danger")

def create_attention_analysis_for_point(text, selected_model):
    """Create attention entropy analysis for a specific point with pre-filled text."""
    try:
        # Create a modified attention entropy layout with the selected text pre-filled
        from pages import sentiment_attention_entropy
        layout = sentiment_attention_entropy.create_layout()
        
        # Find and replace the textarea with pre-filled text
        def update_textarea(component):
            if hasattr(component, 'id') and component.id == 'entropy-input':
                component.value = text
                return component
            elif hasattr(component, 'children') and isinstance(component.children, list):
                for i, child in enumerate(component.children):
                    if hasattr(child, 'children') and isinstance(child.children, str):
                        if "Enter text to analyze:" in child.children:
                            child.children = f"Text: {text[:100]}{'...' if len(text) > 100 else text}"
                    else:
                        update_textarea(child)
            return component
        
        update_textarea(layout)
        return layout
    except Exception as e:
        return html.Div(f"Error creating attention analysis: {str(e)}", className="text-danger")

def create_embeddings_analysis_for_point(text, selected_model):
    """Create token embeddings analysis for a specific point with pre-filled text."""
    try:
        # Create a modified token embeddings layout with the selected text pre-filled
        from pages import sentiment_token_embeddings
        layout = sentiment_token_embeddings.create_layout()
        
        # Find and replace the textarea with pre-filled text
        def update_textarea(component):
            if hasattr(component, 'id') and component.id == 'token-embed-input-text':
                component.value = text
                return component
            elif hasattr(component, 'children') and isinstance(component.children, list):
                for i, child in enumerate(component.children):
                    if hasattr(child, 'children') and isinstance(child.children, str):
                        if "Enter text to analyze:" in child.children:
                            child.children = f"Text: {text[:100]}{'...' if len(text) > 100 else text}"
                    else:
                        update_textarea(child)
            return component
        
        update_textarea(layout)
        return layout
    except Exception as e:
        return html.Div(f"Error creating embeddings analysis: {str(e)}", className="text-danger")

def create_counterfactual_analysis_for_point(text, selected_model):
    """Create counterfactual analysis for a specific point."""
    try:
        from models.counterfactual_analysis import analyze_counterfactuals
        
        # Get original prediction first
        prediction_result = model_api.get_sentiment(text)
        if not prediction_result:
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Could not get prediction for the text."
                ], color="warning")
            ])
        
        original_prediction = prediction_result.get('label', 'Unknown')
        confidence = prediction_result.get('score', 0.0)
        
        # Run counterfactual analysis
        counterfactual_results = analyze_counterfactuals(text, original_prediction, confidence)
        
        if 'error' in counterfactual_results:
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Error in counterfactual analysis: {counterfactual_results['error']}"
                ], color="danger")
            ])
        
        counterfactuals = counterfactual_results.get('counterfactuals', [])
        statistics = counterfactual_results.get('statistics', {})
        
        # Create visualization
        content = html.Div([
            # Header
            html.Div([
                html.H5([
                    html.I(className="fas fa-flask me-2"),
                    "Counterfactual Analysis Results"
                ], className="mb-3"),
                html.P(f"Original text: {text[:150]}{'...' if len(text) > 150 else text}", className="text-muted mb-3")
            ]),
            
            # Statistics summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(statistics.get('total_generated', 0), className="text-primary mb-1"),
                            html.P("Generated", className="mb-0 text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(statistics.get('successful_flips', 0), className="text-success mb-1"),
                            html.P("Successful Flips", className="mb-0 text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{statistics.get('flip_rate', 0):.1f}%", className="text-warning mb-1"),
                            html.P("Flip Rate", className="mb-0 text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(statistics.get('best_strategy', 'None'), className="text-info mb-1"),
                            html.P("Best Strategy", className="mb-0 text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Counterfactual examples
            html.H6("Generated Counterfactuals:", className="mb-3"),
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.P([
                            html.Strong("Strategy: "), 
                            html.Span(cf.get('strategy', 'Unknown'), className="badge bg-secondary me-2"),
                            html.Strong("Change: "), 
                            html.Span(cf.get('change', 'N/A'))
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Text: "), 
                            cf.get('text', '')
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Prediction: "),
                            html.Span(
                                cf.get('new_prediction', 'Unknown'),
                                className=f"badge bg-{'success' if cf.get('flipped', False) else 'secondary'} me-2"
                            ),
                            html.Strong("Confidence: "),
                            html.Span(f"{cf.get('new_confidence', 0):.2f}", className="badge bg-info me-2"),
                            html.Strong("Flipped: "),
                            html.Span(
                                "Yes" if cf.get('flipped', False) else "No",
                                className=f"badge bg-{'success' if cf.get('flipped', False) else 'danger'}"
                            )
                        ], className="mb-0")
                    ])
                ], className="mb-2")
                for cf in counterfactuals[:10]  # Show top 10
            ]) if counterfactuals else html.Div([
                dbc.Alert("No counterfactuals were generated.", color="info")
            ])
        ])
        
        return content
        
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Error creating counterfactual analysis: {str(e)}"
            ], color="danger")
        ])

def create_logit_matrix_content(text, selected_model, task_type):
    """Create logit matrix analysis content for the regular modal."""
    try:
        if not text or not text.strip():
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    "Please enter some text to analyze logit patterns."
                ], color="info")
            ])
        
        if not selected_model:
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "No model selected. Please select a model first."
                ], color="warning")
            ])
        
        # Set the model for analysis
        model_api.set_selected_model(selected_model["model_path"], selected_model["model_type"])
        
        # Get logit matrix data
        logit_data = model_api.get_logit_matrix(text.strip())
        
        if not logit_data:
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Failed to extract logit matrix. Please try again."
                ], color="danger")
            ])
        
        # Import visualization functions
        from components.visualizations import create_logit_heatmap, create_logit_comparison_chart
        from pages.logit_matrix import create_logit_analysis_content
        
        # Create the analysis content
        return create_logit_analysis_content(logit_data, task_type)
        
    except Exception as e:
        logger.error(f"Error creating logit matrix content: {str(e)}")
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Error analyzing logits: {str(e)}"
            ], color="danger")
        ])

def create_proper_ner_entity_visualization(text, selected_model):
    """Create proper NER entity visualization using the correct components function."""
    try:
        # Get NER predictions using the API
        ner_result = model_api.get_ner_prediction(text)
        
        if not ner_result or 'entities' not in ner_result:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("No entities found in the text.")
            ], className="text-center py-4")
        
        entities = ner_result['entities']
        
        if not entities:
            return html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.P("No entities detected in the provided text.")
            ], className="text-center py-4")
        
        # Map the entity structure to what the visualization function expects
        mapped_entities = []
        for entity in entities:
            mapped_entities.append({
                "text": entity.get("word", ""),           # NER returns "word", viz expects "text"
                "label": entity.get("label", ""),         # Both use "label" 
                "start_idx": entity.get("start", 0),      # NER returns "start", viz expects "start_idx"
                "end_idx": entity.get("end", 0),          # NER returns "end", viz expects "end_idx"
                "score": entity.get("confidence", 0.0)    # NER returns "confidence", viz expects "score"
            })
        
        # Use the proper visualization function from components
        from components.visualizations import create_entity_visualization
        return create_entity_visualization(mapped_entities, text)
        
    except Exception as e:
        logger.error(f"Error in proper NER entity visualization: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error creating entity visualization: {str(e)}")
        ], className="text-center py-4")

def create_ner_attention_entropy_for_point(text, selected_model):
    """Create NER attention entropy analysis for a specific point with pre-filled text."""
    try:
        # Create a modified attention entropy layout with the selected text pre-filled
        from pages import sentiment_attention_entropy
        layout = sentiment_attention_entropy.create_layout()
        
        # Find and replace the textarea with pre-filled text
        def update_textarea(component):
            if hasattr(component, 'id') and component.id == 'entropy-input':
                component.value = text
                return component
            elif hasattr(component, 'children') and isinstance(component.children, list):
                for i, child in enumerate(component.children):
                    if hasattr(child, 'children') and isinstance(child.children, str):
                        if "Enter text to analyze:" in child.children:
                            child.children = f"Text: {text[:100]}{'...' if len(text) > 100 else text}"
                    else:
                        update_textarea(child)
            return component
        
        update_textarea(layout)
        return layout
    except Exception as e:
        return html.Div(f"Error creating NER attention analysis: {str(e)}", className="text-danger")

# Callback to handle NER detailed results modal
@callback(
    Output("ner-detailed-modal", "is_open"),
    [Input("ner-detailed-results-btn", "n_clicks"),
     Input("close-ner-detailed-modal", "n_clicks")],
    [State("ner-detailed-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_ner_detailed_modal(open_clicks, close_clicks, is_open):
    """Toggle the NER detailed results modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "ner-detailed-results-btn":
        return True
    elif button_id == "close-ner-detailed-modal":
        return False
    
    return is_open

# Note: NER feature buttons are now handled in the main feature callback (handle_feature_analysis)

def create_ner_entity_visualization(text, selected_model):
    """Create NER entity visualization content with pie chart and compact design."""
    try:
        # Get NER predictions
        ner_result = model_api.get_ner_prediction(text)
        
        if not ner_result or 'entities' not in ner_result:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle", style={"fontSize": "2rem", "color": "#17a2b8", "marginBottom": "0.8rem"}),
                    html.H5("No Entities Found", style={"color": "#17a2b8", "marginBottom": "0.8rem"}),
                    html.P("No named entities were detected in the provided text.", style={"fontSize": "0.9rem", "color": "#6c757d"})
                ], style={
                    "textAlign": "center", 
                    "padding": "2rem 1.5rem",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "border": "1px solid #bee5eb"
                })
            ])
        
        entities = ner_result['entities']
        
        # Create highlighted text
        highlighted_text = create_highlighted_text(text, entities)
        
        # Create entity statistics
        entity_stats = {}
        for entity in entities:
            # Handle different possible entity key names
            entity_type = entity.get('entity') or entity.get('label') or entity.get('type') or 'UNKNOWN'
            if entity_type not in entity_stats:
                entity_stats[entity_type] = []
            entity_stats[entity_type].append(entity)
        
        # Create pie chart data
        import plotly.express as px
        import pandas as pd
        
        pie_data = pd.DataFrame([
            {"Entity Type": entity_type, "Count": len(entity_list)}
            for entity_type, entity_list in entity_stats.items()
        ])
        
        colors = {"PER": "#FFE6E6", "ORG": "#E6F3FF", "LOC": "#E6FFE6", "MISC": "#FFF0E6"}
        fig = px.pie(pie_data, values="Count", names="Entity Type", 
                     color="Entity Type", color_discrete_map=colors,
                     title="Entity Distribution")
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=11),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        fig.update_traces(textinfo='label+percent', textfont_size=10)
        
        # Create visualization content
        content = html.Div([
            # Header section - compact
            html.Div([
                html.H4([
                    html.I(className="fas fa-tags", style={"marginRight": "8px", "color": "#007bff"}),
                    "Entity Visualization"
                ], style={"color": "#2c3e50", "marginBottom": "0.5rem", "fontSize": "1.3rem"}),
                
                html.P(f"Analysis of {len(entities)} entities using {selected_model.get('display_name', 'Unknown')} model.", 
                       style={"fontSize": "0.9rem", "color": "#6c757d", "marginBottom": "1rem"})
            ]),
            
            # Highlighted text section - compact
            html.Div([
                html.H5("Highlighted Text", style={"color": "#495057", "marginBottom": "0.5rem", "fontSize": "1.1rem"}),
                html.Div([
                    html.Div(highlighted_text, style={"padding": "1rem", "lineHeight": "1.6", "fontSize": "0.95rem"})
                ], style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "6px",
                    "border": "1px solid #dee2e6",
                    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
                    "marginBottom": "1rem"
                })
            ]),
            
            # Entity Highlighted Color - compact and inline
            html.Div([
                html.H6("Entity Highlighted Color:", style={"color": "#495057", "marginBottom": "0.5rem", "fontSize": "1rem"}),
                html.Div([
                    html.Span([
                        html.Span("PER", style={
                            "backgroundColor": "#FFE6E6", 
                            "padding": "3px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #FFE6E6",
                            "marginRight": "4px",
                            "fontSize": "0.75rem",
                            "fontWeight": "500"
                        }),
                        html.Span("Person", style={"fontSize": "0.8rem", "color": "#495057", "marginRight": "12px"})
                    ]),
                    
                    html.Span([
                        html.Span("ORG", style={
                            "backgroundColor": "#E6F3FF", 
                            "padding": "3px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #E6F3FF",
                            "marginRight": "4px",
                            "fontSize": "0.75rem",
                            "fontWeight": "500"
                        }),
                        html.Span("Organization", style={"fontSize": "0.8rem", "color": "#495057", "marginRight": "12px"})
                    ]),
                    
                    html.Span([
                        html.Span("LOC", style={
                            "backgroundColor": "#E6FFE6", 
                            "padding": "3px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #E6FFE6",
                            "marginRight": "4px",
                            "fontSize": "0.75rem",
                            "fontWeight": "500"
                        }),
                        html.Span("Location", style={"fontSize": "0.8rem", "color": "#495057", "marginRight": "12px"})
                    ]),
                    
                    html.Span([
                        html.Span("MISC", style={
                            "backgroundColor": "#FFF0E6", 
                            "padding": "3px 6px", 
                            "borderRadius": "3px", 
                            "border": "1px solid #FFF0E6",
                            "marginRight": "4px",
                            "fontSize": "0.75rem",
                            "fontWeight": "500"
                        }),
                        html.Span("Miscellaneous", style={"fontSize": "0.8rem", "color": "#495057"})
                    ])
                ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"})
            ], style={
                "backgroundColor": "#f8f9fa",
                "padding": "0.8rem",
                "borderRadius": "6px",
                "border": "1px solid #dee2e6",
                "marginBottom": "1rem"
            }),
            
            # Pie chart
            html.Div([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ], style={"marginBottom": "1rem"}),
            
            # Entity statistics section - compact
            html.Div([
                html.H5("Entity Statistics", style={"color": "#495057", "marginBottom": "0.8rem", "fontSize": "1.1rem"}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H6(entity_type, style={"color": "#2c3e50", "marginBottom": "0.3rem", "fontSize": "1rem"}),
                            html.H4(len(entity_list), style={"color": "#007bff", "marginBottom": "0.3rem", "fontSize": "1.5rem"}),
                            html.P("entities", style={"color": "#6c757d", "marginBottom": "0.8rem", "fontSize": "0.8rem"}),
                            html.Div([
                                html.P(f"• {entity['word']} ({entity['score']:.2f})", 
                                       style={"marginBottom": "0.2rem", "fontSize": "0.8rem"})
                                for entity in entity_list[:2]  # Show top 2
                            ] + ([html.P(f"... +{len(entity_list) - 2} more", 
                                        style={"fontStyle": "italic", "color": "#6c757d", "fontSize": "0.75rem"})]
                                if len(entity_list) > 2 else []))
                        ], style={
                            "backgroundColor": "#ffffff",
                            "padding": "1rem",
                            "borderRadius": "6px",
                            "border": "1px solid #dee2e6",
                            "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.05)",
                            "textAlign": "center",
                            "height": "100%"
                        })
                    ], style={"marginBottom": "0.8rem"})
                    for entity_type, entity_list in entity_stats.items()
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "0.8rem"})
            ])
        ], style={"padding": "0.8rem"})
        
        return content
        
    except Exception as e:
        logger.error(f"Error in NER entity visualization: {str(e)}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"fontSize": "3rem", "color": "#dc3545", "marginBottom": "1rem"}),
                html.H4("Visualization Error", style={"color": "#dc3545", "marginBottom": "1rem"}),
                html.P(f"Error creating entity visualization: {str(e)}", style={"fontSize": "1.1rem", "color": "#6c757d"})
            ], style={
                "textAlign": "center", 
                "padding": "3rem 2rem",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "10px",
                "border": "2px solid #f5c6cb"
            })
        ])

def create_ner_attention_entropy(text, selected_model):
    """Create NER attention entropy visualization like sentiment analysis style."""
    try:
        # Get attention entropy data for NER
        entropy_result = model_api.get_attention_entropy(text)
        
        if not entropy_result or 'entropy' not in entropy_result:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle", style={"fontSize": "2rem", "color": "#17a2b8", "marginBottom": "0.8rem"}),
                    html.H5("Attention Data Unavailable", style={"color": "#17a2b8", "marginBottom": "0.8rem"}),
                    html.P("Attention entropy could not be calculated for this model.", style={"fontSize": "0.9rem", "color": "#6c757d"})
                ], style={
                    "textAlign": "center", 
                    "padding": "2rem 1.5rem",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "border": "1px solid #bee5eb"
                })
            ])
        
        # Create entropy heatmap visualization
        from components.visualizations import create_clickable_entropy_heatmap
        entropy_matrix = entropy_result['entropy']
        fig = create_clickable_entropy_heatmap(entropy_matrix)
        
        # Create compact visualization content
        content = html.Div([
            # Header section - compact
            html.Div([
                html.H4([
                    html.I(className="fas fa-brain", style={"marginRight": "8px", "color": "#28a745"}),
                    "NER Attention Entropy"
                ], style={"color": "#2c3e50", "marginBottom": "0.5rem", "fontSize": "1.3rem"}),
                
                html.P(f"Attention entropy analysis using {selected_model.get('display_name', 'Unknown')} model.", 
                       style={"fontSize": "0.9rem", "color": "#6c757d", "marginBottom": "1rem"})
            ]),
            
            # Entropy heatmap
            html.Div([
                html.H5("Entropy Heatmap", style={"color": "#495057", "marginBottom": "0.5rem", "fontSize": "1.1rem"}),
                html.Div([
                    dcc.Graph(
                        id="ner-entropy-heatmap-graph", 
                        figure=fig, 
                        config={'displayModeBar': False},
                        style={"height": "400px"}
                    )
                ], style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "6px",
                    "border": "1px solid #dee2e6",
                    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
                    "padding": "1rem",
                    "marginBottom": "1rem"
                }),
                
                html.P("Click on any cell to see the detailed attention matrix for that layer and head.", 
                       style={"fontSize": "0.85rem", "color": "#17a2b8", "fontStyle": "italic", "marginBottom": "1rem"})
            ]),
            
            # Attention matrix display area
            html.Div(id="ner-entropy-attention-matrix", className="mt-3", style={"marginBottom": "1rem"}),
            
            # Statistics section - compact
            html.Div([
                html.H5("Entropy Statistics", style={"color": "#495057", "marginBottom": "0.8rem", "fontSize": "1.1rem"}),
                html.Div([
                    html.Div([
                        html.H6("Layers", style={"color": "#2c3e50", "marginBottom": "0.3rem", "fontSize": "1rem"}),
                        html.H4(entropy_matrix.shape[0], style={"color": "#28a745", "marginBottom": "0.3rem", "fontSize": "1.5rem"}),
                        html.P("transformer layers", style={"color": "#6c757d", "fontSize": "0.8rem"})
                    ], style={
                        "backgroundColor": "#ffffff",
                        "padding": "1rem",
                        "borderRadius": "6px",
                        "border": "1px solid #dee2e6",
                        "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.05)",
                        "textAlign": "center"
                    }),
                    
                    html.Div([
                        html.H6("Heads", style={"color": "#2c3e50", "marginBottom": "0.3rem", "fontSize": "1rem"}),
                        html.H4(entropy_matrix.shape[1], style={"color": "#28a745", "marginBottom": "0.3rem", "fontSize": "1.5rem"}),
                        html.P("attention heads", style={"color": "#6c757d", "fontSize": "0.8rem"})
                    ], style={
                        "backgroundColor": "#ffffff",
                        "padding": "1rem",
                        "borderRadius": "6px",
                        "border": "1px solid #dee2e6",
                        "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.05)",
                        "textAlign": "center"
                    }),
                    
                    html.Div([
                        html.H6("Avg Entropy", style={"color": "#2c3e50", "marginBottom": "0.3rem", "fontSize": "1rem"}),
                        html.H4(f"{entropy_matrix.mean():.3f}", style={"color": "#28a745", "marginBottom": "0.3rem", "fontSize": "1.5rem"}),
                        html.P("across all heads", style={"color": "#6c757d", "fontSize": "0.8rem"})
                    ], style={
                        "backgroundColor": "#ffffff",
                        "padding": "1rem",
                        "borderRadius": "6px",
                        "border": "1px solid #dee2e6",
                        "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.05)",
                        "textAlign": "center"
                    })
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))", "gap": "0.8rem", "marginBottom": "1rem"})
            ]),
            
            # Information section - compact
            html.Div([
                html.H6("Understanding Attention Entropy", style={"color": "#495057", "marginBottom": "0.5rem", "fontSize": "1rem"}),
                html.Div([
                    html.P("• Higher entropy = more diffuse attention (model looks at many tokens)", style={"marginBottom": "0.3rem", "fontSize": "0.85rem"}),
                    html.P("• Lower entropy = more focused attention (model focuses on specific tokens)", style={"marginBottom": "0.3rem", "fontSize": "0.85rem"}),
                    html.P("• Different layers and heads show different attention patterns for entity recognition", style={"marginBottom": "0", "fontSize": "0.85rem"})
                ], style={
                    "backgroundColor": "#e7f3ff",
                    "padding": "1rem",
                    "borderRadius": "6px",
                    "border": "1px solid #b3d9ff"
                })
            ])
        ], style={"padding": "0.8rem"})
        
        return content
        
    except Exception as e:
        logger.error(f"Error in NER attention entropy: {str(e)}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"fontSize": "3rem", "color": "#dc3545", "marginBottom": "1rem"}),
                html.H4("Analysis Error", style={"color": "#dc3545", "marginBottom": "1rem"}),
                html.P(f"Error creating attention entropy analysis: {str(e)}", style={"fontSize": "1.1rem", "color": "#6c757d"})
            ], style={
                "textAlign": "center", 
                "padding": "3rem 2rem",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "10px",
                "border": "2px solid #f5c6cb"
            })
        ])

def create_highlighted_text(text, entities):
    """Create highlighted text with entity annotations."""
    if not entities:
        return text
    
    # Sort entities by start position (reverse order to avoid index shifting)
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
    
    highlighted_parts = []
    last_end = len(text)
    
    for entity in sorted_entities:
        start = entity.get('start', 0)
        end = entity.get('end', start)
        label = entity.get('label', 'MISC')
        entity_text = entity.get('word', text[start:end])
        
        # Add text after this entity
        if end < last_end:
            highlighted_parts.insert(0, text[end:last_end])
        
        # Add highlighted entity
        color_map = {
            'PER': '#FFE6E6',
            'ORG': '#E6F3FF', 
            'LOC': '#E6FFE6',
            'MISC': '#FFF0E6'
        }
        bg_color = color_map.get(label, '#F0F0F0')
        
        highlighted_parts.insert(0, html.Span(
            entity_text,
            style={
                'backgroundColor': bg_color,
                'padding': '2px 4px',
                'borderRadius': '3px',
                'border': f'1px solid {bg_color}',
                'marginRight': '2px'
            },
            title=f"{label}: {entity_text}"
        ))
        
        last_end = start
    
    # Add remaining text at the beginning
    if last_end > 0:
        highlighted_parts.insert(0, text[:last_end])
    
    return highlighted_parts

def create_entity_results_table(entities):
    """Create a table showing entity results."""
    if not entities:
        return html.P("No entities found.", className="text-muted")
    
    rows = []
    for i, entity in enumerate(entities, 1):
        rows.append(html.Tr([
            html.Td(i),
            html.Td(entity.get('word', '')),
            html.Td(entity.get('label', '')),
            html.Td(f"{entity.get('confidence', 0):.2f}" if 'confidence' in entity else 'N/A')
        ]))
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("#"),
                html.Th("Entity"),
                html.Th("Type"),
                html.Th("Confidence")
            ])
        ]),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True)

# Note: Callback moved to pages/error_analysis.py

# Callback to handle "Analyze Dataset" button with sample size input
@callback(
    Output("current-analysis-store", "data", allow_duplicate=True),
    [Input("analyze-dataset-btn", "n_clicks")],
    [State("sample-size-input", "value"),
     State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def analyze_dataset_with_sample_size(n_clicks, sample_size, selected_dataset, selected_model):
    """Handle analyze dataset button click with user-specified sample size."""
    logger.info(f"ANALYZE DATASET CALLBACK CALLED: n_clicks={n_clicks}, dataset={selected_dataset}")
    print(f"ANALYZE DATASET CALLBACK CALLED: n_clicks={n_clicks}, dataset={selected_dataset}")
    
    if not n_clicks or not selected_dataset or not selected_model:
        logger.info("ANALYZE DATASET CALLBACK: Early return due to missing inputs")
        print("ANALYZE DATASET CALLBACK: Early return due to missing inputs")
        return dash.no_update
    
    # Use user-specified sample size or default
    num_samples = sample_size if sample_size and sample_size > 0 else 200
    confidence_threshold = 0.7
    
    try:
        # Load dataset samples
        datasets = scan_datasets()
        dataset_info = None
        for task_datasets in datasets.values():
            if selected_dataset in task_datasets:
                dataset_info = task_datasets[selected_dataset]
                break
        
        if not dataset_info:
            logger.error(f"Dataset {selected_dataset} not found")
            return dash.no_update

        # Use appropriate split for different datasets
        if dataset_info["key"] == 'IMDb':
            split = 'test'
        else:
            split = 'dev'
        
        from utils.dataset_scanner import load_dataset_samples
        samples = load_dataset_samples(dataset_info["key"], 'sentiment', split=split, max_samples=num_samples)
        
        if not samples:
            logger.error(f"No samples found in dataset {dataset_info['key']}")
            return dash.no_update
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_info['key']} dataset")
        
        # Run sentiment analysis on all samples
        results = []
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"Processing sample {i+1}/{len(samples)}...")
            
            text = sample['text']
            true_label = sample['label']
            
            try:
                # Clear any cached models to avoid "Already borrowed" error
                if i % 10 == 0:  # Clear cache every 10 samples
                    model_api.clear_cache()
                
                prediction = model_api.get_sentiment(text)
                predicted_label = prediction.get("label", "Unknown")
                confidence = prediction.get("score", 0)
                
                # Convert labels to consistent format
                if str(predicted_label).lower() in ["positive", "1", "1.0"]:
                    predicted_label = "1"
                elif str(predicted_label).lower() in ["negative", "0", "0.0"]:
                    predicted_label = "0"
                
                results.append({
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "correct": str(predicted_label) == str(true_label)
                })
            except Exception as e:
                logger.error(f"Error processing sample: {str(e)}")
                continue
        
        # Calculate overall metrics
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total if total > 0 else 0
        
        # Identify high confidence errors
        high_conf_errors = [r for r in results if not r["correct"] and r["confidence"] >= confidence_threshold]
        
        # Categorize errors
        error_patterns = categorize_error_patterns(high_conf_errors)
        
        # Store analysis results in the analysis store
        analysis_results = {
            "results": results,
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "high_conf_errors": high_conf_errors,
            "error_patterns": error_patterns,
            "confidence_threshold": confidence_threshold,
            "sample_size_used": num_samples
        }
        
        # Debug: Check what we're storing
        print(f"DEBUG Storage: selected_dataset={selected_dataset}")
        print(f"DEBUG Storage: selected_model={selected_model}")
        print(f"DEBUG Storage: model_path={selected_model['model_path']}")
        print(f"DEBUG Storage: analysis_results keys={list(analysis_results.keys())}")
        
        # Clear previous analysis and store new results
        analysis_store.clear_analysis(selected_dataset, selected_model["model_path"])
        analysis_store.store_dataset_analysis(selected_dataset, selected_model["model_path"], analysis_results)
        
        logger.info(f"About to create backup file...")
        print(f"DEBUG: About to create backup file...")
        
        # Also save to file as backup
        import json
        import os
        
        try:
            # Ensure TempFiles directory exists
            temp_dir = "TempFiles"
            os.makedirs(temp_dir, exist_ok=True)
            
            backup_data = {
                "dataset": selected_dataset,
                "model_path": selected_model["model_path"],
                "results": analysis_results,
                "timestamp": time.time()
            }
            backup_file_path = os.path.join(temp_dir, "analysis_backup.json")
            with open(backup_file_path, "w") as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Analysis backup saved to: {backup_file_path}")
            print(f"DEBUG Storage: Data stored and backed up to file: {backup_file_path}")
            print(f"DEBUG Storage: All keys after storage={analysis_store.get_all_keys()}")
        except Exception as backup_error:
            logger.error(f"Failed to create backup file: {str(backup_error)}")
            print(f"DEBUG Storage: Backup failed - {str(backup_error)}")
        
        logger.info(f"Analysis completed: {accuracy:.2%} accuracy on {total} samples")
        
        return {"analysis_completed": True, "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Error in dataset analysis: {str(e)}")
        return dash.no_update

# Callback to create backup file whenever analysis is stored
@callback(
    Output("backup-status-store", "data", allow_duplicate=True),
    Input("current-analysis-store", "data"),
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def create_analysis_backup(analysis_data, selected_dataset, selected_model):
    """Create backup file whenever analysis is completed."""
    if not analysis_data or not selected_dataset or not selected_model:
        return dash.no_update
    
    logger.info(f"BACKUP CALLBACK TRIGGERED: analysis_data={analysis_data}")
    print(f"BACKUP CALLBACK TRIGGERED: analysis_data={analysis_data}")
    
    try:
        # Get stored analysis results
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, selected_model["model_path"])
        
        if not stored_analysis:
            logger.info("BACKUP: No stored analysis found")
            print("BACKUP: No stored analysis found")
            return dash.no_update
        
        # Create backup file
        import json
        import os
        
        # Ensure TempFiles directory exists
        temp_dir = "TempFiles"
        os.makedirs(temp_dir, exist_ok=True)
        
        backup_data = {
            "dataset": selected_dataset,
            "model_path": selected_model["model_path"],
            "results": stored_analysis["results"],
            "timestamp": time.time()
        }
        backup_file_path = os.path.join(temp_dir, "analysis_backup.json")
        with open(backup_file_path, "w") as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"BACKUP: Analysis backup saved to: {backup_file_path}")
        print(f"BACKUP: Analysis backup saved to: {backup_file_path}")
        
        return {"backup_created": True, "timestamp": time.time()}
        
    except Exception as backup_error:
        logger.error(f"BACKUP: Failed to create backup file: {str(backup_error)}")
        print(f"BACKUP: Failed to create backup file: {str(backup_error)}")
        return {"backup_failed": True, "error": str(backup_error)}


# Callback for Error Analysis info toggle
@callback(
    Output("error-analysis-info-collapse", "is_open"),
    Input("error-analysis-info-toggle", "n_clicks"),
    State("error-analysis-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_error_analysis_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

def handle_qa_model_viz(text, selected_model):
    """Handle QA model visualization (BertViz) by opening an external page layout."""
    try:
        from pages.qa_model_viz import create_layout as create_viz_layout
        content = create_viz_layout(default_context=text or "", default_question="")
        return content
    except Exception as e:
        logger.error(f"Error in QA model viz: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Visualization Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")

def handle_qa_counterfactual_flow(text, selected_model):
    """Handle QA counterfactual data flow analysis."""
    try:
        from pages.qa_counterfactual_flow import create_layout as create_cf_layout
        content = create_cf_layout(default_context=text or "", default_question="")
        return content
    except Exception as e:
        logger.error(f"Error in QA counterfactual flow: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")

def handle_qa_knowledge_assessment(text, selected_model):
    """Handle QA knowledge assessment analysis."""
    try:
        # Extract context and question from current QA interface
        context = ""
        question = ""
        
        # If text is provided, try to extract context and question
        if text:
            # For now, we'll use the text as context and provide a default question
            context = text
            question = "Please provide a question for analysis"
        
        # Import the QA knowledge assessment page
        from pages.qa_knowledge_assessment import create_layout
        
        # Create layout with default values
        content = create_layout(default_context=context, default_question=question)
        
        # Add a note about the current context if available
        if text:
            note = html.Div([
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    f"Note: Using current QA context from main interface. You can modify the inputs below or use the example button."
                ], color="info", className="mb-3")
            ])
            
            # Prepend the note to the content
            if hasattr(content, 'children') and isinstance(content.children, list):
                content.children.insert(1, note)  # Insert after header
        
        return content
        
    except Exception as e:
        logger.error(f"Error in QA knowledge assessment: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger mb-3"),
            html.H5("Analysis Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ], className="text-center py-5")


def handle_knowledge_competition_analysis(text, selected_model):
    """Handle knowledge competition analysis."""
    try:
        # Set the selected model for analysis
        if selected_model and isinstance(selected_model, dict):
            model_path = selected_model.get("model_path")
            model_type = selected_model.get("model_type", "qa")
            if model_path:
                model_api.set_selected_model(model_path, model_type)
                logger.info(f"Set model for knowledge competition: {model_path} (type: {model_type})")
        
        # Return the knowledge competition layout
        return create_knowledge_competition_layout()
        
    except Exception as e:
        logger.error(f"Error in knowledge competition analysis: {e}")
        return html.Div([
            dbc.Alert(f"Error in knowledge competition analysis: {str(e)}", color="danger")
        ])

# QA Knowledge Assessment Callbacks
@app.callback(
    [Output("qa-knowledge-context-input", "value"),
     Output("qa-knowledge-question-input", "value")],
    Input("load-qa-knowledge-example-btn", "n_clicks"),
    prevent_initial_call=True
)
def load_qa_knowledge_example(n_clicks):
    """Load an example for QA knowledge assessment."""
    if not n_clicks:
        return "", ""
    
    examples = [
        {
            "context": "The capital of France is Paris. Paris is known for the Eiffel Tower and is located on the Seine River.",
            "question": "What is the capital of France?"
        },
        {
            "context": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California.",
            "question": "Who founded Apple Inc.?"
        },
        {
            "context": "The Earth orbits around the Sun. This orbit takes approximately 365.25 days to complete, which is why we have a leap year every four years.",
            "question": "What does the Earth orbit around?"
        },
        {
            "context": "Shakespeare wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth. He lived during the Elizabethan era in England.",
            "question": "Who wrote Romeo and Juliet?"
        }
    ]
    
    # Cycle through examples based on click count
    example = examples[(n_clicks - 1) % len(examples)]
    return example["context"], example["question"]

@app.callback(
    Output("qa-knowledge-results", "children"),
    Input("run-qa-knowledge-btn", "n_clicks"),
    [State("qa-knowledge-context-input", "value"),
     State("qa-knowledge-question-input", "value"),
     State("qa-knowledge-num-runs", "value"),
     State("qa-knowledge-embedding-type", "value"),
     State("qa-knowledge-reduction-method", "value")],
    prevent_initial_call=True
)
def run_qa_knowledge_assessment_analysis(n_clicks, context, question, num_runs, embedding_type, reduction_method):
    """Run the QA knowledge assessment analysis."""
    if not n_clicks:
        return html.Div()
    
    # Validate inputs
    if not context or not context.strip():
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide a context for the analysis."
        ], color="warning")
    
    if not question or not question.strip():
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide a question for the analysis."
        ], color="warning")
    
    try:
        # Import the analysis function
        from pages.qa_knowledge_assessment import run_knowledge_assessment, create_knowledge_results
        
        # Run the knowledge assessment
        assessment_results = run_knowledge_assessment(
            context=context.strip(),
            question=question.strip(),
            num_runs=num_runs or 10,
            embedding_type=embedding_type or "text"
        )
        
        if "error" in assessment_results:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Assessment Error: {assessment_results['error']}"
            ], color="danger")
        
        # Create the results visualization
        results_content = create_knowledge_results(assessment_results, reduction_method or "both")
        
        return results_content
        
    except Exception as e:
        logger.error(f"Error in QA knowledge assessment: {str(e)}")
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Unexpected error: {str(e)}"
        ], color="danger")

# QA Model Visualization Callbacks (to ensure proper registration)
@app.callback(
    Output("model-viz-container", "children", allow_duplicate=True),
    Input("generate-model-viz", "n_clicks"),
    prevent_initial_call=True
)
def show_model_viz_loading(n_clicks):
    """Show loading indicator immediately when button is clicked."""
    if n_clicks:
        return html.Div([
            html.Div([
                html.I(className="fas fa-spinner fa-spin fa-3x text-primary mb-3"),
                html.H5("Generating Model Visualization...", className="text-primary"),
                html.P("Extracting attention patterns from the QA model", className="text-muted"),
                html.Div([
                    html.Div(className="progress-bar progress-bar-striped progress-bar-animated", 
                            style={"width": "100%"})
                ], className="progress mb-3", style={"height": "8px"}),
                html.Small("This may take a few seconds depending on the model size", className="text-muted")
            ], className="text-center p-4", style={
                "background": "rgba(255, 255, 255, 0.95)",
                "border-radius": "15px",
                "box-shadow": "0 4px 20px rgba(0, 0, 0, 0.1)"
            })
        ], className="d-flex justify-content-center align-items-center", 
           style={"min-height": "300px"})
    return dash.no_update

@app.callback(
    [Output("model-viz-container", "children"),
     Output("generate-model-viz", "disabled")],
    [Input("generate-model-viz", "n_clicks")],
    [State("model-viz-context", "value"),
     State("model-viz-question", "value")],
    prevent_initial_call=True
)
def generate_qa_model_visualization(n_clicks, context, question):
    """Generate the actual QA model visualization."""
    if not n_clicks:
        return html.Div(), False
    
    if not context or not context.strip():
        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide a context for visualization."
        ], color="warning")
        return error_alert, False
    
    if not question or not question.strip():
        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide a question for visualization."
        ], color="warning")
        return error_alert, False
    
    try:
        from pages.qa_model_viz import get_qa_model_data, create_3d_model_visualization
        import json
        
        logger.info(f"Starting QA model visualization for context: '{context[:50]}...' and question: '{question}'")
        
        # Get model data
        model_data = get_qa_model_data(context.strip(), question.strip())
        
        if not model_data:
            error_alert = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Failed to extract model data. Please check your inputs and ensure the QA model is loaded."
            ], color="danger")
            return error_alert, False
        
        logger.info("Model data extracted successfully, creating 3D visualization...")
        
        # Create single 3D visualization
        from pages.qa_model_viz import create_3d_model_visualization
        model_3d_fig = create_3d_model_visualization(model_data)
        
        success_div = html.Div([
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Successfully generated 3D model visualization for {model_data['num_layers']} layers and {model_data['num_heads']} heads!"
            ], color="success", className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-cube me-2"),
                    "3D Transformer Model Architecture"
                ]),
                dbc.CardBody([
                    dcc.Graph(
                        figure=model_3d_fig, 
                        id="model-3d-graph",
                        style={"height": "700px"}
                    )
                ])
            ])
        ])
        
        return success_div, False
        
    except Exception as e:
        logger.error(f"Error generating QA model visualization: {e}")
        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error: {str(e)}"
        ], color="danger")
        return error_alert, False

# QA Counterfactual Flow Callbacks
@app.callback(
    Output("cf-flow-results", "children"),
    Input("run-cf-flow-analysis", "n_clicks"),
    [State("cf-flow-factual-context", "value"),
     State("cf-flow-counterfactual-context", "value"),
     State("cf-flow-question", "value"),
     State("cf-flow-analysis-type", "value")],
    prevent_initial_call=True
)
def run_counterfactual_flow_analysis_app(n_clicks, factual_context, cf_context, question, analysis_type):
    """Run the counterfactual flow analysis."""
    if not n_clicks:
        return html.Div()
    
    # Validate inputs
    if not all([factual_context, cf_context, question]):
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide factual context, counterfactual context, and question."
        ], color="warning")
    
    try:
        from pages.qa_counterfactual_flow import extract_counterfactual_flow_data, create_overview_visualization
        import json
        
        # Extract flow data
        flow_data = extract_counterfactual_flow_data(
            factual_context.strip(),
            cf_context.strip(), 
            question.strip(),
            analysis_type
        )
        
        if not flow_data:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Failed to extract flow data. Please check your inputs."
            ], color="danger")
        
        # Create overview visualization
        overview_fig = create_overview_visualization(flow_data)
        
        # Store flow data for detail modal
        flow_data_json = {
            'factual_data': [layer.tolist() for layer in flow_data['factual_data']],
            'counterfactual_data': [layer.tolist() for layer in flow_data['counterfactual_data']],
            'factual_tokens': flow_data['factual_tokens'],
            'cf_tokens': flow_data['cf_tokens'],
            'num_layers': flow_data['num_layers'],
            'analysis_type': flow_data['analysis_type'],
            'factual_context': flow_data['factual_context'],
            'cf_context': flow_data['cf_context'],
            'question': flow_data['question']
        }
        
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Successfully analyzed {flow_data['num_layers']} layers using {analysis_type} analysis!"
            ], color="success", className="mb-4"),
            
            # Main Overview Card
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-line me-2"),
                    "Data Flow Overview"
                ]),
                dbc.CardBody([
                    dcc.Graph(
                        figure=overview_fig,
                        id="cf-flow-overview-graph",
                        style={"height": "500px"},
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                        }
                    ),
                    html.Hr(className="my-4"),
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle me-2"),
                            "Click any layer in the chart above to see detailed analysis below"
                        ], className="text-muted text-center mb-3"),
                        html.Div([
                            dbc.Button("Test Layer 0", id="test-layer-btn", color="outline-secondary", size="sm"),
                            html.Small(" (Debug: Click to test layer detail)", className="text-muted ms-2")
                        ], className="text-center")
                    ])
                ])
            ], className="mb-4"),
            
            # Integrated Layer Detail View (replaces modal)
            html.Div(
                id="cf-flow-layer-detail-section",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-mouse-pointer fa-2x text-muted mb-3"),
                                html.H5("Click any layer above to see detailed analysis", className="text-muted"),
                                html.P("Interactive layer-by-layer breakdown will appear here", className="text-muted")
                            ], className="text-center py-4")
                        ])
                    ], className="border-dashed", style={"border-style": "dashed", "border-color": "#dee2e6"})
                ]
            ),
            
            # Store data for callbacks
            html.Div(
                id="stored-cf-flow-data",
                children=json.dumps(flow_data_json),
                style={"display": "none"}
            )
        ])
        
    except Exception as e:
        logger.error(f"Error in counterfactual flow analysis: {e}")
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error: {str(e)}"
        ], color="danger")

@app.callback(
    Output("cf-flow-layer-detail-section", "children"),
    [Input("cf-flow-overview-graph", "clickData"),
     Input("test-layer-btn", "n_clicks")],
    State("stored-cf-flow-data", "children"),
    prevent_initial_call=False
)
def update_layer_detail_section_app(click_data, test_btn_clicks, stored_data):
    """Update the integrated layer detail section when a layer is clicked."""
    
    # Check if test button was clicked
    if test_btn_clicks and stored_data:
        try:
            import json
            import numpy as np
            from pages.qa_counterfactual_flow import create_enhanced_layer_detail
            
            flow_data = json.loads(stored_data)
            flow_data['factual_data'] = [np.array(layer) for layer in flow_data['factual_data']]
            flow_data['counterfactual_data'] = [np.array(layer) for layer in flow_data['counterfactual_data']]
            
            # Test with layer 0
            detail_content = create_enhanced_layer_detail(flow_data, 0)
            return detail_content
        except Exception as e:
            return dbc.Alert(f"Test error: {str(e)}", color="danger")
    
    # Handle graph clicks
    if not click_data:
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-mouse-pointer fa-2x text-muted mb-3"),
                        html.H5("Click any layer above to see detailed analysis", className="text-muted"),
                        html.P("Interactive layer-by-layer breakdown will appear here", className="text-muted")
                    ], className="text-center py-4")
                ])
            ], className="border-dashed", style={"border-style": "dashed", "border-color": "#dee2e6"})
        ])
    
    # Debug: Show that click was detected
    print(f"[DEBUG] Click detected! Click data: {click_data}")
    
    if not stored_data:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "No analysis data available. Please run the analysis first."
        ], color="warning")
    
    try:
        import json
        import numpy as np
        from pages.qa_counterfactual_flow import create_enhanced_layer_detail
        
        flow_data = json.loads(stored_data)
        
        # Debug logging
        print(f"[DEBUG] Flow data keys: {list(flow_data.keys())}")
        
        # Convert back to numpy arrays
        flow_data['factual_data'] = [np.array(layer) for layer in flow_data['factual_data']]
        flow_data['counterfactual_data'] = [np.array(layer) for layer in flow_data['counterfactual_data']]
        
        # Get clicked layer - handle both trace clicks
        point = click_data['points'][0]
        layer_idx = int(point['x'])
        
        print(f"[DEBUG] Processing layer {layer_idx} detail")
        
        # For now, just show which layer was clicked to confirm callback works
        return dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"🎯 Layer {layer_idx + 1} clicked! (Click data: {point.get('curveNumber', 'unknown')} trace)"
        ], color="success")
        
    except Exception as e:
        print(f"[DEBUG] Error in callback: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error loading layer detail: {str(e)}"
        ], color="danger")

def create_enhanced_layer_flow_visualization(flow_data, layer_idx):
    """Create enhanced visualization showing input→processing→output flow for a specific layer."""
    try:
        factual_data = flow_data['factual_data']
        cf_data = flow_data['counterfactual_data']
        factual_tokens = flow_data['factual_tokens']
        cf_tokens = flow_data['cf_tokens']
        analysis_type = flow_data['analysis_type']
        
        if layer_idx >= len(factual_data):
            return html.Div("Layer index out of range")
        
        # Get current layer data
        factual_current = factual_data[layer_idx]
        cf_current = cf_data[layer_idx]
        
        # Get previous layer (input) and next layer (output) if available
        factual_prev = factual_data[layer_idx - 1] if layer_idx > 0 else None
        cf_prev = cf_data[layer_idx - 1] if layer_idx > 0 else None
        factual_next = factual_data[layer_idx + 1] if layer_idx < len(factual_data) - 1 else None
        cf_next = cf_data[layer_idx + 1] if layer_idx < len(cf_data) - 1 else None
        
        # Create comprehensive flow visualization
        if analysis_type == "attention":
            return create_attention_flow_detail(
                factual_current, cf_current, factual_prev, cf_prev, 
                factual_next, cf_next, factual_tokens, cf_tokens, layer_idx
            )
        else:  # hidden states
            return create_hidden_state_flow_detail(
                factual_current, cf_current, factual_prev, cf_prev,
                factual_next, cf_next, factual_tokens, cf_tokens, layer_idx
            )
            
    except Exception as e:
        logger.error(f"Error creating enhanced visualization: {e}")
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error creating visualization: {str(e)}"
        ], color="danger")

def create_attention_flow_detail(factual_current, cf_current, factual_prev, cf_prev, 
                                factual_next, cf_next, factual_tokens, cf_tokens, layer_idx):
    """Create detailed attention flow visualization."""
    
    # Calculate attention statistics
    factual_attention_avg = np.mean(factual_current, axis=(1, 2))  # Average per head
    cf_attention_avg = np.mean(cf_current, axis=(1, 2))
    attention_diff = np.abs(cf_attention_avg - factual_attention_avg)
    
    # Find most different heads
    top_diff_heads = np.argsort(attention_diff)[-3:][::-1]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📥 Input: Attention Patterns", "📤 Output: Attention Changes",
            "🔍 Head-wise Differences", "🎯 Top Divergent Heads",
            "📊 Token Attention (Factual)", "📊 Token Attention (Counterfactual)"
        ),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Input patterns (previous layer influence)
    if factual_prev is not None:
        prev_factual_avg = np.mean(factual_prev, axis=(1, 2))
        prev_cf_avg = np.mean(cf_prev, axis=(1, 2))
        
        fig.add_trace(go.Heatmap(
            z=[prev_factual_avg, prev_cf_avg],
            x=[f"Head {i}" for i in range(len(prev_factual_avg))],
            y=["Factual Input", "Counterfactual Input"],
            colorscale="Blues",
            name="Input Patterns"
        ), row=1, col=1)
    
    # Output changes
    if factual_next is not None:
        next_factual_avg = np.mean(factual_next, axis=(1, 2))
        next_cf_avg = np.mean(cf_next, axis=(1, 2))
        output_diff = np.abs(next_cf_avg - next_factual_avg)
        
        fig.add_trace(go.Heatmap(
            z=[output_diff],
            x=[f"Head {i}" for i in range(len(output_diff))],
            y=["Output Difference"],
            colorscale="Reds",
            name="Output Changes"
        ), row=1, col=2)
    
    # Head-wise differences
    fig.add_trace(go.Bar(
        x=[f"Head {i}" for i in range(len(attention_diff))],
        y=attention_diff,
        marker_color='orange',
        name="Head Differences"
    ), row=2, col=1)
    
    # Top divergent heads
    fig.add_trace(go.Bar(
        x=[f"Head {i}" for i in top_diff_heads],
        y=[attention_diff[i] for i in top_diff_heads],
        marker_color='red',
        name="Top Divergent"
    ), row=2, col=2)
    
    # Token attention matrices for top divergent head
    if len(top_diff_heads) > 0:
        top_head = top_diff_heads[0]
        
        # Factual token attention
        factual_token_attn = factual_current[top_head]
        fig.add_trace(go.Heatmap(
            z=factual_token_attn,
            x=factual_tokens[:factual_token_attn.shape[1]],
            y=factual_tokens[:factual_token_attn.shape[0]],
            colorscale="Blues",
            name="Factual Attention"
        ), row=3, col=1)
        
        # Counterfactual token attention
        cf_token_attn = cf_current[top_head]
        fig.add_trace(go.Heatmap(
            z=cf_token_attn,
            x=cf_tokens[:cf_token_attn.shape[1]],
            y=cf_tokens[:cf_token_attn.shape[0]],
            colorscale="Reds",
            name="Counterfactual Attention"
        ), row=3, col=2)
    
    fig.update_layout(
        height=800,
        title_text=f"Layer {layer_idx} Attention Flow Analysis",
        showlegend=False
    )
    
    # Create summary cards
    summary_cards = create_flow_summary_cards(
        factual_current, cf_current, factual_tokens, cf_tokens, 
        layer_idx, "attention", top_diff_heads
    )
    
    return html.Div([
        summary_cards,
        dcc.Graph(figure=fig),
        create_layer_insights(factual_current, cf_current, layer_idx, "attention")
    ])

def create_hidden_state_flow_detail(factual_current, cf_current, factual_prev, cf_prev,
                                   factual_next, cf_next, factual_tokens, cf_tokens, layer_idx):
    """Create detailed hidden state flow visualization."""
    
    # Get sequence length and hidden size
    seq_len, hidden_size = factual_current[0].shape
    
    # Calculate token-wise differences
    factual_token_norms = np.linalg.norm(factual_current[0], axis=1)
    cf_token_norms = np.linalg.norm(cf_current[0], axis=1)
    token_diff = np.abs(cf_token_norms - factual_token_norms)
    
    # Find most different tokens
    top_diff_tokens = np.argsort(token_diff)[-5:][::-1]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "📥 Input→Processing: Token Magnitudes", "📤 Processing→Output: Changes",
            "🎯 Most Affected Tokens", "📊 Hidden Dimension Analysis"
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )
    
    # Input to processing
    if factual_prev is not None:
        prev_factual_norms = np.linalg.norm(factual_prev[0], axis=1)
        prev_cf_norms = np.linalg.norm(cf_prev[0], axis=1)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(prev_factual_norms))),
            y=prev_factual_norms,
            mode='lines+markers',
            name='Factual Input',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(prev_cf_norms))),
            y=prev_cf_norms,
            mode='lines+markers',
            name='Counterfactual Input',
            line=dict(color='red')
        ), row=1, col=1)
    
    # Processing to output
    if factual_next is not None:
        next_factual_norms = np.linalg.norm(factual_next[0], axis=1)
        next_cf_norms = np.linalg.norm(cf_next[0], axis=1)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(next_factual_norms))),
            y=next_factual_norms,
            mode='lines+markers',
            name='Factual Output',
            line=dict(color='darkblue')
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(next_cf_norms))),
            y=next_cf_norms,
            mode='lines+markers',
            name='Counterfactual Output',
            line=dict(color='darkred')
        ), row=1, col=2)
    
    # Most affected tokens
    token_labels = [factual_tokens[i] if i < len(factual_tokens) else f"Token_{i}" for i in top_diff_tokens]
    fig.add_trace(go.Bar(
        x=token_labels,
        y=[token_diff[i] for i in top_diff_tokens],
        marker_color='purple',
        name="Token Differences"
    ), row=2, col=1)
    
    # Hidden dimension analysis (sample of dimensions)
    sample_dims = min(50, hidden_size)
    dim_indices = np.linspace(0, hidden_size-1, sample_dims, dtype=int)
    
    factual_dim_avg = np.mean(np.abs(factual_current[0][:, dim_indices]), axis=0)
    cf_dim_avg = np.mean(np.abs(cf_current[0][:, dim_indices]), axis=0)
    
    fig.add_trace(go.Heatmap(
        z=[factual_dim_avg, cf_dim_avg],
        x=[f"Dim {i}" for i in dim_indices],
        y=["Factual", "Counterfactual"],
        colorscale="Viridis",
        name="Dimension Analysis"
    ), row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text=f"Layer {layer_idx} Hidden State Flow Analysis",
        showlegend=True
    )
    
    # Create summary cards
    summary_cards = create_flow_summary_cards(
        factual_current, cf_current, factual_tokens, cf_tokens,
        layer_idx, "hidden", top_diff_tokens
    )
    
    return html.Div([
        summary_cards,
        dcc.Graph(figure=fig),
        create_layer_insights(factual_current, cf_current, layer_idx, "hidden")
    ])

def create_flow_summary_cards(factual_data, cf_data, factual_tokens, cf_tokens, 
                             layer_idx, analysis_type, top_indices):
    """Create summary cards showing key metrics."""
    
    if analysis_type == "attention":
        factual_avg = np.mean(factual_data)
        cf_avg = np.mean(cf_data)
        max_diff_head = top_indices[0] if len(top_indices) > 0 else 0
        
        metric1_label = "Avg Attention"
        metric2_label = "Max Diff Head"
        metric2_value = f"Head {max_diff_head}"
    else:  # hidden
        factual_avg = np.mean(np.linalg.norm(factual_data[0], axis=1))
        cf_avg = np.mean(np.linalg.norm(cf_data[0], axis=1))
        max_diff_token = top_indices[0] if len(top_indices) > 0 else 0
        
        metric1_label = "Avg Magnitude"
        metric2_label = "Max Diff Token"
        metric2_value = factual_tokens[max_diff_token] if max_diff_token < len(factual_tokens) else f"Token_{max_diff_token}"
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{factual_avg:.3f}", className="text-primary mb-1"),
                    html.P("Factual", className="text-muted mb-0"),
                    html.Small(metric1_label, className="text-info")
                ])
            ], color="light", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{cf_avg:.3f}", className="text-danger mb-1"),
                    html.P("Counterfactual", className="text-muted mb-0"),
                    html.Small(metric1_label, className="text-info")
                ])
            ], color="light", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{abs(cf_avg - factual_avg):.3f}", className="text-warning mb-1"),
                    html.P("Difference", className="text-muted mb-0"),
                    html.Small("Absolute Diff", className="text-info")
                ])
            ], color="light", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(metric2_value, className="text-info mb-1"),
                    html.P(metric2_label, className="text-muted mb-0"),
                    html.Small("Most Affected", className="text-info")
                ])
            ], color="light", outline=True)
        ], width=3)
    ], className="mb-4")

def create_layer_insights(factual_data, cf_data, layer_idx, analysis_type):
    """Create insights about what happened in this layer."""
    
    if analysis_type == "attention":
        factual_entropy = -np.sum(factual_data * np.log(factual_data + 1e-10), axis=-1)
        cf_entropy = -np.sum(cf_data * np.log(cf_data + 1e-10), axis=-1)
        
        avg_factual_entropy = np.mean(factual_entropy)
        avg_cf_entropy = np.mean(cf_entropy)
        
        if avg_cf_entropy > avg_factual_entropy:
            insight = "Counterfactual processing shows more diffuse attention (higher entropy), suggesting uncertainty."
        else:
            insight = "Counterfactual processing shows more focused attention (lower entropy), suggesting confident redirection."
    else:  # hidden
        factual_variance = np.var(factual_data[0], axis=1)
        cf_variance = np.var(cf_data[0], axis=1)
        
        avg_factual_var = np.mean(factual_variance)
        avg_cf_var = np.mean(cf_variance)
        
        if avg_cf_var > avg_factual_var:
            insight = "Counterfactual processing shows higher representation variance, indicating active information restructuring."
        else:
            insight = "Counterfactual processing shows lower representation variance, suggesting stable but redirected processing."
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-lightbulb me-2"),
            f"Layer {layer_idx} Processing Insights"
        ]),
        dbc.CardBody([
            html.P(insight, className="mb-3"),
            html.Div([
                html.Strong("What happened here: "),
                html.Span(f"The model processed factual vs counterfactual information differently at layer {layer_idx}. "),
                html.Span("This layer shows how the competing information sources influence the model's internal representations.")
            ])
        ])
    ], className="mt-3")

if __name__ == "__main__":
    print("Starting App")
    app.run_server(debug=DEBUG_MODE, host=HOST, port=PORT)
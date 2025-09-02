"""
Logit Matrix Heatmap visualization page for both sentiment and NER tasks.
Provides detailed analysis of raw model predictions before softmax.
"""
import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import logging

from models.api import model_api
from components.visualizations import create_logit_heatmap, create_logit_comparison_chart

logger = logging.getLogger(__name__)

def create_logit_matrix_modal():
    """Create the logit matrix analysis modal."""
    return dbc.Modal([
        dbc.ModalHeader([
            html.Div([
                html.I(className="fas fa-chart-area me-2"),
                html.Span("Logit Matrix Analysis", className="fw-bold")
            ])
        ]),
        dbc.ModalBody([
            # Input section
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-edit me-2"),
                    html.Span("Enter Text for Analysis")
                ]),
                dbc.CardBody([
                    dbc.Textarea(
                        id="logit-text-input",
                        placeholder="Enter text to analyze logit patterns...",
                        rows=3,
                        className="mb-3"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="fas fa-chart-bar me-2"), "Analyze Logits"],
                                id="analyze-logits-btn",
                                color="primary",
                                className="w-100"
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="fas fa-info-circle me-2"), "What are Logits?"],
                                id="logit-info-btn",
                                color="info",
                                outline=True,
                                className="w-100"
                            )
                        ], md=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Loading spinner
            dbc.Spinner(
                html.Div(id="logit-analysis-content"),
                color="primary",
                type="border",
                fullscreen=False
            ),
            
            # Info collapse
            dbc.Collapse([
                dbc.Alert([
                    html.H5([html.I(className="fas fa-lightbulb me-2"), "Understanding Logits"], className="alert-heading"),
                    html.P([
                        "Logits are the raw, unnormalized prediction scores from the neural network before applying softmax. ",
                        "They provide insight into the model's confidence and decision-making process:"
                    ]),
                    html.Ul([
                        html.Li([html.Strong("Raw Scores: "), "Higher logit values indicate stronger model confidence"]),
                        html.Li([html.Strong("Relative Differences: "), "The gap between logits shows decision certainty"]),
                        html.Li([html.Strong("Before Softmax: "), "Logits can be negative and have no upper bound"]),
                        html.Li([html.Strong("Model Debugging: "), "Helps identify where the model is uncertain or biased"])
                    ]),
                    html.Hr(),
                    html.P([
                        html.Strong("For Sentiment: "), "Shows confidence in POSITIVE vs NEGATIVE classification"
                    ]),
                    html.P([
                        html.Strong("For NER: "), "Shows token-level confidence for each entity type"
                    ])
                ], color="info", className="mb-0")
            ], id="logit-info-collapse", is_open=False)
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-logit-modal", className="ms-auto", color="secondary")
        ])
    ], id="logit-matrix-modal", size="xl", scrollable=True, is_open=False)

def create_logit_analysis_content(logit_data, task_type):
    """Create the main content for logit analysis."""
    if not logit_data:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Failed to analyze logits. Please check your input and try again."
        ], color="warning")
    
    # Create prediction summary card
    if task_type == "sentiment":
        prediction_card = create_sentiment_prediction_summary(logit_data)
    else:
        prediction_card = create_ner_prediction_summary(logit_data)
    
    # Create visualizations
    main_viz = create_logit_heatmap(logit_data, task_type)
    comparison_viz = create_logit_comparison_chart(logit_data, task_type)
    
    return html.Div([
        prediction_card,
        
        # Main logit visualization
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-chart-area me-2"),
                html.Span("Logit Matrix Visualization")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    figure=main_viz,
                    config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}}
                )
            ])
        ], className="mb-4"),
        
        # Comparison visualization
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-balance-scale me-2"),
                html.Span("Logits vs Probabilities Comparison")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    figure=comparison_viz,
                    config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}}
                ),
                html.Small([
                    "This chart shows the relationship between raw logit scores and final probabilities after softmax normalization."
                ], className="text-muted")
            ])
        ], className="mb-4"),
        
        # Technical details
        create_technical_details_card(logit_data, task_type)
    ])

def create_sentiment_prediction_summary(logit_data):
    """Create prediction summary for sentiment analysis."""
    predicted_class = logit_data["predicted_class"]
    confidence = logit_data["confidence"]
    logits = logit_data["logits"]
    
    # Calculate logit statistics
    max_logit = max(logits)
    min_logit = min(logits)
    logit_range = max_logit - min_logit
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-heart me-2"),
            html.Span("Sentiment Prediction Summary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4([
                        html.Span(predicted_class, className=f"badge bg-{'success' if predicted_class == 'POSITIVE' else 'danger'} me-2"),
                        html.Small(f"{confidence:.1%} confidence", className="text-muted")
                    ])
                ], md=6),
                dbc.Col([
                    html.P([
                        html.Strong("Logit Range: "), f"{logit_range:.3f}",
                        html.Br(),
                        html.Strong("Decision Margin: "), f"{abs(logits[0] - logits[1]):.3f}"
                    ], className="mb-0")
                ], md=6)
            ]),
            html.Hr(),
            html.P([
                html.Strong("Text: "), f'"{logit_data["text"]}"'
            ], className="mb-0 text-muted")
        ])
    ], className="mb-4")

def create_ner_prediction_summary(logit_data):
    """Create prediction summary for NER analysis."""
    tokens = logit_data["tokens"]
    predicted_labels = logit_data["predicted_labels"]
    token_confidences = logit_data["token_confidences"]
    
    # Count entities
    entity_counts = {}
    for label in predicted_labels:
        if label != "O":
            entity_type = label.split("-")[-1] if "-" in label else label
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Calculate average confidence
    avg_confidence = np.mean(token_confidences)
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-tags me-2"),
            html.Span("NER Prediction Summary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Detected Entities:"),
                    html.Div([
                        dbc.Badge(f"{entity_type}: {count}", color="primary", className="me-2 mb-1")
                        for entity_type, count in entity_counts.items()
                    ] if entity_counts else [dbc.Badge("No entities detected", color="secondary")])
                ], md=6),
                dbc.Col([
                    html.P([
                        html.Strong("Tokens Analyzed: "), f"{len(tokens)}",
                        html.Br(),
                        html.Strong("Average Confidence: "), f"{avg_confidence:.1%}"
                    ], className="mb-0")
                ], md=6)
            ]),
            html.Hr(),
            html.P([
                html.Strong("Text: "), f'"{logit_data["text"]}"'
            ], className="mb-0 text-muted")
        ])
    ], className="mb-4")

def create_technical_details_card(logit_data, task_type):
    """Create technical details card."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-cogs me-2"),
            html.Span("Technical Details")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Model Output Shape:"),
                    html.P([
                        f"Classes: {logit_data['num_classes']}",
                        html.Br(),
                        f"Task Type: {task_type.title()}"
                    ] + ([html.Br(), f"Sequence Length: {logit_data['sequence_length']}"] if task_type == "ner" else []))
                ], md=4),
                dbc.Col([
                    html.H6("Logit Statistics:"),
                    html.P([
                        f"Max Logit: {max(np.array(logit_data['logits']).flatten()):.3f}",
                        html.Br(),
                        f"Min Logit: {min(np.array(logit_data['logits']).flatten()):.3f}",
                        html.Br(),
                        f"Mean Logit: {np.mean(np.array(logit_data['logits'])):.3f}"
                    ])
                ], md=4),
                dbc.Col([
                    html.H6("Class Labels:"),
                    html.P([
                        html.Div([
                            dbc.Badge(class_name, color="outline-secondary", className="me-1 mb-1")
                            for class_name in logit_data["class_names"]
                        ])
                    ])
                ], md=4)
            ])
        ])
    ], className="mb-4")

# Callbacks
@callback(
    Output("logit-matrix-modal", "is_open", allow_duplicate=True),
    [Input("open-logit-matrix", "n_clicks"),
     Input("close-logit-modal", "n_clicks")],
    [State("logit-matrix-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_logit_modal(open_clicks, close_clicks, is_open):
    """Toggle the logit matrix modal."""
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@callback(
    Output("logit-info-collapse", "is_open"),
    [Input("logit-info-btn", "n_clicks")],
    [State("logit-info-collapse", "is_open")]
)
def toggle_logit_info(n_clicks, is_open):
    """Toggle the logit information collapse."""
    if n_clicks:
        return not is_open
    return is_open

@callback(
    Output("logit-analysis-content", "children"),
    [Input("analyze-logits-btn", "n_clicks")],
    [State("logit-text-input", "value")]
)
def analyze_logits(n_clicks, text):
    """Analyze logits for the input text."""
    if not n_clicks or not text or not text.strip():
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Enter some text and click 'Analyze Logits' to see the logit matrix visualization."
            ], color="info")
        ])
    
    try:
        # Get current model info
        model_info = model_api.get_model_info()
        if not model_info:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "No model selected. Please select a model first."
            ], color="warning")
        
        task_type = model_info["type"]
        
        # Get logit matrix data
        logit_data = model_api.get_logit_matrix(text.strip())
        
        if not logit_data:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Failed to extract logit matrix. Please try again."
            ], color="danger")
        
        return create_logit_analysis_content(logit_data, task_type)
        
    except Exception as e:
        logger.error(f"Error in logit analysis: {str(e)}")
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error analyzing logits: {str(e)}"
        ], color="danger")
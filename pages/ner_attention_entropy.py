"""
NER attention entropy visualization page.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import logging

from models.api import model_api

logger = logging.getLogger(__name__)

def create_layout():
    """
    Create the layout for the NER attention entropy visualization page.
    
    Returns:
        A Dash layout object
    """
    layout = html.Div([
        dbc.Container([
            # Header Section
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3([
                            html.I(className="fas fa-chart-line me-2 text-info"),
                            "Attention Entropy Analysis"
                        ], className="mb-2"),
                        html.P("Analyze attention patterns and entropy in NER model predictions", 
                               className="text-muted mb-4")
                    ])
                ], width=12)
            ]),
            
            # Input Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-keyboard me-2"),
                                "Text Input & Settings"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Label("Enter text to analyze:", className="fw-bold mb-2"),
                            dbc.Textarea(
                                id="ner-entropy-input", 
                                value="John Smith works at Microsoft in Seattle. He was born on January 15, 1985.",
                                className="mb-4",
                                style={"height": "120px"},
                                placeholder="Enter your text here..."
                            ),
                            
                            # Settings section
                            html.Hr(),
                            html.H6("Analysis Settings", className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Layer:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id="ner-entropy-layer",
                                        options=[{"label": f"Layer {i}", "value": i} for i in range(12)],
                                        value=0,
                                        clearable=False
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Head:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id="ner-entropy-head",
                                        options=[{"label": f"Head {i}", "value": i} for i in range(12)],
                                        value=0,
                                        clearable=False
                                    )
                                ], width=6)
                            ], className="mb-4"),
                            
                            # Generate button
                            dbc.Button([
                                html.I(className="fas fa-chart-line me-2"),
                                "Analyze Attention"
                            ], id="ner-entropy-button", color="info", size="lg", className="w-100"),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            # Results Section
            dbc.Row([
                dbc.Col([
                    dbc.Spinner([
                        html.Div(id="ner-entropy-error-output", className="text-danger mb-3"),
                        html.Div(id="ner-entropy-results")
                    ], color="info", type="border", fullscreen=False),
                ], width=12),
            ]),
            
            # Info Section
            dbc.Row([
                dbc.Col([
                    dbc.Collapse([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6([
                                    html.I(className="fas fa-info-circle me-2"),
                                    "About Attention Entropy"
                                ], className="mb-0")
                            ]),
                            dbc.CardBody([
                                html.P([
                                    html.Strong("Attention Entropy"), 
                                    " measures how focused or diffuse the attention patterns are in transformer models."
                                ]),
                                html.Ul([
                                    html.Li([html.Strong("Low Entropy:"), " Focused attention on specific tokens"]),
                                    html.Li([html.Strong("High Entropy:"), " Diffuse attention across many tokens"]),
                                    html.Li([html.Strong("Layer Analysis:"), " Different layers capture different patterns"]),
                                    html.Li([html.Strong("Head Analysis:"), " Multiple attention heads work in parallel"]),
                                ]),
                                html.P([
                                    "This visualization helps understand how the NER model focuses on different parts ",
                                    "of the input when making entity predictions."
                                ], className="text-muted mb-0")
                            ]),
                        ], className="border-info"),
                    ], id="ner-entropy-info-collapse", is_open=False),
                    
                    dbc.Button([
                        html.I(className="fas fa-question-circle me-2"),
                        "Learn More About Attention Entropy"
                    ], id="ner-entropy-info-toggle", color="outline-info", size="sm", className="mt-3")
                ], width=12),
            ], className="mt-4"),
            
        ], fluid=True, className="p-3"),
    ])
    
    return layout

def create_attention_entropy_visualization(text, layer=0, head=0):
    """
    Create attention entropy visualization for NER.
    
    Args:
        text: Input text
        layer: Layer index
        head: Head index
        
    Returns:
        HTML div with visualizations
    """
    try:
        # Get attention weights from NER model
        attention_data = model_api.get_ner_attention(text, layer, head)
        
        if not attention_data or "attention_weights" not in attention_data:
            return html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                html.P("Could not retrieve attention data from the model.")
            ])
        
        tokens = attention_data.get("tokens", text.split())
        attention_weights = attention_data["attention_weights"]
        
        # Calculate entropy for each token
        entropies = []
        for i, token_attention in enumerate(attention_weights):
            # Normalize attention weights
            attention_probs = np.array(token_attention)
            attention_probs = attention_probs / (attention_probs.sum() + 1e-8)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8))
            entropies.append(entropy)
        
        # Create entropy bar chart
        entropy_fig = go.Figure()
        entropy_fig.add_trace(go.Bar(
            x=tokens,
            y=entropies,
            name="Attention Entropy",
            marker_color="lightblue",
            text=[f"{e:.2f}" for e in entropies],
            textposition="outside"
        ))
        
        entropy_fig.update_layout(
            title=f"Attention Entropy by Token (Layer {layer}, Head {head})",
            xaxis_title="Tokens",
            yaxis_title="Entropy",
            height=400,
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis=dict(tickangle=-45)
        )
        
        # Create attention heatmap
        attention_matrix = np.array(attention_weights)
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale="Blues",
            showscale=True,
            hoverongaps=False
        ))
        
        heatmap_fig.update_layout(
            title=f"Attention Heatmap (Layer {layer}, Head {head})",
            xaxis_title="Target Tokens",
            yaxis_title="Source Tokens",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Create summary statistics
        avg_entropy = np.mean(entropies)
        max_entropy = np.max(entropies)
        min_entropy = np.min(entropies)
        max_entropy_token = tokens[np.argmax(entropies)]
        min_entropy_token = tokens[np.argmin(entropies)]
        
        stats_card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Entropy Statistics"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Average Entropy", className="text-muted"),
                        html.H4(f"{avg_entropy:.3f}", className="text-primary")
                    ], width=4),
                    dbc.Col([
                        html.H6("Highest Entropy", className="text-muted"),
                        html.H4(f"{max_entropy:.3f}", className="text-success"),
                        html.Small(f"Token: {max_entropy_token}", className="text-muted")
                    ], width=4),
                    dbc.Col([
                        html.H6("Lowest Entropy", className="text-muted"),
                        html.H4(f"{min_entropy:.3f}", className="text-info"),
                        html.Small(f"Token: {min_entropy_token}", className="text-muted")
                    ], width=4)
                ])
            ])
        ], className="mb-4")
        
        return html.Div([
            stats_card,
            
            dbc.Card([
                dbc.CardHeader("Attention Entropy by Token"),
                dbc.CardBody([
                    dcc.Graph(figure=entropy_fig, config={'displayModeBar': True})
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Attention Weight Heatmap"),
                dbc.CardBody([
                    dcc.Graph(figure=heatmap_fig, config={'displayModeBar': True})
                ])
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating attention entropy visualization: {str(e)}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.P(f"Error: {str(e)}")
        ])

@callback(
    [Output("ner-entropy-results", "children"),
     Output("ner-entropy-error-output", "children")],
    [Input("ner-entropy-button", "n_clicks")],
    [State("ner-entropy-input", "value"),
     State("ner-entropy-layer", "value"),
     State("ner-entropy-head", "value")],
    prevent_initial_call=True
)
def update_ner_entropy_visualization(n_clicks, input_text, layer, head):
    """Update NER attention entropy visualization."""
    if not n_clicks:
        return "", ""
    
    if not input_text or not input_text.strip():
        return "", "Please enter some text to analyze."

    try:
        # Check if model is loaded
        if not model_api.selected_model_path:
            return "", "Error: No model selected. Please select a model from the main interface first."
        
        visualization = create_attention_entropy_visualization(input_text, layer, head)
        return visualization, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if "meta tensor" in str(e).lower():
            error_msg = "Error: Model loading issue. Please try reloading the model or use a different model."
        return "", error_msg

# Callback for info toggle
@callback(
    Output("ner-entropy-info-collapse", "is_open"),
    Input("ner-entropy-info-toggle", "n_clicks"),
    State("ner-entropy-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_ner_entropy_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open
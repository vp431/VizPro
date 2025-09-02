"""
Sentiment analysis attention entropy visualization page - Professional Design.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from components.visualizations import create_clickable_entropy_heatmap, create_attention_heatmap_matrix
from models.api import model_api

def create_layout():
    """
    Create the layout for the sentiment attention entropy visualization page.
    
    Returns:
        A Dash layout object
    """
    layout = html.Div([
        # Main Entropy Container
        html.Div([
            # Header Section
            html.Div([
                html.H4([
                    html.I(className="fas fa-chart-area", style={"marginRight": "0.5rem"}),
                    "Attention Entropy Analysis"
                ]),
                html.P("Visualize how diffuse or focused the attention is across layers and heads", 
                       className="entropy-header-subtitle")
            ], className="entropy-header"),
            
            # Input Section
            html.Div([
                html.Label("Enter text to analyze:", className="entropy-input-label"),
                dcc.Textarea(
                    id="entropy-input", 
                    value="This movie was fantastic! The acting was superb and the plot was engaging.",
                    className="entropy-textarea",
                    placeholder="Enter your text here..."
                ),
                
                # Generate Button
                dbc.Button([
                    html.I(className="fas fa-chart-area me-2"),
                    "Analyze Entropy"
                ], id="entropy-visualize-button", color="info", size="lg", className="w-100"),
                
                # Error Output
                html.Div(id="entropy-error-output", className="entropy-error")
            ], className="entropy-input-section"),
            
            # Hidden data store
            dcc.Store(id="entropy-data", data={}),
            
            # Results Section
            html.Div([
                html.Div(id="entropy-loading-container", children=[], className="entropy-loading"),
                html.Div(id="entropy-result"),
            ], className="entropy-results-container"),
            
            # Info Section
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "Learn More About Attention Entropy"
                ], id="entropy-info-toggle", color="outline-info", size="sm", className="mt-3"),
                
                dbc.Collapse([
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle", style={"marginRight": "0.5rem"}),
                            "About Attention Entropy"
                        ]),
                        html.P([
                            html.Strong("Attention Entropy"), 
                            " measures how diffuse or focused the attention mechanism is in each layer and head of the transformer model."
                        ]),
                        html.Ul([
                            html.Li([html.Strong("High Entropy (Yellow/Red):"), " Attention is spread across many tokens"]),
                            html.Li([html.Strong("Low Entropy (Blue/Purple):"), " Attention is focused on specific tokens"]),
                            html.Li([html.Strong("Interactive:"), " Click on any cell to see the detailed attention matrix"]),
                        ]),
                        html.P([
                            "This visualization helps understand how different layers and heads process information differently. ",
                            "Early layers often show more diffuse attention, while later layers tend to be more focused."
                        ], className="entropy-explanation")
                    ], className="entropy-info-content")
                ], id="entropy-info-collapse", is_open=False),
            ], className="entropy-info-section"),
            
        ], className="entropy-modal-container"),
    ])
    
    return layout

@callback(
    Output("entropy-data", "data"),
    Input("entropy-visualize-button", "n_clicks"),
    State("entropy-input", "value"),
    prevent_initial_call=True,
)
def process_entropy_input(n_clicks, input_text):
    if not n_clicks or not input_text or not input_text.strip():
        return {}

    try:
        # Check if model is loaded
        if not model_api.selected_model_path:
            return {}
        
        entropy_result = model_api.get_attention_entropy(input_text)
        sentiment_result = model_api.get_sentiment_with_attention(input_text)

        if not entropy_result or "entropy" not in entropy_result:
            return {}

        attention_data = {
            "tokens": sentiment_result.get("tokens", []),
            "attentions": [att.tolist() if isinstance(att, np.ndarray) else att for att in sentiment_result.get("attentions", [])],
            "entropy": entropy_result["entropy"].tolist() if isinstance(entropy_result.get("entropy"), np.ndarray) else entropy_result.get("entropy", []),
        }
        
        return attention_data
    except Exception as e:
        return {}

@callback(
    Output("entropy-result", "children"),
    Input("entropy-data", "data"),
    prevent_initial_call=True
)
def update_entropy_visualization(data):
    if not data or "entropy" not in data or not data["entropy"]:
        return html.Div()

    try:
        entropy = np.array(data["entropy"])
        fig = create_clickable_entropy_heatmap(entropy)
        
        return html.Div([
            dcc.Graph(
                id="attention-entropy-heatmap", 
                figure=fig, 
                config={'responsive': True, 'displayModeBar': True},
                style={"width": "100%", "height": "400px"}
            ),
            dcc.Store(id="entropy-click-data"),
            html.Div(id="entropy-click-graph", className="mt-4")
        ])
    except Exception as e:
        return html.Div(f"Error creating entropy visualization: {str(e)}", className="text-danger")

@callback(
    Output("entropy-click-graph", "children"),
    Input("attention-entropy-heatmap", "clickData"),
    State("entropy-data", "data"),
    prevent_initial_call=True
)
def display_entropy_click_data(clickData, data):
    if not clickData or not data or "attentions" not in data or not data.get("attentions"):
        return html.Div()

    try:
        points = clickData["points"][0]
        layer_idx = int(points["y"].split()[-1])
        head_idx = int(points["x"].split()[-1])
        
        tokens = data.get("tokens", [])
        attentions = data.get("attentions", [])

        if not tokens or not attentions or layer_idx >= len(attentions):
            return html.Div(f"Error: No attention data available for layer {layer_idx}", className="text-danger")

        fig = create_attention_heatmap_matrix(tokens, attentions, layer_idx, head_idx, height=500)
        
        return html.Div([
            html.H5(f"Attention Matrix for Layer {layer_idx}, Head {head_idx}", className="mt-4"),
            dcc.Graph(figure=fig, config={'responsive': True})
        ])
    except Exception as e:
        return html.Div(f"Error displaying entropy click data: {str(e)}", className="text-danger")

# Callback for info toggle
@callback(
    Output("entropy-info-collapse", "is_open"),
    Input("entropy-info-toggle", "n_clicks"),
    State("entropy-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_entropy_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

"""
Sentiment analysis token embeddings visualization page - Professional Design.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go

from components.visualizations import create_embedding_plot
from models.api import model_api

def create_layout():
    """
    Create the professional layout for the sentiment token embeddings visualization page.
    
    Returns:
        A Dash layout object with custom styling
    """
    layout = html.Div([
        # Main Embeddings Container
        html.Div([
            # Header Section
            html.Div([
                html.H4([
                    html.I(className="fas fa-project-diagram", style={"marginRight": "0.5rem"}),
                    "Token Embeddings Visualization"
                ]),
                html.P("Visualize token embeddings in 2D space using dimensionality reduction", 
                       className="token-embed-header-subtitle")
            ], className="token-embed-header"),
            
            # Input Section
            html.Div([
                html.Label("Enter text to analyze:", className="token-embed-input-label"),
                dcc.Textarea(
                    id="token-embed-input-text", 
                    value="This movie was fantastic! The acting was superb and the plot was engaging.",
                    className="token-embed-textarea",
                    placeholder="Enter your text here..."
                ),
                
                # Settings section
                html.Div([
                    html.Label("Dimensionality Reduction Method:", className="token-embed-setting-label"),
                    dcc.RadioItems(
                        id="token-embed-reduction-method",
                        options=[
                            {"label": "t-SNE (better for clusters)", "value": "tsne"},
                            {"label": "PCA (preserves global structure)", "value": "pca"}
                        ],
                        value="tsne",
                        inline=True,
                        className="token-embed-radio-spacing"
                    ),
                ], className="token-embed-settings-group"),
                
                # Generate Button
                dbc.Button([
                    html.I(className="fas fa-sitemap me-2"),
                    "Visualize Embeddings"
                ], id="token-embed-visualize-button", color="success", size="lg", className="w-100"),
                
                # Error Output
                html.Div(id="token-embed-error-output", className="token-embed-error")
            ], className="token-embed-input-section"),
            
            # Results Section
            html.Div([
                html.Div(id="token-embed-loading-container", children=[], className="token-embed-loading"),
                html.Div([
                    html.Div("Token Embeddings Visualization", className="token-embed-result-header"),
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id="token-embed-plot", 
                                config={'responsive': True, 'displayModeBar': False},
                                style={"width": "100%", "height": "450px"}
                            )
                        ], className="token-embed-chart-container"),
                        html.P([
                            "Each point represents a token from your text. Colors indicate token position in the sequence. ",
                            "Closer points have similar semantic representations."
                        ], className="token-embed-explanation")
                    ], className="token-embed-result-body")
                ], id="token-embed-result-card", className="token-embed-result-card", style={"display": "none"}),
            ], className="token-embed-results-container"),
            
            # Info Section
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "Learn More About Token Embeddings"
                ], id="token-embed-info-toggle", color="outline-success", size="sm", className="mt-3"),
                
                dbc.Collapse([
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle", style={"marginRight": "0.5rem"}),
                            "About Token Embeddings"
                        ]),
                        html.P([
                            html.Strong("Token Embeddings"), 
                            " are high-dimensional vector representations of words/tokens that capture semantic meaning."
                        ]),
                        html.Ul([
                            html.Li([html.Strong("t-SNE:"), " Better at revealing local clusters and groupings"]),
                            html.Li([html.Strong("PCA:"), " Preserves global structure and variance"]),
                            html.Li([html.Strong("Colors:"), " Represent token position in the sequence"]),
                            html.Li([html.Strong("Distance:"), " Closer tokens have similar representations"]),
                        ]),
                        html.P([
                            "This visualization helps understand how the model represents different words internally. ",
                            "Similar words should appear close together, and the model's understanding of relationships ",
                            "between words becomes visible in the 2D projection."
                        ], className="token-embed-explanation")
                    ], className="token-embed-info-content")
                ], id="token-embed-info-collapse", is_open=False),
            ], className="token-embed-info-section"),
            
        ], className="token-embed-modal-container"),
    ])
    
    return layout

@callback(
    [Output("token-embed-plot", "figure"),
     Output("token-embed-error-output", "children"),
     Output("token-embed-result-card", "style"),
     Output("token-embed-loading-container", "children")],
    [Input("token-embed-visualize-button", "n_clicks")],
    [State("token-embed-input-text", "value"),
     State("token-embed-reduction-method", "value")],
    prevent_initial_call=True
)
def update_token_embedding_visualization(n_clicks, input_text, reduction_method):
    if not n_clicks:
        return go.Figure(), "", {"display": "none"}, []
    
    if not input_text or not input_text.strip():
        return go.Figure(), "Please enter some text to analyze.", {"display": "none"}, []

    try:
        # Check if model is loaded
        if not model_api.selected_model_path:
            return go.Figure(), "Error: No model selected. Please select a model from the main interface first.", {"display": "none"}, []
        
        embedding_result = model_api.get_sentence_embedding(input_text)

        if not embedding_result or "token_embeddings" not in embedding_result:
            return go.Figure(), "Error: Could not extract embeddings from the model.", {"display": "none"}, []

        token_embeddings = embedding_result["token_embeddings"]
        tokens = embedding_result.get("tokens", input_text.split())

        if token_embeddings is None or len(token_embeddings) == 0 or tokens is None or len(tokens) == 0:
            return go.Figure(), "Error: No token embeddings found.", {"display": "none"}, []

        # Filter out special tokens for cleaner visualization
        filtered_tokens = []
        filtered_embeddings = []
        for i, token in enumerate(tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and not token.startswith("["):
                filtered_tokens.append(token.replace("##", ""))  # Clean subword tokens
                filtered_embeddings.append(token_embeddings[i])
        
        if len(filtered_tokens) < 2:
            return go.Figure(), "Error: Need at least 2 tokens for visualization.", {"display": "none"}, []

        fig = create_embedding_plot(filtered_tokens, filtered_embeddings, method=reduction_method)
        return fig, "", {"display": "block"}, []
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if "meta tensor" in str(e).lower():
            error_msg = "Error: Model loading issue. Please try reloading the model or use a different model."
        return go.Figure(), error_msg, {"display": "none"}, []

# Callback for info toggle
@callback(
    Output("token-embed-info-collapse", "is_open"),
    Input("token-embed-info-toggle", "n_clicks"),
    State("token-embed-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_token_embed_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

"""
LIME explanation page layout and callbacks - Professional Design.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go

from components.visualizations import highlight_lime_text, create_lime_bar_chart
from models.api import model_api

def create_professional_highlighted_text(text, words, weights):
    """
    Create professional highlighted text for LIME results.
    
    Args:
        text: Text string
        words: List of words from LIME explanation
        weights: List of weights from LIME explanation
        
    Returns:
        HTML component with professionally highlighted text
    """
    # Create mapping of words to weights
    word_weights = {}
    for word, weight in zip(words, weights):
        if " " not in word:  # Only highlight single words
            word_weights[word.strip().lower()] = weight
    
    # Split text by spaces but keep spaces
    text_parts = []
    for word in text.split(" "):
        if word:
            text_parts.append(word)
            text_parts.append(" ")
    if text_parts and text_parts[-1] == " ":
        text_parts = text_parts[:-1]  # Remove trailing space
    
    # Create highlighted spans
    highlighted_text = []
    for word in text_parts:
        clean_word = word.strip().lower()
        if clean_word in word_weights:
            weight = word_weights[clean_word]
            if weight > 0:
                # Positive weight - green highlighting
                highlighted_text.append(html.Span(
                    word, 
                    className="lime-word-positive",
                    title=f"Positive impact: {weight:.3f}"
                ))
            else:
                # Negative weight - red highlighting
                highlighted_text.append(html.Span(
                    word, 
                    className="lime-word-negative",
                    title=f"Negative impact: {weight:.3f}"
                ))
        else:
            highlighted_text.append(word)
    
    return highlighted_text

def create_layout():
    """
    Create the professional layout for the LIME explanation page.
    
    Returns:
        A Dash layout object with custom styling
    """
    layout = html.Div([
        
        # Main LIME Container
        html.Div([
            # Header Section
            html.Div([
                html.H4([
                    html.I(className="fas fa-lightbulb", style={"marginRight": "0.5rem"}),
                    "LIME Explanation"
                ]),
                html.P("Understand which words contribute most to the sentiment prediction", 
                       className="lime-header-subtitle")
            ], className="lime-header"),
            
            # Input Section
            html.Div([
                html.Label("Enter text to analyze:", className="lime-input-label"),
                dcc.Textarea(
                    id="lime-input-text", 
                    value="This movie was fantastic! The acting was superb and the plot was engaging from start to finish.",
                    className="lime-textarea",
                    placeholder="Enter your text here..."
                ),
                
                # Settings Row
                html.Div([
                    html.Div([
                        html.Label("Features to explain:", className="lime-setting-label"),
                        dcc.Slider(
                            id="num-features-slider",
                            min=5,
                            max=20,
                            value=5,
                            marks={i: str(i) for i in [5, 10, 15, 20]},
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="lime-setting-group"),
                    
                    html.Div([
                        html.Label("LIME samples:", className="lime-setting-label"),
                        dcc.Slider(
                            id="num-samples-slider",
                            min=1000,
                            max=10000,
                            value=1000,
                            marks={i: f"{i//1000}k" for i in [1000, 2500, 5000, 7500, 10000]},
                            step=1000,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="lime-setting-group"),
                ], className="lime-settings-row"),
                
                # Generate Button
                dbc.Button([
                    html.I(className="fas fa-magic me-2"),
                    "Analyze Text"
                ], id="lime-submit-button", color="primary", size="lg", className="w-100"),
                
                # Error Output
                html.Div(id="lime-error-output", className="lime-error")
            ], className="lime-input-section"),
            
            # Results Section
            html.Div([
                html.Div(id="lime-loading-container", children=[], className="lime-loading"),
                html.Div(id="lime-result-container"),
                html.Div(id="lime-loading-placeholder")
            ], className="lime-results-container"),
            
            # Info Section
            html.Div([
                html.Button([
                    html.I(className="fas fa-question-circle", style={"marginRight": "0.5rem"}),
                    "Learn More About LIME"
                ], id="lime-info-toggle", className="lime-info-toggle"),
                
                dbc.Collapse([
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle", style={"marginRight": "0.5rem"}),
                            "About LIME Explanation"
                        ]),
                        html.P([
                            html.Strong("LIME (Local Interpretable Model-agnostic Explanations)"), 
                            " helps us understand which words in a text contribute most to the model's sentiment prediction."
                        ]),
                        html.Ul([
                            html.Li([html.Strong("Green highlighting:"), " Words that push toward positive sentiment"]),
                            html.Li([html.Strong("Red highlighting:"), " Words that push toward negative sentiment"]),
                            html.Li([html.Strong("Intensity:"), " Stronger colors indicate stronger influence"]),
                        ]),
                        html.P([
                            "LIME works by creating many variations of your text and seeing how the predictions change, ",
                            "allowing us to identify which words are most important for the model's decision."
                        ], className="lime-explanation")
                    ], className="lime-info-content")
                ], id="lime-info-collapse", is_open=False),
            ], className="lime-info-section"),
            
        ], className="lime-modal-container"),
    ])
    
    return layout

@callback(
    [Output("lime-result-container", "children"),
     Output("lime-error-output", "children"),
     Output("lime-loading-container", "children")],
    Input("lime-submit-button", "n_clicks"),
    [State("lime-input-text", "value"),
     State("num-features-slider", "value"),
     State("num-samples-slider", "value")],
    prevent_initial_call=True
)
def update_lime_explanation(n_clicks, input_text, num_features, num_samples):
    if not n_clicks:
        return None, "", []
    
    if not input_text or not input_text.strip():
        return None, "Please enter some text to analyze.", []
    
    try:
        # Check if model is loaded
        if not model_api.selected_model_path:
            return None, "Error: No model selected. Please select a model from the main interface first.", []
        
        explanation = model_api.explain_sentiment(input_text, num_features=num_features, num_samples=num_samples)
        
        # Extract data from LIME explanation
        if "original_prediction" in explanation:
            prediction = explanation["original_prediction"]
            prediction_label = prediction.get("label", "Unknown")
        else:
            prediction_label = "Unknown"
        
        # Get feature importance data
        feature_importance = explanation.get("feature_importance", [])
        words = [item[0] for item in feature_importance]
        weights = [item[1] for item in feature_importance]
        
        fig = create_lime_bar_chart(words, weights)
        
        highlighted_text = create_professional_highlighted_text(explanation["text"], words, weights)
        
        # Professional Text Card
        text_card = html.Div([
            html.Div("Text with Important Words Highlighted", className="lime-result-header"),
            html.Div([
                html.Div(highlighted_text, className="lime-highlighted-text"),
                html.P([
                    "Green highlighting indicates words that push toward positive sentiment. ",
                    "Red highlighting indicates words that push toward negative sentiment. ",
                    "Stronger highlighting indicates stronger influence."
                ], className="lime-explanation")
            ], className="lime-result-body")
        ], className="lime-result-card")
        
        # Professional Chart Card
        importance_card = html.Div([
            html.Div("Word Importance", className="lime-result-header"),
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=fig,
                        style={"width": "100%", "height": "350px"},
                        config={'responsive': True, 'displayModeBar': False}
                    )
                ], className="lime-chart-container"),
                html.P([
                    "Green bars indicate words that push the prediction toward positive sentiment, ",
                    "while red bars indicate words that push the prediction toward negative sentiment."
                ], className="lime-explanation")
            ], className="lime-result-body")
        ], className="lime-result-card")
        
        return [
            html.Div([
                text_card,
                importance_card,
            ]), 
            "",
            []
        ]
    except Exception as e:
        return None, f"Error: {str(e)}", []

# Callback for info toggle
@callback(
    Output("lime-info-collapse", "is_open"),
    Input("lime-info-toggle", "n_clicks"),
    State("lime-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_lime_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open
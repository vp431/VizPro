"""
Error Analysis page with custom UI design.
Inspired by sentiment_lime.py structure with custom CSS styling.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
import time

from models.api import model_api
from models.analysis_store import analysis_store
from models.error_analysis import categorize_error_patterns
from utils.dataset_scanner import scan_datasets, load_dataset_samples

logger = logging.getLogger(__name__)

def create_layout():
    """Create the Error Analysis layout with custom styling."""
    return html.Div([
        # Custom CSS for Error Analysis
        html.Style("""
            .error-analysis-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .error-analysis-header {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .error-analysis-title {
                color: #2c3e50;
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-align: center;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .error-analysis-subtitle {
                color: #6c757d;
                font-size: 1.1rem;
                text-align: center;
                margin-bottom: 0;
            }
            
            .error-control-panel {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .error-control-title {
                color: #2c3e50;
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .error-input-group {
                margin-bottom: 20px;
            }
            
            .error-input-label {
                color: #495057;
                font-weight: 600;
                margin-bottom: 8px;
                display: block;
                font-size: 0.95rem;
            }
            
            .error-input-field {
                width: 100%;
                padding: 12px 16px;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.9);
            }
            
            .error-input-field:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                outline: none;
                background: rgba(255, 255, 255, 1);
            }
            
            .error-analyze-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border: none;
                color: white;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                width: 100%;
            }
            
            .error-analyze-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                background: linear-gradient(45deg, #5a6fd8, #6a42a0);
            }
            
            .error-analyze-btn:active {
                transform: translateY(0);
            }
            
            .error-results-container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                min-height: 400px;
            }
            
            .error-results-title {
                color: #2c3e50;
                font-size: 1.6rem;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .error-stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }
            
            .error-stat-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 1px solid rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            
            .error-stat-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
            }
            
            .error-stat-value {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 5px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .error-stat-label {
                color: #6c757d;
                font-size: 0.9rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .error-chart-container {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .error-loading {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 60px 20px;
                color: #6c757d;
            }
            
            .error-loading-spinner {
                width: 50px;
                height: 50px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error-toggle-btn {
                background: linear-gradient(45deg, #28a745, #20c997);
                border: none;
                color: white;
                padding: 8px 20px;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 15px;
            }
            
            .error-toggle-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
            }
            
            .error-info-panel {
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #2196f3;
            }
            
            .error-info-title {
                color: #1976d2;
                font-weight: 600;
                margin-bottom: 10px;
                font-size: 1.1rem;
            }
            
            .error-info-text {
                color: #424242;
                line-height: 1.6;
                margin-bottom: 0;
            }
        """),
        
        # Main container
        html.Div([
            # Header section
            html.Div([
                html.H1("🔍 Error Analysis", className="error-analysis-title"),
                html.P("Comprehensive analysis of model prediction errors and patterns", 
                       className="error-analysis-subtitle")
            ], className="error-analysis-header"),
            
            # Control panel
            html.Div([
                html.H3([
                    html.I(className="fas fa-cogs", style={"color": "#667eea"}),
                    "Analysis Configuration"
                ], className="error-control-title"),
                
                # Info toggle
                html.Button([
                    html.I(className="fas fa-info-circle me-2"),
                    "Show Analysis Information"
                ], id="error-analysis-info-toggle", className="error-toggle-btn"),
                
                # Collapsible info panel
                dbc.Collapse([
                    html.Div([
                        html.H5("About Error Analysis", className="error-info-title"),
                        html.P([
                            "Error Analysis helps you understand where and why your model makes mistakes. ",
                            "It categorizes errors into patterns like negation handling, intensity modifiers, ",
                            "and contextual understanding issues. Use this tool to identify systematic ",
                            "weaknesses in your model's performance."
                        ], className="error-info-text")
                    ], className="error-info-panel")
                ], id="error-analysis-info-collapse", is_open=False),
                
                # Sample size input
                html.Div([
                    html.Label("Sample Size for Analysis", className="error-input-label"),
                    dcc.Input(
                        id="sample-size-input",
                        type="number",
                        value=200,
                        min=10,
                        max=1000,
                        step=10,
                        className="error-input-field",
                        placeholder="Enter number of samples to analyze..."
                    )
                ], className="error-input-group"),
                
                # Analyze button
                html.Button([
                    html.I(className="fas fa-play me-2"),
                    "Start Error Analysis"
                ], id="analyze-dataset-btn", className="error-analyze-btn")
                
            ], className="error-control-panel"),
            
            # Results section
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-line", style={"color": "#667eea"}),
                    "Analysis Results"
                ], className="error-results-title"),
                
                # Loading/Results content
                html.Div(id="error-analysis-content", children=[
                    html.Div([
                        html.I(className="fas fa-arrow-up", style={"fontSize": "3rem", "color": "#ccc", "marginBottom": "20px"}),
                        html.H4("Ready to Analyze", style={"color": "#6c757d", "marginBottom": "10px"}),
                        html.P("Configure your analysis settings above and click 'Start Error Analysis' to begin.", 
                               style={"color": "#6c757d", "textAlign": "center"})
                    ], className="error-loading")
                ])
                
            ], className="error-results-container")
            
        ], className="error-analysis-container")
    ])

@callback(
    Output("error-analysis-info-collapse", "is_open"),
    Input("error-analysis-info-toggle", "n_clicks"),
    State("error-analysis-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_error_analysis_info(n_clicks, is_open):
    """Toggle the error analysis information panel."""
    if n_clicks:
        return not is_open
    return is_open

@callback(
    Output("error-analysis-content", "children"),
    Input("current-analysis-store", "data"),
    State("dataset-dropdown", "value"),
    State("selected-model-store", "data"),
    prevent_initial_call=True
)
def update_error_analysis_content(analysis_data, selected_dataset, selected_model):
    """Update the error analysis content when analysis is completed."""
    if not analysis_data or not selected_dataset or not selected_model:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"fontSize": "3rem", "color": "#ffc107", "marginBottom": "20px"}),
                html.H4("No Analysis Data", style={"color": "#6c757d", "marginBottom": "10px"}),
                html.P("Please run an analysis first to see results.", 
                       style={"color": "#6c757d", "textAlign": "center"})
            ], className="error-loading")
        ])
    
    try:
        # Get analysis results
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, selected_model["model_path"])
        
        if not stored_analysis:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-search", style={"fontSize": "3rem", "color": "#17a2b8", "marginBottom": "20px"}),
                    html.H4("Loading Analysis...", style={"color": "#6c757d", "marginBottom": "10px"}),
                    html.P("Please wait while we process your analysis results.", 
                           style={"color": "#6c757d", "textAlign": "center"})
                ], className="error-loading")
            ])
        
        results = stored_analysis["results"]
        accuracy = results.get("accuracy", 0)
        total_samples = results.get("total_samples", 0)
        correct_predictions = results.get("correct_predictions", 0)
        high_conf_errors = results.get("high_conf_errors", [])
        error_patterns = results.get("error_patterns", {})
        
        # Create statistics cards
        stats_cards = html.Div([
            html.Div([
                html.Div(f"{accuracy:.1%}", className="error-stat-value"),
                html.Div("Accuracy", className="error-stat-label")
            ], className="error-stat-card"),
            
            html.Div([
                html.Div(f"{total_samples}", className="error-stat-value"),
                html.Div("Total Samples", className="error-stat-label")
            ], className="error-stat-card"),
            
            html.Div([
                html.Div(f"{correct_predictions}", className="error-stat-value"),
                html.Div("Correct Predictions", className="error-stat-label")
            ], className="error-stat-card"),
            
            html.Div([
                html.Div(f"{len(high_conf_errors)}", className="error-stat-value"),
                html.Div("High Conf. Errors", className="error-stat-label")
            ], className="error-stat-card")
        ], className="error-stats-grid")
        
        # Create error pattern visualization
        if error_patterns:
            pattern_names = list(error_patterns.keys())
            pattern_counts = [error_patterns[pattern]["count"] for pattern in pattern_names]
            
            fig = px.bar(
                x=pattern_counts,
                y=pattern_names,
                orientation='h',
                title="Error Pattern Distribution",
                labels={'x': 'Number of Errors', 'y': 'Error Pattern'},
                color=pattern_counts,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=16,
                title_x=0.5
            )
            
            pattern_chart = html.Div([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ], className="error-chart-container")
        else:
            pattern_chart = html.Div([
                html.P("No error patterns found in the analysis.", 
                       style={"textAlign": "center", "color": "#6c757d", "padding": "40px"})
            ], className="error-chart-container")
        
        # Create scatter plot for error distribution
        all_results = results.get("results", [])
        if all_results:
            df = pd.DataFrame(all_results)
            scatter_fig = px.scatter(
                df, x=df.index, y="confidence", color="correct",
                title="Prediction Confidence Distribution",
                labels={"x": "Sample Index", "confidence": "Confidence", "correct": "Correct"},
                color_discrete_map={True: "#28a745", False: "#dc3545"}
            )
            scatter_fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=16,
                title_x=0.5
            )
            
            scatter_chart = html.Div([
                dcc.Graph(figure=scatter_fig, config={'displayModeBar': False})
            ], className="error-chart-container")
        else:
            scatter_chart = html.Div()
        
        return html.Div([
            stats_cards,
            pattern_chart,
            scatter_chart
        ])
        
    except Exception as e:
        logger.error(f"Error updating error analysis content: {str(e)}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"fontSize": "3rem", "color": "#dc3545", "marginBottom": "20px"}),
                html.H4("Analysis Error", style={"color": "#dc3545", "marginBottom": "10px"}),
                html.P(f"Error loading analysis results: {str(e)}", 
                       style={"color": "#6c757d", "textAlign": "center"})
            ], className="error-loading")
        ])
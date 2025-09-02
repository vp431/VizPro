"""
Error Pattern Analysis page for analyzing systematic error patterns in model predictions.
Inspired by sentiment_lime.py structure.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

from models.api import model_api
from models.analysis_store import analysis_store
from models.error_analysis import categorize_error_patterns

logger = logging.getLogger(__name__)

def create_error_patterns_layout():
    """Create the main error patterns analysis layout."""
    return html.Div([
        # Header Section
        html.Div([
            html.H4([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Error Pattern Analysis"
            ], className="text-primary mb-2"),
            html.P("Systematic analysis of error patterns to identify common failure modes", 
                   className="text-muted mb-3")
        ], className="mb-4"),
        
        # Main Content Area
        html.Div(id="error-patterns-content", className="mb-4"),
        
        # Instructions Section
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle me-2"),
                            "Understanding Error Patterns"
                        ]),
                        html.P([
                            html.Strong("Pattern Categories: "),
                            "Errors are automatically categorized based on linguistic patterns and content analysis."
                        ]),
                        html.Ul([
                            html.Li([html.Strong("Negation Errors:"), " Issues with handling negation words"]),
                            html.Li([html.Strong("Intensity Errors:"), " Problems with intensity modifiers"]),
                            html.Li([html.Strong("Context Errors:"), " Difficulties with longer, complex sentences"]),
                            html.Li([html.Strong("Comparison Errors:"), " Issues with comparative statements"]),
                            html.Li([html.Strong("Sarcasm Errors:"), " Potential sarcasm or irony detection failures"]),
                            html.Li([html.Strong("Ambiguity Errors:"), " Problems with ambiguous sentiment words"]),
                        ]),
                        html.P([
                            "Use this analysis to understand systematic weaknesses and guide model improvements."
                        ])
                    ])
                ], title="Pattern Categories", item_id="patterns")
            ], start_collapsed=True, className="mt-3")
        ])
    ], className="error-patterns-container")

def create_error_patterns_display(error_patterns, high_conf_errors):
    """Create display for error pattern analysis results."""
    if not error_patterns or not high_conf_errors:
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-3x text-success mb-3"),
                        html.H5("No High Confidence Errors", className="text-success"),
                        html.P("Great! No high confidence errors were found in the analysis.", className="text-muted"),
                        html.P("This suggests the model is performing well on the analyzed samples.", className="text-muted")
                    ], className="text-center py-5")
                ])
            ], className="border-success")
        ])
    
    # Filter out categories with no errors
    filtered_patterns = {k: v for k, v in error_patterns.items() if v["count"] > 0}
    
    if not filtered_patterns:
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-info-circle fa-3x text-info mb-3"),
                        html.H5("No Categorized Patterns", className="text-info"),
                        html.P("High confidence errors were found but couldn't be categorized into known patterns.", className="text-muted"),
                        html.P(f"Total high confidence errors: {len(high_conf_errors)}", className="text-muted")
                    ], className="text-center py-5")
                ])
            ], className="border-info")
        ])
    
    # Create pattern distribution chart
    categories = list(filtered_patterns.keys())
    counts = [filtered_patterns[cat]["count"] for cat in categories]
    
    # Create bar chart
    fig = px.bar(
        x=categories,
        y=counts,
        labels={"x": "Error Category", "y": "Number of Errors"},
        title=f"Error Pattern Distribution ({sum(counts)} total errors)",
        color=counts,
        color_continuous_scale="Reds"
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    
    # Create enhanced pattern details cards
    pattern_cards = []
    pattern_colors = {
        "negation_errors": {"color": "danger", "icon": "fas fa-times-circle", "bg": "#dc3545"},
        "intensity_errors": {"color": "warning", "icon": "fas fa-exclamation-triangle", "bg": "#fd7e14"},
        "context_errors": {"color": "info", "icon": "fas fa-file-text", "bg": "#17a2b8"},
        "comparison_errors": {"color": "success", "icon": "fas fa-balance-scale", "bg": "#28a745"},
        "sarcasm_errors": {"color": "primary", "icon": "fas fa-smile-wink", "bg": "#007bff"},
        "ambiguity_errors": {"color": "secondary", "icon": "fas fa-question-circle", "bg": "#6c757d"}
    }
    
    for category, data in filtered_patterns.items():
        if data["count"] > 0:
            # Get pattern styling
            pattern_style = pattern_colors.get(category, {"color": "secondary", "icon": "fas fa-tag", "bg": "#6c757d"})
            
            # Get some example texts
            examples = data["examples"][:3]  # Show up to 3 examples
            
            example_items = []
            for i, example in enumerate(examples, 1):
                example_items.append(
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-quote-left me-2 text-muted"),
                            html.Strong(f"Example {i}: ", className="text-muted"),
                            html.Span(example["text"][:150] + "..." if len(example["text"]) > 150 else example["text"], 
                                     className="text-break")
                        ], className="mb-2"),
                        html.Small([
                            f"True: {'Positive' if str(example.get('true_label', '')) == '1' else 'Negative'} | ",
                            f"Predicted: {'Positive' if str(example.get('predicted_label', '')) == '1' else 'Negative'} | ",
                            f"Confidence: {example.get('confidence', 0):.2f}"
                        ], className="text-muted")
                    ], className="p-3 mb-2 bg-light rounded border-start border-3", 
                       style={"borderLeftColor": pattern_style["bg"]})
                )
            
            card = html.Div([
                # Enhanced header with gradient and icon
                html.Div([
                    html.H6([
                        html.I(className=f"{pattern_style['icon']} me-2"),
                        category.replace("_", " ").title()
                    ], className="text-white mb-1"),
                    html.Div([
                        html.Span(f"{data['count']} errors", className="badge bg-light text-dark")
                    ])
                ], className="p-3 d-flex justify-content-between align-items-center", style={
                    "background": f"linear-gradient(135deg, {pattern_style['bg']} 0%, {pattern_style['bg']}dd 100%)",
                    "borderRadius": "12px 12px 0 0"
                }),
                
                # Enhanced body with better formatting
                html.Div([
                    # Description
                    html.Div([
                        html.P(data["description"], className="text-muted mb-3 fst-italic")
                    ]),
                    
                    # Examples section
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-list-ul me-2 text-muted"),
                            "Example Cases:"
                        ], className="mb-3"),
                        html.Div(example_items) if example_items else html.P("No examples available", className="text-muted")
                    ])
                ], className="p-3", style={
                    "background": "white",
                    "borderRadius": "0 0 12px 12px"
                })
            ], className="mb-3 border rounded", style={
                "boxShadow": "0 4px 15px rgba(0,0,0,0.1)",
                "transition": "transform 0.2s ease",
                "cursor": "default"
            })
            
            pattern_cards.append(card)
    
    return html.Div([
        # Enhanced summary statistics
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(len(high_conf_errors), className="text-danger mb-1"),
                    html.P("High Confidence Errors", className="text-muted mb-0")
                ], className="text-center p-3 bg-light rounded border", style={
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
                })
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H4(len(filtered_patterns), className="text-warning mb-1"),
                    html.P("Error Categories", className="text-muted mb-0")
                ], className="text-center p-3 bg-light rounded border", style={
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
                })
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H4(f"{(sum(filtered_patterns[cat]['count'] for cat in filtered_patterns)/len(high_conf_errors)*100):.1f}%", className="text-info mb-1"),
                    html.P("Categorized", className="text-muted mb-0")
                ], className="text-center p-3 bg-light rounded border", style={
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
                })
            ], width=4)
        ], className="mb-4"),
        
        # Enhanced pattern distribution chart
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Error Pattern Distribution"
                ], className="text-white mb-0")
            ], className="p-3", style={
                "background": "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
                "borderRadius": "12px 12px 0 0"
            }),
            html.Div([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ], className="p-3", style={
                "background": "white",
                "borderRadius": "0 0 12px 12px"
            })
        ], className="mb-4 border rounded", style={
            "boxShadow": "0 4px 15px rgba(0,0,0,0.1)",
            "overflow": "hidden"
        }),
        
        # Enhanced pattern details section
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-list me-2"),
                    "Pattern Details"
                ], className="text-white mb-0")
            ], className="p-3 mb-3", style={
                "background": "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
                "borderRadius": "12px"
            }),
            html.Div(pattern_cards, style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                "gap": "1.5rem"
            })
        ])
    ])

def create_no_analysis_message():
    """Create message when no analysis has been run."""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle fa-3x text-muted mb-3"),
                    html.H4("Click 'Analyze Dataset' First", className="text-muted mb-3"),
                    html.P("Error pattern analysis will be available after dataset analysis.", className="text-muted"),
                    html.P("This analysis identifies systematic error patterns to help improve model performance.", className="text-muted")
                ], className="text-center py-5")
            ])
        ], className="border-secondary")
    ])

# Callback to populate error patterns analysis
@callback(
    Output("error-patterns-content", "children"),
    [Input("error-analysis-modal", "is_open"),
     Input("current-analysis-store", "data")],
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def populate_error_patterns_analysis(modal_is_open, analysis_store_data, selected_dataset, selected_model):
    """Populate the error patterns analysis with stored data."""
    if not modal_is_open or not selected_dataset or not selected_model:
        return html.Div()
    
    try:
        # Get stored analysis results
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, selected_model["model_path"])
        
        if not stored_analysis:
            return create_no_analysis_message()
        
        # Get the results from stored analysis
        analysis_results = stored_analysis.get("results", {})
        error_patterns = analysis_results.get("error_patterns", {})
        high_conf_errors = analysis_results.get("high_conf_errors", [])
        
        return create_error_patterns_display(error_patterns, high_conf_errors)
        
    except Exception as e:
        logger.error(f"Error creating error patterns analysis: {str(e)}")
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("Error Loading Analysis: "),
                f"An error occurred: {str(e)}"
            ], color="danger")
        ])

def handle_error_patterns_analysis(button_id, selected_dataset, selected_model):
    """Handle error patterns analysis for model level."""
    if not selected_dataset:
        return "Input Required", html.Div("Please select a dataset to analyze.", className="text-muted text-center py-3")
    
    if button_id == "feature-btn-error_patterns":
        title = "Error Pattern Analysis"
        
        # Check if we have stored analysis results - use model_path from selected_model
        model_path = selected_model.get("model_path") if selected_model else None
        if not model_path:
            return title, create_no_analysis_message()
            
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, model_path)
        
        # If not found, try loading from backup file
        if not stored_analysis:
            print("DEBUG Error Patterns: Trying to load from backup file...")
            try:
                import json
                import os
                backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
                with open(backup_file_path, "r") as f:
                    backup_data = json.load(f)
                    if (backup_data.get("dataset") == selected_dataset and 
                        backup_data.get("model_path") == model_path):
                        print("DEBUG Error Patterns: Found matching backup data!")
                        # Extract the results properly - backup_data already contains the results structure
                        results = backup_data.get("results", {})
                        
                        # Check if we have high_conf_errors in the results
                        if "results" in results and isinstance(results["results"], list):
                            # Process the results to extract high confidence errors
                            high_conf_errors = []
                            for item in results["results"]:
                                if (item.get("confidence", 0) > 0.8 and 
                                    item.get("correct", True) == False):
                                    high_conf_errors.append(item)
                            
                            # Categorize error patterns
                            if high_conf_errors:
                                error_patterns = categorize_error_patterns(high_conf_errors)
                                results["error_patterns"] = error_patterns
                                results["high_conf_errors"] = high_conf_errors
                            
                        stored_analysis = {"results": results}
            except Exception as e:
                print(f"DEBUG Error Patterns: Error loading backup: {e}")
        
        if not stored_analysis:
            content = create_no_analysis_message()
        else:
            # Get the analysis results
            analysis_results = stored_analysis.get("results", {})
            error_patterns = analysis_results.get("error_patterns", {})
            high_conf_errors = analysis_results.get("high_conf_errors", [])
            sample_size_used = analysis_results.get("sample_size_used", 0)
            
            content = html.Div([
                # Header with stats
                html.Div([
                    html.H4([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Error Pattern Analysis Results"
                    ], className="text-primary mb-2"),
                    html.P(f"Analysis of error patterns from {sample_size_used} samples", 
                           className="text-muted mb-3")
                ], className="mb-4"),
                
                # Main content
                html.Div([
                    create_error_patterns_display(error_patterns, high_conf_errors)
                ], id="error-patterns-content"),
                
                # Instructions
                html.Div([
                    dbc.Accordion([
                        dbc.AccordionItem([
                            html.Div([
                                html.H6([
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Understanding Error Patterns"
                                ]),
                                html.P([
                                    html.Strong("Pattern Categories: "),
                                    "Errors are automatically categorized based on linguistic patterns and content analysis."
                                ]),
                                html.Ul([
                                    html.Li([html.Strong("Negation Errors:"), " Issues with handling negation words"]),
                                    html.Li([html.Strong("Intensity Errors:"), " Problems with intensity modifiers"]),
                                    html.Li([html.Strong("Context Errors:"), " Difficulties with longer, complex sentences"]),
                                    html.Li([html.Strong("Comparison Errors:"), " Issues with comparative statements"]),
                                    html.Li([html.Strong("Sarcasm Errors:"), " Potential sarcasm or irony detection failures"]),
                                    html.Li([html.Strong("Ambiguity Errors:"), " Problems with ambiguous sentiment words"]),
                                ]),
                                html.P([
                                    "Use this analysis to understand systematic weaknesses and guide model improvements."
                                ])
                            ])
                        ], title="Pattern Categories", item_id="patterns")
                    ], start_collapsed=True, className="mt-3")
                ])
            ])
        
        return title, content
    
    return f"Feature: {button_id}", html.Div(f"Feature {button_id} not implemented yet.", className="text-muted text-center py-3")
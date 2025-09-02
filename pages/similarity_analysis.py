"""
Similarity Analysis page for finding similar examples to selected data points.
Inspired by sentiment_lime.py structure.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go

from models.api import model_api
from models.analysis_store import analysis_store
from models.similarity_analysis import find_similar_examples

logger = logging.getLogger(__name__)

def create_similarity_analysis_layout():
    """Create the main similarity analysis layout."""
    return html.Div([
        # Header Section
        html.Div([
            html.H4([
                html.I(className="fas fa-search me-2"),
                "Similarity Analysis"
            ], className="text-primary mb-2"),
            html.P("Find similar examples in the dataset to understand model behavior patterns", 
                   className="text-muted mb-3")
        ], className="mb-4"),
        
        # Main Content Area
        html.Div(id="similarity-analysis-content", className="mb-4"),
        
        # Instructions Section
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle me-2"),
                            "How to Use Similarity Analysis"
                        ]),
                        html.P([
                            html.Strong("Finding Similar Examples: "),
                            "Click on any point in the Error Analysis scatter plot to find similar examples in the dataset."
                        ]),
                        html.Ul([
                            html.Li([html.Strong("Similarity Score:"), " Higher scores indicate more similar text content"]),
                            html.Li([html.Strong("Text Comparison:"), " Compare the selected text with similar examples"]),
                            html.Li([html.Strong("Pattern Analysis:"), " Identify common patterns in similar texts"]),
                        ]),
                        html.P([
                            "This helps understand why the model made certain predictions and identify systematic patterns."
                        ])
                    ])
                ], title="Instructions", item_id="instructions")
            ], start_collapsed=True, className="mt-3")
        ])
    ], className="similarity-analysis-container")

def create_similarity_results_display(text, similar_examples):
    """Create display for similarity analysis results."""
    if not similar_examples:
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-info-circle fa-2x text-info mb-3"),
                        html.H5("No Similar Examples Found", className="text-info"),
                        html.P("No similar examples found in the dataset.", className="text-muted"),
                        html.Small([
                            "Try adjusting the similarity threshold or analyzing a different example.",
                            html.Br(),
                            f"Current text: '{text[:100]}...'"
                        ], className="text-muted")
                    ], className="text-center py-4")
                ])
            ], className="border-info")
        ])
    
    # Create results table
    rows = []
    for i, example in enumerate(similar_examples, 1):
        rows.append(
            html.Tr([
                html.Td(i, className="text-center fw-bold"),
                html.Td(example["text"], className="text-break"),
                html.Td([
                    dbc.Badge(f"{example['similarity']:.3f}", 
                             color="primary" if example['similarity'] > 0.8 else "secondary",
                             className="fs-6")
                ], className="text-center")
            ])
        )
    
    return html.Div([
        # Original text display
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-file-text me-2"),
                html.Strong("Selected Text")
            ]),
            dbc.CardBody([
                html.P(text, className="mb-0 text-break")
            ])
        ], className="mb-4"),
        
        # Similar examples table
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-list me-2"),
                html.Strong(f"Similar Examples ({len(similar_examples)} found)")
            ]),
            dbc.CardBody([
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("#", className="text-center", style={"width": "10%"}),
                            html.Th("Similar Text", style={"width": "70%"}),
                            html.Th("Similarity Score", className="text-center", style={"width": "20%"})
                        ])
                    ]),
                    html.Tbody(rows)
                ], bordered=True, hover=True, responsive=True, striped=True)
            ])
        ])
    ])

def create_no_analysis_message():
    """Create message when no analysis has been run."""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-search fa-3x text-muted mb-3"),
                    html.H4("Click 'Analyze Dataset' First", className="text-muted mb-3"),
                    html.P("Similarity analysis will be available after dataset analysis.", className="text-muted"),
                    html.P("Once analysis is complete, click on any point in the Error Analysis scatter plot to find similar examples.", className="text-muted")
                ], className="text-center py-5")
            ])
        ], className="border-secondary")
    ])

def create_no_point_selected_message():
    """Create message when no point is selected."""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-mouse-pointer fa-3x text-info mb-3"),
                    html.H4("Select a Data Point", className="text-info mb-3"),
                    html.P("Click on any point in the Error Analysis scatter plot to find similar examples.", className="text-muted"),
                    html.P("This will help you understand patterns in the model's predictions.", className="text-muted")
                ], className="text-center py-5")
            ])
        ], className="border-info")
    ])

# Callback to handle similarity analysis after point click
@callback(
    Output("similarity-analysis-content", "children"),
    [Input("selected-error-store", "data")],
    [State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def update_similarity_analysis_display(point_data, selected_dataset, selected_model):
    """Find similar examples to the selected point and create display."""
    if not point_data:
        return create_no_point_selected_message()
    
    try:
        text = point_data["text"]
        
        # Check if we have stored analysis results
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, selected_model["model_path"])
        if not stored_analysis:
            return create_no_analysis_message()
        
        # Use the similarity analysis function
        similar_examples = find_similar_examples(text, selected_dataset)
        
        return create_similarity_results_display(text, similar_examples)
            
    except Exception as e:
        logger.exception("Error in similarity analysis")
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("Error in Similarity Analysis: "),
                f"An error occurred: {str(e)}. Please try again with a different example."
            ], color="danger")
        ])

# Callback to analyze error similarity using clustering
@callback(
    Output("similarity-analysis-results", "children"),
    Input("analyze-similarity-button", "n_clicks"),
    [State("num-clusters-input", "value"),
     State("dataset-dropdown", "value"),
     State("selected-model-store", "data")],
    prevent_initial_call=True
)
def analyze_error_similarity(n_clicks, num_clusters, selected_dataset, selected_model):
    """Analyze similarity patterns in errors using clustering from b2.py logic"""
    if not n_clicks or not selected_dataset or not selected_model:
        return html.Div()
    
    print(f"USER CLICKED: 'Analyze Error Similarity' button - Clustering errors into {num_clusters} clusters")
    
    try:
        # Get stored analysis results
        model_path = selected_model.get("model_path")
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, model_path)
        
        # If not found, try loading from backup file
        if not stored_analysis:
            try:
                import json
                import os
                backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
                with open(backup_file_path, "r") as f:
                    backup_data = json.load(f)
                    if (backup_data.get("dataset") == selected_dataset and 
                        backup_data.get("model_path") == model_path):
                        stored_analysis = {"results": backup_data["results"]}
            except Exception as e:
                print(f"Error loading backup: {e}")
        
        if not stored_analysis:
            return html.Div("No analysis data found. Please run dataset analysis first.", className="text-warning")
        
        # Extract high confidence errors from the results
        results = stored_analysis.get("results", {})
        
        # Check if we have the results list
        if "results" in results and isinstance(results["results"], list):
            # Extract high confidence errors
            high_conf_errors = []
            for item in results["results"]:
                if (item.get("confidence", 0) > 0.8 and 
                    item.get("correct", True) == False):
                    high_conf_errors.append(item)
        else:
            high_conf_errors = results.get("high_conf_errors", [])
        
        if not high_conf_errors:
            return html.Div("No high confidence errors found for clustering analysis.", className="text-info")
        
        if len(high_conf_errors) < num_clusters:
            return html.Div(f"Not enough error examples ({len(high_conf_errors)}) for {num_clusters} clusters.", className="text-warning")
        
        # Import required libraries for clustering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        
        error_sentences = [error["text"] for error in high_conf_errors]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(error_sentences)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Create t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(error_sentences)-1))
        tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
        
        # Create DataFrame for visualization
        df_viz = pd.DataFrame({
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'cluster': cluster_labels,
            'sentence': [s[:50] + "..." for s in error_sentences]
        })
        
        # Create scatter plot
        fig = px.scatter(
            df_viz, x='x', y='y', color='cluster',
            hover_data=['sentence'],
            title="Error Clustering Visualization (t-SNE)",
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
        )
        
        # Create cluster summaries
        cluster_summaries = []
        for cluster_id in range(num_clusters):
            cluster_sentences = [error_sentences[i] for i in range(len(error_sentences)) if cluster_labels[i] == cluster_id]
            
            # Get most common words in this cluster
            cluster_vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
            cluster_tfidf = cluster_vectorizer.fit_transform(cluster_sentences)
            feature_names = cluster_vectorizer.get_feature_names_out()
            
            cluster_summaries.append({
                'id': cluster_id,
                'size': len(cluster_sentences),
                'top_words': list(feature_names),
                'examples': cluster_sentences[:3]  # First 3 examples
            })
        
        # Create enhanced cluster summary cards
        summary_cards = []
        colors = ["primary", "success", "info", "warning", "danger", "secondary", "dark", "light"]
        for summary in cluster_summaries:
            color = colors[summary['id'] % len(colors)]
            card = html.Div([
                # Card header with color coding
                html.Div([
                    html.H6([
                        html.I(className="fas fa-layer-group me-2"),
                        f"Cluster {summary['id']}"
                    ], className="text-white mb-1"),
                    html.Small(f"{summary['size']} errors", className="text-white-50")
                ], className="p-3", style={
                    "background": f"var(--bs-{color})" if color != "light" else "#6c757d",
                    "borderRadius": "8px 8px 0 0"
                }),
                
                # Card body with better formatting
                html.Div([
                    # Top words section
                    html.Div([
                        html.Strong("Key Terms: ", className="text-muted"),
                        html.Div([
                            html.Span(word, className=f"badge bg-{color} me-1 mb-1") 
                            for word in summary['top_words']
                        ])
                    ], className="mb-3"),
                    
                    # Examples section
                    html.Div([
                        html.Strong("Example Errors:", className="text-muted mb-2"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-quote-left me-2 text-muted"),
                                html.Span(ex, className="text-break")
                            ], className="p-2 mb-2 bg-light rounded border-start border-3", 
                               style={"borderLeftColor": f"var(--bs-{color})" if color != "light" else "#6c757d"})
                            for ex in summary['examples']
                        ])
                    ])
                ], className="p-3", style={
                    "background": "white",
                    "borderRadius": "0 0 8px 8px"
                })
            ], className="border rounded", style={
                "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                "transition": "transform 0.2s ease",
                "cursor": "default"
            })
            summary_cards.append(card)
        
        result = html.Div([
            # Enhanced header with gradient
            html.Div([
                html.H4([
                    html.I(className="fas fa-project-diagram me-2"),
                    "Similarity-Based Error Analysis"
                ], className="text-white mb-2"),
                html.P(f"Clustered {len(high_conf_errors)} high-confidence errors into {num_clusters} groups", 
                       className="text-white-50 mb-0")
            ], className="p-3 mb-0", style={
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "borderRadius": "12px 12px 0 0"
            }),
            
            # Content area with better styling
            html.Div([
                # Visualization section
                html.Div([
                    html.H5([
                        html.I(className="fas fa-chart-scatter me-2"),
                        "Cluster Visualization"
                    ], className="text-primary mb-3"),
                    dcc.Graph(figure=fig, style={"height": "500px"})
                ], className="mb-4"),
                
                html.Hr(className="my-4"),
                
                # Cluster summaries section
                html.Div([
                    html.H5([
                        html.I(className="fas fa-layer-group me-2"),
                        "Cluster Details"
                    ], className="text-primary mb-3"),
                    html.Div(summary_cards, style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(350px, 1fr))",
                        "gap": "1rem"
                    })
                ])
            ], className="p-4", style={
                "background": "rgba(255, 255, 255, 0.98)",
                "borderRadius": "0 0 12px 12px"
            })
        ], style={
            "borderRadius": "12px",
            "overflow": "hidden",
            "boxShadow": "0 8px 25px rgba(0,0,0,0.15)",
            "border": "1px solid #e9ecef"
        })
        
        print(f"Similarity analysis complete. Found {num_clusters} clusters")
        return result
        
    except Exception as e:
        print(f"Error in similarity analysis: {str(e)}")
        return html.Div(f"Error analyzing similarity: {str(e)}", className="text-danger")

def handle_similarity_analysis(button_id, selected_dataset, selected_model):
    """Handle similarity analysis for model level."""
    if not selected_dataset:
        return "Input Required", html.Div("Please select a dataset to analyze.", className="text-muted text-center py-3")
    
    if button_id == "feature-btn-similarity_analysis":
        title = "Similarity Analysis"
        
        # Check if we have stored analysis results - use model_path from selected_model
        model_path = selected_model.get("model_path") if selected_model else None
        if not model_path:
            return title, create_no_analysis_message()
            
        stored_analysis = analysis_store.get_dataset_analysis(selected_dataset, model_path)
        
        # If not found, try loading from backup file
        if not stored_analysis:
            print("DEBUG Similarity Analysis: Trying to load from backup file...")
            try:
                import json
                import os
                backup_file_path = os.path.join("TempFiles", "analysis_backup.json")
                with open(backup_file_path, "r") as f:
                    backup_data = json.load(f)
                    if (backup_data.get("dataset") == selected_dataset and 
                        backup_data.get("model_path") == model_path):
                        print("DEBUG Similarity Analysis: Found matching backup data!")
                        stored_analysis = {"results": backup_data["results"]}
            except Exception as e:
                print(f"DEBUG Similarity Analysis: Error loading backup: {e}")
        
        if stored_analysis:
            # Add similarity analysis controls and results
            content = html.Div([
                # Header Section
                html.Div([
                    html.H4([
                        html.I(className="fas fa-search me-2"),
                        "Similarity Analysis"
                    ], className="text-primary mb-2"),
                    html.P("Analyze similarity patterns in errors using clustering and visualization.", 
                           className="text-muted mb-3")
                ], className="mb-4"),
                
                # Controls Section
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-cogs me-2"),
                        html.Strong("Analysis Controls")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Number of Clusters:", className="form-label"),
                                dbc.Input(
                                    id="num-clusters-input",
                                    type="number",
                                    value=3,
                                    min=2,
                                    max=10,
                                    step=1,
                                    className="mb-2"
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("", className="form-label"),
                                dbc.Button(
                                    [html.I(className="fas fa-play me-2"), "Analyze Error Similarity"],
                                    id="analyze-similarity-button",
                                    color="primary",
                                    className="w-100"
                                )
                            ], width=8)
                        ])
                    ])
                ], className="mb-4"),
                
                # Results area
                html.Div(id="similarity-analysis-results", className="mb-4"),
                
                # Instructions
                html.Div([
                    dbc.Accordion([
                        dbc.AccordionItem([
                            html.Div([
                                html.H6([
                                    html.I(className="fas fa-info-circle me-2"),
                                    "How to Use Similarity Analysis"
                                ]),
                                html.P([
                                    html.Strong("Clustering Analysis: "),
                                    "This analysis groups similar error examples together using TF-IDF vectorization and K-means clustering."
                                ]),
                                html.Ul([
                                    html.Li([html.Strong("t-SNE Visualization:"), " 2D visualization of error clusters"]),
                                    html.Li([html.Strong("Cluster Summaries:"), " Top words and example sentences for each cluster"]),
                                    html.Li([html.Strong("Pattern Identification:"), " Identify common error patterns across similar texts"]),
                                ]),
                                html.P([
                                    "Use this to understand systematic error patterns and improve model performance."
                                ])
                            ])
                        ], title="Instructions", item_id="instructions")
                    ], start_collapsed=True, className="mt-3")
                ])
            ])
        else:
            content = create_no_analysis_message()
        
        return title, content
    
    return f"Feature: {button_id}", html.Div(f"Feature {button_id} not implemented yet.", className="text-muted text-center py-3")
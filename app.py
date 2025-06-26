# === Standard Library ===
import os
import re
import io
import uuid
import logging
from collections import Counter
from io import BytesIO

# === Third-Party Libraries ===
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless environments like Azure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# === NLP / ML Libraries ===
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import lime
import lime.lime_text
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# === App-specific Imports ===
from model_utils import BERTAttentionVisualizer

# === Logging Setup ===
logging.basicConfig(level=logging.DEBUG)

# === Dash App Initialization ===
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    update_title=None
)
server = app.server  # Required for Azure deployment

# === Lazy Model Loader ===
visualizer = None
model_info = None

def get_visualizer():
    global visualizer, model_info
    if visualizer is None:
        try:
            logging.info("🔄 Loading BERTAttentionVisualizer and TinyBERT model...")
            visualizer = BERTAttentionVisualizer()
            model_info = visualizer.get_model_info()
            logging.info("✅ Model and visualizer loaded.")
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            # Provide fallback model info for UI rendering even if model fails to load
            visualizer = None
            model_info = {
                "model_name": "Model loading failed",
                "num_layers": 4,
                "num_heads": 12,
                "hidden_size": 312,
                "device": "cpu"
            }
            logging.info("⚠️ Using fallback model info.")
    return visualizer, model_info

# Cache for attention results to avoid recalculating
attention_cache = {}

# === UI ===
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Sentiment Analysis", href="/sentiment-analysis")),
        dbc.NavItem(dbc.NavLink("NER", href="/named-entity-recognition")),
        dbc.NavItem(dbc.NavLink("xAI (LIME)", href="/lime-explanation")),
        dbc.NavItem(dbc.NavLink("Error Analysis", href="/error-analysis")),
    ],
    brand="Transformers Visualization Tool",
    brand_href="/",
    color="primary",
    dark=True,
)

# Call the lazy loader before using model_info
visualizer, model_info = get_visualizer()

# Now this will work
print(f"Model: {model_info['model_name']}")


# Main layout with navbar and content
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id="page-content", children=[])
])

# Home page layout
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Attention Visualization", className="text-center my-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("Model Information", className="alert-heading"),
                html.P([
                    f"Model: {model_info['model_name']}",
                    html.Br(),
                    f"Layers: {model_info['num_layers']}",
                    html.Br(),
                    f"Attention Heads: {model_info['num_heads']}",
                    html.Br(),
                    f"Hidden Size: {model_info['hidden_size']}",
                    html.Br(),
                    f"Device: {model_info['device']}"
                ])
            ], color="info", className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Input Sentence", className="card-title"),
                    dcc.Textarea(
                        id="input-text",
                        value="IIT Delhi is a Top Engineering Institute in India",
                        style={"width": "100%", "height": 100},
                    ),
                    dbc.Button("Visualize Attention", id="submit-button", color="primary", className="mt-3"),
                    html.Div(id="error-output", className="text-danger mt-2")
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Attention Maps", className="mb-3"),
            html.Div([
                dbc.Label("Layer:"),
                dcc.Slider(
                    id="layer-slider",
                    min=0,
                    max=model_info['num_layers'] - 1,
                    value=0,
                    marks={i: str(i) for i in range(model_info['num_layers'])},
                    step=1
                ),
            ], className="mb-3"),
            html.Div([
                dbc.Label("Attention Head:"),
                dcc.Slider(
                    id="head-slider",
                    min=0,
                    max=model_info['num_heads'] - 1,
                    value=0,
                    marks={i: str(i) for i in range(model_info['num_heads'])},
                    step=1
                ),
            ], className="mb-3"),
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=[dcc.Graph(id="attention-heatmap", config={'responsive': True})],
            ),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="tokenized-output", className="mt-4")
        ])
    ]),
    
    # Store component to keep track of the current data
    dcc.Store(id='attention-data')
], fluid=True)


# Sentiment Analysis page
sentiment_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Sentiment Analysis Examples", className="text-center my-4")
        ])
    ]),
    
    html.Div(id="sentiment-content", className="mt-4"),
    
    # Store for dataset examples
    dcc.Store(id='sentiment-examples-data'),
    
    # Store for current example attention data
    dcc.Store(id='sentiment-attention-data')
], fluid=True)


# Named Entity Recognition page
ner_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Named Entity Recognition Examples", className="text-center my-4")
        ])
    ]),
    
    html.Div(id="ner-content", className="mt-4"),
    
    # Store for dataset examples
    dcc.Store(id='ner-examples-data'),
    
    # Store for current example attention data
    dcc.Store(id='ner-attention-data')
], fluid=True)


# LIME Explanation page
lime_explanation_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("xAI with LIME", className="text-center my-4"),
            html.P([
                "LIME helps us understand which words in a text ",
                "contribute most to the model's sentiment prediction."
            ], className="lead text-center")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Input Text for Explanation", className="card-title"),
                    dcc.Textarea(
                        id="lime-input-text",
                        value="This movie was fantastic! The acting was superb and the plot was engaging from start to finish.",
                        style={"width": "100%", "height": 100},
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of features to explain:"),
                            dcc.Slider(
                                id="num-features-slider",
                                min=5,
                                max=20,
                                value=10,
                                marks={i: str(i) for i in [5, 10, 15, 20]},
                                step=1
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Number of samples:"),
                            dcc.Slider(
                                id="num-samples-slider",
                                min=500,
                                max=2000,
                                value=1000,
                                marks={i: str(i) for i in [500, 1000, 1500, 2000]},
                                step=500
                            ),
                        ], width=6),
                    ], className="mt-3"),
                    dbc.Button("Explain Prediction", id="lime-submit-button", color="primary", className="mt-3"),
                    html.Div(id="lime-error-output", className="text-danger mt-2")
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="lime-result-container", className="mt-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Try Examples from IMDb Dataset", className="mb-3 mt-4"),
            html.Div(id="lime-examples-container")
        ])
    ]),
    
    # Store for LIME explanation data
    dcc.Store(id='lime-explanation-data'),
    
    # Store for IMDb examples
    dcc.Store(id='imdb-examples-data')
], fluid=True)


# Error Analysis page layout
error_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Model Error Analysis", className="text-center my-4"),
            html.P([
                "This page analyzes where the sentiment model fails on the SST-2 dataset. ",
                "It helps identify patterns in sentences where the model makes incorrect predictions."
            ], className="lead text-center")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Dataset Analysis Settings", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of samples to analyze:"),
                            dcc.Slider(
                                id="num-samples-slider-error",
                                min=100,
                                max=500,
                                step=100,
                                value=200,
                                marks={i: str(i) for i in range(100, 501, 100)},
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Confidence threshold:"),
                            dcc.Slider(
                                id="confidence-threshold-slider",
                                min=0.5,
                                max=0.95,
                                step=0.05,
                                value=0.7,
                                marks={i/100: str(i/100) for i in range(50, 96, 5)},
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    dbc.Button("Analyze Model Errors", id="analyze-errors-button", color="primary", className="mt-3"),
                    html.Div(id="analysis-loading-indicator", className="mt-2")
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="error-analysis-results", className="mt-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Error Analysis Tools"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.H5("LIME Analysis of Error Cases", className="mt-3"),
                            html.P([
                                "Select an error example to analyze why the model made a mistake on this specific sentence. ",
                                "LIME will show which words influenced the incorrect prediction."
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select an error example:"),
                                    dcc.Dropdown(
                                        id="error-example-dropdown",
                                        options=[],
                                        placeholder="First run error analysis above"
                                    ),
                                ], width=8),
                                dbc.Col([
                                    dbc.Button(
                                        "Analyze with LIME", 
                                        id="analyze-error-with-lime-button", 
                                        color="info", 
                                        className="mt-4"
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            html.Div(id="lime-error-analysis-result")
                        ], label="LIME Analysis"),
                        
                        dbc.Tab([
                            html.H5("Counterfactual Testing", className="mt-3"),
                            html.P([
                                "Test how small changes to a misclassified sentence affect the model's prediction. ",
                                "This helps identify which parts of the sentence are causing the model to make errors."
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select an error example:"),
                                    dcc.Dropdown(
                                        id="counterfactual-example-dropdown",
                                        options=[],
                                        placeholder="First run error analysis above"
                                    ),
                                ], width=8),
                                dbc.Col([
                                    dbc.Button(
                                        "Test Counterfactuals", 
                                        id="test-counterfactuals-button", 
                                        color="info", 
                                        className="mt-4"
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            html.Div(id="counterfactual-results")
                        ], label="Counterfactual Testing"),
                        
                        dbc.Tab([
                            html.H5("Similarity-Based Error Analysis", className="mt-3"),
                            html.P([
                                "Find patterns in error cases using TF-IDF similarity and clustering. ",
                                "This helps identify groups of similar errors that might share common causes."
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Number of clusters:"),
                                    dbc.Input(
                                        id="num-clusters-input",
                                        type="number",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=5
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Button(
                                        "Analyze Error Similarity", 
                                        id="analyze-similarity-button", 
                                        color="info", 
                                        className="mt-4"
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            html.Div(id="similarity-analysis-results")
                        ], label="Similarity Analysis")
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Store for error analysis data
    dcc.Store(id='error-analysis-data')
], fluid=True)


# Callback for page routing
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/sentiment-analysis":
        return sentiment_analysis_layout
    elif pathname == "/named-entity-recognition":
        return ner_layout
    elif pathname == "/lime-explanation":
        return lime_explanation_layout
    elif pathname == "/error-analysis":
        return error_analysis_layout
    else:
        return home_layout


# Callback to load sentiment examples
@app.callback(
    Output('sentiment-examples-data', 'data'),
    Input('url', 'pathname')
)
def load_sentiment_examples(pathname):
    if pathname != "/sentiment-analysis":
        return {}
    
    try:
        # Load SST-2 dataset
        dataset = load_dataset("nyu-mll/glue", "sst2")
        
        # Get examples from validation set
        val_set = dataset['validation']
        
        # Get predictions for a subset of examples
        examples = []
        correct_count = 0
        incorrect_count = 0
        
        # Process examples until we have enough correct and incorrect ones
        for i, ex in enumerate(val_set):
            if i >= 100:  # Limit to first 100 examples to avoid long processing
                break
                
            # Skip empty sentences
            if not ex["sentence"].strip():
                continue
            
            # Get sentiment prediction
            try:
                result = visualizer.predict_sentiment(ex["sentence"])
                prediction = result["label"]
                score = result["score"]
                
                # Check if prediction is correct
                is_correct = prediction == ex["label"]
                
                # Add example if we need more of this type
                if (is_correct and correct_count < 5) or (not is_correct and incorrect_count < 5):
                    examples.append({
                        "sentence": ex["sentence"],
                        "true_label": ex["label"],
                        "pred_label": prediction,
                        "score": score,
                        "correct": is_correct
                    })
                    
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                            
                # Stop if we have enough examples
                if correct_count >= 5 and incorrect_count >= 5:
                    break
                    
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                continue
        
        return {"task": "sentiment", "examples": examples}
        
    except Exception as e:
        print(f"Error loading sentiment dataset: {str(e)}")
        return {"task": "sentiment", "examples": [], "error": str(e)}


# Callback to load NER examples
@app.callback(
    Output('ner-examples-data', 'data'),
    Input('url', 'pathname')
)
def load_ner_examples(pathname):
    if pathname != "/named-entity-recognition":
        return {}
    
    try:
        # Load CoNLL-2003 dataset with trust_remote_code=True
        dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
        
        # Get examples from test set
        test_set = dataset['test']
        
        # Map CoNLL-2003 numeric labels to BIO tags
        # Based on the dataset documentation: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
        ner_tag_map = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG", 
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC"
        }
        
        examples = []
        correct_count = 0
        incorrect_count = 0
        
        # Process examples until we have enough correct and incorrect ones
        for i, ex in enumerate(test_set):
            if i >= 50:  # Limit to first 50 examples to avoid long processing
                break
            
            # Join tokens to form a sentence
            sentence = " ".join(ex["tokens"])
            
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            # Get NER prediction
            try:
                result = visualizer.predict_ner(sentence)
                pred_entities = result["entity_labels"]
                
                # Convert dataset numeric NER tags to BIO tags for comparison
                true_entities = [ner_tag_map.get(tag, "O") for tag in ex["ner_tags"]]
                
                # For simplicity, consider an example correct if more than 70% of tokens match
                total_tokens = len(ex["tokens"])
                matching_count = 0
                
                # Count matching tokens
                for j in range(min(len(true_entities), len(pred_entities))):
                    if pred_entities[j] == true_entities[j]:
                        matching_count += 1
                
                accuracy = matching_count / total_tokens if total_tokens > 0 else 0
                is_correct = accuracy > 0.7
                
                # Add example if we need more of this type
                if (is_correct and correct_count < 5) or (not is_correct and incorrect_count < 5):
                    examples.append({
                        "sentence": sentence,
                        "tokens": ex["tokens"],
                        "true_entities": true_entities,
                        "pred_entities": pred_entities[:len(ex["tokens"])],  # Trim to match token length
                        "accuracy": accuracy,
                        "correct": is_correct
                    })
                    
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                        
                # Stop if we have enough examples
                if correct_count >= 5 and incorrect_count >= 5:
                    break
                    
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                continue
        
        return {"task": "ner", "examples": examples}
        
    except Exception as e:
        print(f"Error loading NER dataset: {str(e)}")
        return {"task": "ner", "examples": [], "error": str(e)}


# Callback to render sentiment content
@app.callback(
    Output('sentiment-content', 'children'),
    Input('sentiment-examples-data', 'data')
)
def render_sentiment_content(data):
    if not data or not data.get('examples'):
        return html.Div("Loading sentiment examples...")
    
    # Split examples into correct and incorrect predictions
    correct_examples = [ex for ex in data['examples'] if ex.get('correct', False)]
    incorrect_examples = [ex for ex in data['examples'] if not ex.get('correct', False)]
    
    # Create example cards for each group
    correct_cards = dbc.Row([
        dbc.Col(create_example_card(example, i, "sentiment", True), width=12, md=6, lg=4)
        for i, example in enumerate(correct_examples)
    ])
    
    incorrect_cards = dbc.Row([
        dbc.Col(create_example_card(example, i, "sentiment", False), width=12, md=6, lg=4)
        for i, example in enumerate(incorrect_examples)
    ])
    
    # Add pattern analysis button
    pattern_analysis_section = html.Div([
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    html.Div(id="pattern-analysis-content")
                )
            ),
            id="pattern-analysis-collapse",
            is_open=False
        )
    ], className="mb-4")
    
    return html.Div([
        pattern_analysis_section,
        html.H2("Correct Predictions", className="mb-3"),
        correct_cards,
        html.H2("Incorrect Predictions", className="mt-5 mb-3"),
        incorrect_cards
    ])


# Callback to render NER content
@app.callback(
    Output('ner-content', 'children'),
    Input('ner-examples-data', 'data')
)
def render_ner_content(data):
    if not data or "examples" not in data or not data["examples"]:
        return html.Div("Loading NER examples... This may take a minute.", className="text-center my-5")
    
    examples = data["examples"]
    
    # Create tabs for correct and incorrect examples
    correct_examples = [ex for ex in examples if ex.get("correct", False)]
    incorrect_examples = [ex for ex in examples if not ex.get("correct", False)]
    
    # Create slides for examples
    slides = []
    
    # Add correct examples
    slides.append(html.H4("Correct Examples", className="mt-4"))
    for i, ex in enumerate(correct_examples):
        slides.append(create_example_card(ex, i, "ner", True))
    
    # Add incorrect examples
    slides.append(html.H4("Incorrect Examples", className="mt-4"))
    for i, ex in enumerate(incorrect_examples):
        slides.append(create_example_card(ex, i, "ner", False))
    
    return html.Div(slides)

def create_example_card(example, index, task, is_correct):
    """Create a card for a dataset example"""
    card_id = f"{'correct' if is_correct else 'incorrect'}-{index}-{task}"
    
    if task == "sentiment":
        true_label = "Positive" if example.get("true_label") == 1 else "Negative"
        pred_label = "Positive" if example.get("pred_label") == 1 else "Negative"
        score = example.get("score", 0)
        
        card_content = [
            html.H5(f"Example {index+1}", className="card-title"),
            html.P(f"Sentence: {example.get('sentence')}", className="card-text"),
            html.P([
                html.Strong("True label: "), 
                html.Span(true_label, className="badge bg-primary")
            ], className="card-text"),
            html.P([
                html.Strong("Predicted label: "), 
                html.Span(pred_label, className=f"badge {'bg-success' if example.get('correct') else 'bg-danger'}")
            ], className="card-text"),
            html.P(f"Confidence: {score:.4f}", className="card-text"),
            dbc.Button(
                "Visualize Attention", 
                id={"type": "example-button", "index": card_id},
                color="primary",
                className="mt-2"
            ),
            # Modal for visualization
            dbc.Modal(
                [
                    dbc.ModalHeader(f"Attention Visualization - Example {index+1}"),
                    dbc.ModalBody(id={"type": "example-viz", "index": card_id}),
                    dbc.ModalFooter(
                        dbc.Button("Close", id={"type": "close-modal", "index": card_id}, className="ms-auto")
                    ),
                ],
                id={"type": "viz-modal", "index": card_id},
                size="xl",
                is_open=False,
            ),
        ]
    else:  # NER
        # Create a table to show token-level predictions
        tokens = example.get("tokens", [])
        true_entities = example.get("true_entities", [])
        pred_entities = example.get("pred_entities", [])
        accuracy = example.get("accuracy", 0)
        
        # Create a table for token-level predictions
        table_header = [
            html.Thead(html.Tr([
                html.Th("Token"),
                html.Th("True Entity"),
                html.Th("Predicted Entity"),
                html.Th("Match")
            ]))
        ]
        
        table_rows = []
        for i in range(min(len(tokens), 10)):  # Show first 10 tokens
            # Determine if this prediction is correct
            is_token_correct = (i < len(true_entities) and i < len(pred_entities) and 
                               (true_entities[i] == pred_entities[i]))
            
            # Display true entity
            true_entity_display = true_entities[i] if i < len(true_entities) else ""
            
            # Display predicted entity
            pred_entity_display = pred_entities[i] if i < len(pred_entities) else ""
            
            # Check if prediction matches
            is_match = true_entity_display == pred_entity_display
            
            row = html.Tr([
                html.Td(tokens[i]),
                html.Td(true_entity_display),
                html.Td(
                    html.Span(
                        pred_entity_display,
                        className=f"{'text-success' if is_match else 'text-danger'}"
                    )
                ),
                html.Td(
                    html.Span(
                        "✓" if is_match else "✗",
                        className=f"{'text-success' if is_match else 'text-danger'}"
                    )
                )
            ])
            table_rows.append(row)
        
        table_body = [html.Tbody(table_rows)]
        
        card_content = [
            html.H5(f"Example {index+1}", className="card-title"),
            html.P(f"Sentence: {example.get('sentence')}", className="card-text"),
            html.P(f"Accuracy: {accuracy:.2f}", className="card-text"),
            dbc.Table(table_header + table_body, bordered=True, hover=True, size="sm")
        ]
    
    return dbc.Card(dbc.CardBody(card_content), className="mb-3")


def create_attention_heatmap_from_result(tokens, attentions, layer_idx, head_idx):
    """Create a heatmap figure from tokens and attention data"""
    # Handle different attention formats
    if isinstance(attentions, list) and len(attentions) > layer_idx:
        # Format: [layers, batch, heads, seq_len, seq_len]
        attention_map = attentions[layer_idx][0, head_idx]
    else:
        # Fallback or different format
        print(f"Warning: Unexpected attention format. Shape: {np.array(attentions).shape}")
        attention_map = np.zeros((len(tokens), len(tokens)))
    
    # Create a more visually appealing heatmap similar to the image
    fig = go.Figure()
    
    # Use a dark background
    fig.update_layout(
        template="plotly_dark",
        title=f"Layer {layer_idx}, Head {head_idx}",
        height=600,
        width=800,
        margin=dict(l=60, r=60, t=50, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        plot_bgcolor="#000000"
    )
    
    # Add token labels on both sides
    token_positions = list(range(len(tokens)))
    for i, token in enumerate(tokens):
        # Left side token labels
        fig.add_annotation(
            x=-0.05, 
            y=i,
            text=token,
            showarrow=False,
            font=dict(color="white", size=12),
            xanchor="right",
            xref="paper",
            yref="y"
        )
        
        # Right side token labels
        fig.add_annotation(
            x=1.05, 
            y=i,
            text=token,
            showarrow=False,
            font=dict(color="white", size=12),
            xanchor="left",
            xref="paper",
            yref="y"
        )
    
    # Draw attention lines
    max_weight = np.max(attention_map)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            # Get the attention weight from token i to token j
            # In attention maps, rows are query tokens and columns are key tokens
            weight = attention_map[i, j]
            
            # Normalize weight for better visualization
            norm_weight = weight / max_weight if max_weight > 0 else 0
            
            if norm_weight > 0.1:  # Only draw significant attention weights
                # Use a color gradient from dark to bright orange based on attention weight
                color_intensity = min(1.0, norm_weight * 1.5)  # Scale up for visibility
                color = f"rgba(255, {int(165 * color_intensity)}, 0, {color_intensity})"
                
                fig.add_shape(
                    type="line",
                    x0=0, y0=i,
                    x1=1, y1=j,
                    line=dict(color=color, width=1 + 4 * norm_weight),
                    xref="paper",
                    yref="y",
                    layer="below"
                )
    
    # Set axis ranges
    fig.update_xaxes(range=[-0.2, 1.2])
    fig.update_yaxes(range=[-0.5, len(tokens) - 0.5])
    
    return fig


def create_attention_heatmap(attention_result, layer_idx, head_idx):
    """Create a heatmap figure from attention data"""
    tokens = attention_result["tokens"]
    attentions = attention_result["attentions"]
    
    return create_attention_heatmap_from_result(tokens, attentions, layer_idx, head_idx)


@app.callback(
    Output('attention-data', 'data'),
    Input("submit-button", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True
)
def process_text_input(n_clicks, input_text):
    """Process the input text and store the attention data"""
    if input_text is None or input_text.strip() == "":
        input_text = "The quick brown fox jumps over the lazy dog."
    
    # Use cache if available to avoid recomputing
    if input_text in attention_cache:
        return attention_cache[input_text]
        
    try:
        # Get attention data
        result = visualizer.get_attention(input_text)
        
        # Store in cache and return
        attention_cache[input_text] = {
            'tokens': result['tokens'],
            'input_text': input_text
        }
        return attention_cache[input_text]
    except Exception as e:
        return {
            'error': str(e),
            'input_text': input_text
        }


# Callback to update visualization
@app.callback(
    [Output("attention-heatmap", "figure"),
     Output("tokenized-output", "children"),
     Output("error-output", "children")],
    [Input('attention-data', 'data'),
     Input("layer-slider", "value"),
     Input("head-slider", "value")]
)
def update_visualization(data, layer_idx, head_idx):
    """Update the visualization based on the current data and slider values"""
    if data is None:
        # Initial state - create an empty placeholder figure
        fig = go.Figure()
        fig.update_layout(
            title="Enter a sentence and click 'Visualize Attention'",
            height=500
        )
        return fig, "", ""
        
    if 'error' in data:
        # Handle error case
        fig = go.Figure()
        fig.update_layout(
            title="Error occurred",
            annotations=[
                dict(
                    text="Error processing input",
                    showarrow=False,
                    font=dict(size=20)
                )
            ],
            height=500
        )
        return fig, "", f"Error: {data['error']}"
    
    try:
        # Get the tokens from stored data
        tokens = data['tokens']
        input_text = data['input_text']
        
        # Get attention map for the selected layer and head
        result = visualizer.get_attention_map(input_text, layer_idx, head_idx)
        attention_map = result["attention_map"]
        
        # Create heatmap with improved performance settings
        fig = go.Figure(data=go.Heatmap(
            z=attention_map,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight"),
            hoverongaps=False,
            zmin=0, zmax=np.max(attention_map)
        ))
        
        fig.update_layout(
            title=f"Attention Map - Layer {layer_idx}, Head {head_idx}",
            xaxis=dict(title="Tokens (Target)"),
            yaxis=dict(title="Tokens (Source)"),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            uirevision='constant'  # Preserve zoom on updates
        )
        
        # Display tokenized output
        token_output = html.Div([
            html.H5("Tokenized Input:"),
            html.P(" ".join(tokens))
        ])
        
        return fig, token_output, ""
    except Exception as e:
        # Capture detailed error information
        error_msg = f"Error: {str(e)}"
        traceback_str = traceback.format_exc()
        print(f"Error in callback: {error_msg}\n{traceback_str}")
        
        # Create error figure
        fig = go.Figure()
        fig.update_layout(
            title="Error in visualization",
            annotations=[
                dict(
                    text="An error occurred while generating the visualization",
                    showarrow=False,
                    font=dict(size=16)
                )
            ],
            height=500
        )
        
        return fig, "", error_msg


# Callback to toggle the visualization modal
@app.callback(
    Output({"type": "viz-modal", "index": dash.MATCH}, "is_open"),
    [Input({"type": "example-button", "index": dash.MATCH}, "n_clicks"),
     Input({"type": "close-modal", "index": dash.MATCH}, "n_clicks")],
    [State({"type": "viz-modal", "index": dash.MATCH}, "is_open"),
     State({"type": "example-button", "index": dash.MATCH}, "id")],
    prevent_initial_call=True
)
def toggle_viz_modal(open_clicks, close_clicks, is_open, button_id):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return is_open
    
    # Get the ID of the component that triggered the callback
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if "example-button" in trigger_id:
        # Log to terminal when button is clicked
        index_parts = button_id["index"].split("-")
        is_correct = index_parts[0] == "correct"
        index = index_parts[1]
        task = index_parts[2]
        print(f"\n[USER ACTION] Clicked visualization button for {task} example {index} (correct: {is_correct})")
        return True
    elif "close-modal" in trigger_id:
        return False
    
    return is_open


def calculate_attention_attribution(tokens, attentions, layer_idx, head_idx, label):
    """
    Calculate attention attribution scores that show how each token influences the sentiment prediction.
    
    Args:
        tokens: List of tokens from the model tokenizer
        attentions: Attention weights from the model
        layer_idx: Index of the layer to analyze
        head_idx: Index of the attention head to analyze
        label: Predicted sentiment label (0=negative, 1=positive)
    
    Returns:
        Dictionary containing:
            - tokens: List of tokens
            - attribution_scores: List of attribution scores for each token
            - positive_tokens: List of tokens with strong positive contribution
            - negative_tokens: List of tokens with strong negative contribution
    """
    # Get attention map for specified layer and head
    attention_map = attentions[layer_idx][0, head_idx]
    
    # Calculate aggregated attention for each token (sum of attention received)
    # This shows which tokens other tokens are attending to
    token_importance = attention_map.sum(axis=0)
    
    # Normalize scores to range [0, 1]
    if token_importance.max() > 0:
        token_importance = token_importance / token_importance.max()
    
    # Combine tokens with their importance scores, excluding special tokens
    token_scores = []
    for i, (token, score) in enumerate(zip(tokens, token_importance)):
        # Skip special tokens [CLS], [SEP], etc.
        if token in ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '<cls>', '<sep>']:
            continue
        token_scores.append((token, score))
    
    # Sort by importance score (descending)
    token_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Identify most influential tokens (top 30%)
    threshold = 0.3
    top_tokens = token_scores[:max(1, int(len(token_scores) * threshold))]
    
    # Classify as positive or negative contribution based on sentiment label
    # For positive sentiment, high attention tokens are positive contributors
    # For negative sentiment, high attention tokens are negative contributors
    if label == 1:  # Positive sentiment
        positive_tokens = [t[0] for t in top_tokens]
        negative_tokens = []
    else:  # Negative sentiment
        positive_tokens = []
        negative_tokens = [t[0] for t in top_tokens]
    
    return {
        "tokens": tokens,
        "attribution_scores": token_importance,
        "positive_tokens": positive_tokens,
        "negative_tokens": negative_tokens,
        "token_scores": token_scores
    }


# Function to create a visualization of attention attribution
def create_attention_attribution_visualization(attribution_results):
    """Create visualization for attention attribution analysis"""
    tokens = attribution_results["tokens"]
    scores = attribution_results["attribution_scores"]
    positive_tokens = set(attribution_results["positive_tokens"])
    negative_tokens = set(attribution_results["negative_tokens"])
    
    # Create bar chart for token attribution
    fig = go.Figure()
    
    # Skip special tokens and prepare data for visualization
    display_tokens = []
    display_scores = []
    colors = []
    hover_texts = []
    
    # Filter out special tokens for display
    special_tokens = ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '<cls>', '<sep>']
    
    for i, token in enumerate(tokens):
        # Skip special tokens
        if token in special_tokens:
            continue
            
        display_tokens.append(token)
        display_scores.append(scores[i])
        
        # Determine color based on sentiment contribution
        if token in positive_tokens:
            colors.append('rgba(0, 255, 0, 0.7)')  # Green for positive
            sentiment = "Positive"
        elif token in negative_tokens:
            colors.append('rgba(255, 0, 0, 0.7)')  # Red for negative
            sentiment = "Negative"
        else:
            colors.append('rgba(180, 180, 180, 0.5)')  # Gray for neutral
            sentiment = "Neutral"
        
        hover_texts.append(f"Token: {token}<br>Importance: {scores[i]:.4f}<br>Contribution: {sentiment}")
    
    # Add trace only if we have tokens to display
    if display_tokens:
        fig.add_trace(go.Bar(
            x=list(range(len(display_tokens))),
            y=display_scores,
            marker_color=colors,
            text=display_tokens,
            hovertext=hover_texts,
            hoverinfo="text"
        ))
        
        # Update layout with token labels
        fig.update_layout(
            title="Token Importance for Sentiment Prediction",
            xaxis=dict(
                title="Tokens",
                tickmode='array',
                tickvals=list(range(len(display_tokens))),
                ticktext=display_tokens,
                tickangle=45
            ),
            yaxis=dict(title="Attention Attribution Score"),
            height=400,
            margin=dict(l=50, r=50, t=80, b=120)
        )
    else:
        # Empty figure with message if no tokens to display
        fig.update_layout(
            title="No valid tokens to display",
            height=400
        )
    
    return fig


# Add a callback to update the most influential tokens list when layer/head changes
@app.callback(
    Output({"type": "influential-tokens", "index": dash.MATCH}, "children"),
    [Input({"type": "example-layer-slider", "index": dash.MATCH}, "value"),
     Input({"type": "example-head-slider", "index": dash.MATCH}, "value")],
    State({"type": "example-layer-slider", "index": dash.MATCH}, "id"),
    prevent_initial_call=True
)
def update_influential_tokens(layer_idx, head_idx, slider_id):
    """Update the most influential tokens list when layer or head changes"""
    example_id = slider_id["index"]
    
    # Get the cached attention data
    if not hasattr(app, 'example_attention_cache') or example_id not in app.example_attention_cache:
        # Return empty list if no data
        return []
    
    result = app.example_attention_cache[example_id]
    tokens = result["tokens"]
    attentions = result["attentions"]
    label = result["label"]
    
    # Calculate new attribution with updated layer/head
    attribution_results = calculate_attention_attribution(
        tokens, attentions, layer_idx, head_idx, label
    )
    
    # Filter out special tokens
    special_tokens = ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '<cls>', '<sep>']
    filtered_top_tokens = [(token, score) for token, score in attribution_results["token_scores"][:5] 
                          if token not in special_tokens]
    
    # Create updated token list
    token_list = [
        html.Li(f"{token}: {score:.4f}", 
               style={"color": "green" if token in attribution_results["positive_tokens"] 
                      else "red" if token in attribution_results["negative_tokens"] 
                      else "gray"})
        for token, score in filtered_top_tokens
    ]
    
    return token_list


# Modify the visualize_sentiment_example callback to make the token list update dynamically
@app.callback(
    Output({"type": "example-viz", "index": dash.MATCH}, "children"),
    Input({"type": "viz-modal", "index": dash.MATCH}, "is_open"),
    State({"type": "example-button", "index": dash.MATCH}, "id"),
    State("sentiment-examples-data", "data"),
    prevent_initial_call=True
)
def visualize_sentiment_example(is_open, button_id, sentiment_data):
    if not is_open or not sentiment_data:
        return []
    
    # Extract example index from button ID
    index_parts = button_id["index"].split("-")
    is_correct = index_parts[0] == "correct"
    index = int(index_parts[1])
    task = index_parts[2]
    
    # Only handle sentiment examples
    if task != "sentiment":
        return []
    
    # Get the example
    examples = [ex for ex in sentiment_data["examples"] if ex.get("correct", False) == is_correct]
    if index >= len(examples):
        return html.Div("Example not found")
    
    example = examples[index]
    sentence = example.get("sentence", "")
    
    print(f"[VISUALIZATION] Generating attention visualization for sentence: \"{sentence}\"")
    
    try:
        # Use sentiment model to get attention
        result = visualizer.predict_sentiment(sentence)
        tokens = result["tokens"]
        attentions = result["attentions"]
        label = result["label"]
        
        # Create visualization components
        viz_components = []
        
        # Add prediction info
        sentiment_label = "Positive" if label == 1 else "Negative"
        viz_components.append(
            dbc.Alert(
                f"Prediction: {sentiment_label} (confidence: {result['score']:.4f})",
                color="success" if label == 1 else "danger",
                className="mb-3"
            )
        )
        
        # Layer and head sliders
        viz_components.extend([
            html.Div([
                dbc.Label("Layer:"),
                dcc.Slider(
                    id={"type": "example-layer-slider", "index": button_id["index"]},
                    min=0,
                    max=len(attentions) - 1,
                    value=1,  # Default to layer 1 as in the image
                    marks={i: str(i) for i in range(len(attentions))},
                    step=1
                ),
            ], className="mb-3"),
            html.Div([
                dbc.Label("Attention Head:"),
                dcc.Slider(
                    id={"type": "example-head-slider", "index": button_id["index"]},
                    min=0,
                    max=attentions[0].shape[1] - 1,  # Number of heads in the first layer
                    value=2,  # Default to head 2 as in the image
                    marks={i: str(i) for i in range(attentions[0].shape[1])},
                    step=1
                ),
            ], className="mb-3"),
        ])
        
        # Attention heatmap
        viz_components.append(
            dcc.Graph(
                id={"type": "example-heatmap", "index": button_id["index"]},
                figure=create_attention_heatmap_from_result(tokens, attentions, 1, 2),  # Layer 1, Head 2 as in the image
                config={'responsive': True}
            )
        )
        
        # Calculate and display attention attribution
        default_layer = 1
        default_head = 2
        attribution_results = calculate_attention_attribution(
            tokens, attentions, default_layer, default_head, label
        )
        
        # Add attribution visualization without the explanatory text
        viz_components.append(
            dcc.Graph(
                id={"type": "example-attribution", "index": button_id["index"]},
                figure=create_attention_attribution_visualization(attribution_results),
                config={'responsive': True}
            )
        )
        
        # Add token importance analysis, excluding special tokens
        special_tokens = ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '<cls>', '<sep>']
        filtered_top_tokens = [(token, score) for token, score in attribution_results["token_scores"][:5] 
                              if token not in special_tokens]
        
        token_list = [
            html.Li(f"{token}: {score:.4f}", 
                   style={"color": "green" if token in attribution_results["positive_tokens"] 
                          else "red" if token in attribution_results["negative_tokens"] 
                          else "gray"})
            for token, score in filtered_top_tokens
        ]
        
        # Add heading with current layer/head info
        viz_components.append(
            html.Div([
                html.H6(f"Most Influential Tokens (Layer {default_layer}, Head {default_head}):", 
                       id={"type": "token-heading", "index": button_id["index"]}),
                html.Ul(token_list, id={"type": "influential-tokens", "index": button_id["index"]})
            ], className="mt-3 mb-4")
        )
        
        # Tokenized output
        viz_components.append(
            html.Div([
                html.H5("Tokenized Input:"),
                html.P(" ".join(tokens))
            ])
        )
        
        # Store attention data for this example
        app.example_attention_cache = app.example_attention_cache if hasattr(app, 'example_attention_cache') else {}
        app.example_attention_cache[button_id["index"]] = {
            "tokens": tokens,
            "attentions": attentions,
            "label": label
        }
        
        print(f"[VISUALIZATION] Successfully generated attention visualization with {len(tokens)} tokens")
        return viz_components
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"[ERROR] Error visualizing example: {str(e)}\n{traceback_str}")
        return html.Div(f"Error visualizing attention: {str(e)}")


# Add a callback to update the token heading when layer/head changes
@app.callback(
    Output({"type": "token-heading", "index": dash.MATCH}, "children"),
    [Input({"type": "example-layer-slider", "index": dash.MATCH}, "value"),
     Input({"type": "example-head-slider", "index": dash.MATCH}, "value")],
    prevent_initial_call=True
)
def update_token_heading(layer_idx, head_idx):
    """Update the token heading to show current layer and head"""
    return f"Most Influential Tokens (Layer {layer_idx}, Head {head_idx}):"


# Add a callback to update attribution visualization when layer/head changes
@app.callback(
    Output({"type": "example-attribution", "index": dash.MATCH}, "figure"),
    [Input({"type": "example-layer-slider", "index": dash.MATCH}, "value"),
     Input({"type": "example-head-slider", "index": dash.MATCH}, "value")],
    State({"type": "example-layer-slider", "index": dash.MATCH}, "id"),
    prevent_initial_call=True
)
def update_example_attribution(layer_idx, head_idx, slider_id):
    """Update attribution visualization when layer or head changes"""
    example_id = slider_id["index"]
    print(f"[DEBUG] Updating attribution visualization for layer {layer_idx}, head {head_idx}")
    
    # Get the cached attention data
    if not hasattr(app, 'example_attention_cache') or example_id not in app.example_attention_cache:
        # Return empty figure if no data
        print(f"[DEBUG] No cached data found for {example_id}")
        return go.Figure()
    
    result = app.example_attention_cache[example_id]
    tokens = result["tokens"]
    attentions = result["attentions"]
    label = result["label"]
    
    # Verify the requested layer/head is valid
    if layer_idx >= len(attentions) or head_idx >= attentions[0].shape[1]:
        print(f"[DEBUG] Invalid layer/head: {layer_idx}/{head_idx}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Invalid layer {layer_idx} or head {head_idx}",
            height=400
        )
        return fig
    
    # Calculate new attribution with updated layer/head
    attribution_results = calculate_attention_attribution(
        tokens, attentions, layer_idx, head_idx, label
    )
    
    # Create updated visualization
    figure = create_attention_attribution_visualization(attribution_results)
    print(f"[DEBUG] Successfully updated attribution visualization")
    return figure


# Restore the callback to update the attention heatmap when sliders change
@app.callback(
    Output({"type": "example-heatmap", "index": dash.MATCH}, "figure"),
    [Input({"type": "example-layer-slider", "index": dash.MATCH}, "value"),
     Input({"type": "example-head-slider", "index": dash.MATCH}, "value")],
    State({"type": "example-layer-slider", "index": dash.MATCH}, "id"),
    prevent_initial_call=True
)
def update_example_heatmap(layer_idx, head_idx, slider_id):
    """Update the attention heatmap when layer or head changes"""
    example_id = slider_id["index"]
    
    # Get the cached attention data for this example
    if not hasattr(app, 'example_attention_cache') or example_id not in app.example_attention_cache:
        # Return empty figure if no data
        return go.Figure()
    
    result = app.example_attention_cache[example_id]
    return create_attention_heatmap_from_result(result["tokens"], result["attentions"], layer_idx, head_idx)


def analyze_attention_patterns(sentiment_data, layer_idx=1, head_idx=2):
    """
    Analyze attention patterns across multiple examples to identify common patterns.
    
    Args:
        sentiment_data: Dictionary containing sentiment examples
        layer_idx: Index of the layer to analyze
        head_idx: Index of the attention head to analyze
        
    Returns:
        Dictionary containing pattern analysis results
    """
    if not sentiment_data or "examples" not in sentiment_data:
        return None
    
    positive_examples = [ex for ex in sentiment_data["examples"] if ex.get("label") == 1]
    negative_examples = [ex for ex in sentiment_data["examples"] if ex.get("label") == 0]
    
    print(f"[ANALYSIS] Analyzing attention patterns for {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    
    # Initialize storage for token frequency
    positive_token_scores = {}
    negative_token_scores = {}
    
    # List of special tokens to exclude
    special_tokens = ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '<cls>', '<sep>']
    
    # Analyze positive examples
    for example in positive_examples[:10]:  # Limit to 10 examples for performance
        sentence = example.get("sentence", "")
        try:
            # Get attention for this example
            result = visualizer.predict_sentiment(sentence)
            tokens = result["tokens"]
            attentions = result["attentions"]
            
            # Calculate attribution
            attribution = calculate_attention_attribution(tokens, attentions, layer_idx, head_idx, 1)
            
            # Add top tokens to frequency dict, excluding special tokens
            for token, score in attribution["token_scores"][:3]:  # Top 3 tokens
                if token in special_tokens:
                    continue
                    
                if token not in positive_token_scores:
                    positive_token_scores[token] = {"count": 0, "score": 0}
                positive_token_scores[token]["count"] += 1
                positive_token_scores[token]["score"] += score
        except Exception as e:
            print(f"Error analyzing positive example: {str(e)}")
    
    # Analyze negative examples
    for example in negative_examples[:10]:  # Limit to 10 examples for performance
        sentence = example.get("sentence", "")
        try:
            # Get attention for this example
            result = visualizer.predict_sentiment(sentence)
            tokens = result["tokens"]
            attentions = result["attentions"]
            
            # Calculate attribution
            attribution = calculate_attention_attribution(tokens, attentions, layer_idx, head_idx, 0)
            
            # Add top tokens to frequency dict, excluding special tokens
            for token, score in attribution["token_scores"][:3]:  # Top 3 tokens
                if token in special_tokens:
                    continue
                    
                if token not in negative_token_scores:
                    negative_token_scores[token] = {"count": 0, "score": 0}
                negative_token_scores[token]["count"] += 1
                negative_token_scores[token]["score"] += score
        except Exception as e:
            print(f"Error analyzing negative example: {str(e)}")
    
    # Calculate average scores
    for token in positive_token_scores:
        positive_token_scores[token]["avg_score"] = positive_token_scores[token]["score"] / positive_token_scores[token]["count"]
    
    for token in negative_token_scores:
        negative_token_scores[token]["avg_score"] = negative_token_scores[token]["score"] / negative_token_scores[token]["count"]
    
    # Sort by frequency and then score
    sorted_positive = sorted(
        positive_token_scores.items(),
        key=lambda x: (x[1]["count"], x[1]["avg_score"]),
        reverse=True
    )
    
    sorted_negative = sorted(
        negative_token_scores.items(),
        key=lambda x: (x[1]["count"], x[1]["avg_score"]),
        reverse=True
    )
    
    return {
        "positive_patterns": sorted_positive[:10],  # Top 10 patterns
        "negative_patterns": sorted_negative[:10],
        "layer_idx": layer_idx,
        "head_idx": head_idx
    }

def create_pattern_analysis_visualization(pattern_analysis):
    """Create visualization for attention pattern analysis"""
    if not pattern_analysis:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No pattern analysis data available",
            height=400
        )
        return fig
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Positive Sentiment Patterns", "Negative Sentiment Patterns"),
        shared_yaxes=True
    )
    
    # Add positive patterns
    positive_tokens = [item[0] for item in pattern_analysis["positive_patterns"]]
    positive_counts = [item[1]["count"] for item in pattern_analysis["positive_patterns"]]
    positive_scores = [item[1]["avg_score"] for item in pattern_analysis["positive_patterns"]]
    
    fig.add_trace(
        go.Bar(
            x=positive_tokens,
            y=positive_counts,
            name="Frequency",
            marker_color="rgba(0, 255, 0, 0.7)",
            hovertext=[f"Token: {token}<br>Count: {count}<br>Avg Score: {score:.4f}" 
                       for token, count, score in zip(positive_tokens, positive_counts, positive_scores)]
        ),
        row=1, col=1
    )
    
    # Add negative patterns
    negative_tokens = [item[0] for item in pattern_analysis["negative_patterns"]]
    negative_counts = [item[1]["count"] for item in pattern_analysis["negative_patterns"]]
    negative_scores = [item[1]["avg_score"] for item in pattern_analysis["negative_patterns"]]
    
    fig.add_trace(
        go.Bar(
            x=negative_tokens,
            y=negative_counts,
            name="Frequency",
            marker_color="rgba(255, 0, 0, 0.7)",
            hovertext=[f"Token: {token}<br>Count: {count}<br>Avg Score: {score:.4f}" 
                       for token, count, score in zip(negative_tokens, negative_counts, negative_scores)]
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Common Attention Patterns (Layer {pattern_analysis['layer_idx']}, Head {pattern_analysis['head_idx']})",
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=120)
    )
    
    # Update axes
    fig.update_xaxes(title="Tokens", tickangle=45)
    fig.update_yaxes(title="Frequency")
    
    return fig



# Callback to load IMDb examples
@app.callback(
    Output('imdb-examples-data', 'data'),
    Input('url', 'pathname')
)
def load_imdb_examples(pathname):
    if pathname != "/lime-explanation":
        return {}
    
    try:
        # Load IMDb dataset
        dataset = load_dataset("imdb", split="test")
        
        # Randomly select 5 examples (3 positive, 2 negative)
        positive_examples = [ex for ex in dataset if ex["label"] == 1][:30]
        negative_examples = [ex for ex in dataset if ex["label"] == 0][:20]
        
        # Randomly select examples
        import random
        random.seed(42)  # For reproducibility
        selected_positive = random.sample(positive_examples, 3)
        selected_negative = random.sample(negative_examples, 2)
        
        # Truncate long reviews
        def truncate_text(text, max_length=250):
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text
        
        # Format examples
        examples = []
        for i, ex in enumerate(selected_positive + selected_negative):
            examples.append({
                "id": i,
                "text": truncate_text(ex["text"]),
                "label": ex["label"]
            })
        
        return {"examples": examples}
    except Exception as e:
        print(f"Error loading IMDb examples: {str(e)}")
        return {"examples": []}


# Render IMDb examples
@app.callback(
    Output('lime-examples-container', 'children'),
    Input('imdb-examples-data', 'data')
)
def render_imdb_examples(data):
    if not data or "examples" not in data:
        return html.Div("Error loading examples")
    
    examples = data["examples"]
    if not examples:
        return html.Div("No examples available")
    
    cards = []
    for example in examples:
        label = "Positive" if example["label"] == 1 else "Negative"
        color = "success" if example["label"] == 1 else "danger"
        
        cards.append(
            dbc.Card([
                dbc.CardHeader(f"Example {example['id']+1}: {label} Review", className=f"bg-{color} text-white"),
                dbc.CardBody([
                    html.P(example["text"], className="card-text"),
                    dbc.Button(
                        "Explain This Review", 
                        id={"type": "explain-example-button", "index": example["id"]},
                        color="primary",
                        className="mt-2"
                    )
                ])
            ], className="mb-3")
        )
    
    return html.Div(cards)


# Callback to explain prediction with LIME
@app.callback(
    Output('lime-explanation-data', 'data'),
    [Input('lime-submit-button', 'n_clicks'),
     Input({"type": "explain-example-button", "index": dash.ALL}, "n_clicks")],
    [State('lime-input-text', 'value'),
     State('num-features-slider', 'value'),
     State('num-samples-slider', 'value'),
     State('imdb-examples-data', 'data')],
    prevent_initial_call=True
)
def explain_with_lime(submit_clicks, example_clicks, input_text, num_features, num_samples, imdb_data):
    ctx = callback_context
    
    if not ctx.triggered:
        return {}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        # Check if an example button was clicked
        if "{" in trigger_id:  # It's a pattern-matching ID
            import json
            button_id = json.loads(trigger_id)
            example_idx = button_id["index"]
            text = imdb_data["examples"][example_idx]["text"]
        else:
            # User submitted their own text
            text = input_text
        
        # Get explanation
        explanation = visualizer.explain_sentiment(text, num_features=num_features, num_samples=num_samples)
        
        # Convert to serializable format
        result = {
            "text": explanation["text"],
            "words": explanation["words"],
            "weights": explanation["weights"],
            "prediction": int(explanation["prediction"]),
            "probability": float(explanation["probability"])
        }
        
        return result
    except Exception as e:
        print(f"Error in LIME explanation: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}


# Render LIME explanation results
@app.callback(
    Output('lime-result-container', 'children'),
    Input('lime-explanation-data', 'data')
)
def render_lime_explanation(data):
    if not data:
        return html.Div()
    
    if "error" in data:
        return html.Div(f"Error: {data['error']}", className="text-danger")
    
    prediction_label = "Positive" if data["prediction"] == 1 else "Negative"
    prediction_color = "success" if data["prediction"] == 1 else "danger"
    
    # Create explanation visualization
    words = data["words"]
    weights = data["weights"]
    
    # Sort by absolute weight to show most important features first
    word_weights = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)
    
    # Create bar chart
    fig = go.Figure()
    
    positive_words = [(word, weight) for word, weight in word_weights if weight > 0]
    negative_words = [(word, weight) for word, weight in word_weights if weight <= 0]
    
    # Add bars for positive words
    if positive_words:
        pos_words, pos_weights = zip(*positive_words)
        fig.add_trace(go.Bar(
            y=pos_words,
            x=pos_weights,
            orientation='h',
            marker_color='green',
            name='Positive Impact'
        ))
    
    # Add bars for negative words
    if negative_words:
        neg_words, neg_weights = zip(*negative_words)
        fig.add_trace(go.Bar(
            y=neg_words,
            x=neg_weights,
            orientation='h',
            marker_color='red',
            name='Negative Impact'
        ))
    
    fig.update_layout(
        title="Feature Importance for Sentiment Prediction",
        xaxis_title="Impact on Prediction",
        yaxis_title="Word",
        height=400 + (len(words) * 20),  # Adjust height based on number of words
        margin=dict(l=20, r=20, t=50, b=20),
        barmode='group'
    )
    
    # Highlight words in the text
    def highlight_text(text, words, weights):
        # Create a mapping of words to their weights
        word_to_weight = {word: weight for word, weight in zip(words, weights)}
        
        # Split the text into words
        import re
        text_words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Create highlighted spans for each word
        highlighted_words = []
        for word in text_words:
            if word.lower() in word_to_weight:
                weight = word_to_weight[word.lower()]
                # Scale color intensity based on weight
                intensity = min(255, int(abs(weight) * 200))
                
                if weight > 0:
                    # Green for positive impact
                    color = f"rgba(0, {intensity}, 0, 0.3)"
                    border_color = f"rgba(0, {intensity}, 0, 1)"
                else:
                    # Red for negative impact
                    color = f"rgba({intensity}, 0, 0, 0.3)"
                    border_color = f"rgba({intensity}, 0, 0, 1)"
                
                highlighted_words.append(
                    html.Span(
                        word + " ", 
                        style={
                            "background-color": color,
                            "border": f"1px solid {border_color}",
                            "border-radius": "3px",
                            "padding": "2px",
                            "margin": "1px"
                        }
                    )
                )
            else:
                highlighted_words.append(word + " ")
        
        return html.Div(highlighted_words)
    
    return html.Div([
        dbc.Alert([
            html.H5(f"Prediction: {prediction_label}", className="alert-heading"),
            html.P(f"Confidence: {data['probability']:.2f}")
        ], color=prediction_color, className="mb-4"),
        
        html.H5("Text with Important Words Highlighted"),
        highlight_text(data["text"], words, weights),
        
        html.H5("Feature Importance", className="mt-4"),
        dcc.Graph(figure=fig),
        
        html.Hr(),
        html.P([
            "The chart above shows which words had the most influence on the model's prediction. ",
            "Green bars indicate words that push the prediction toward positive sentiment, ",
            "while red bars indicate words that push the prediction toward negative sentiment."
        ])
    ])


# Add this function after the categorize_error_patterns function
def categorize_error_patterns(high_conf_errors):
    """
    Categorize high confidence errors into common error patterns using a more structured approach
    inspired by Azimuth's smart tags concept.
    
    Args:
        high_conf_errors: List of high confidence error examples
        
    Returns:
        Dictionary of error categories and counts
    """
    # Define error categories with detailed linguistic patterns
    error_categories = {
        "negation_errors": {
            "description": "Errors involving negation words that reverse sentiment",
            "examples": [],
            "patterns": {
                "words": ["not", "n't", "no", "never", "none", "nothing", "neither", "nor", "without", "lack", "absent"],
                "phrases": ["far from", "by no means", "not at all", "not even", "not much", "not very", "hardly", "barely"]
            }
        },
        "intensity_errors": {
            "description": "Errors involving intensity modifiers that amplify sentiment",
            "examples": [],
            "patterns": {
                "words": ["very", "extremely", "really", "absolutely", "completely", "totally", "utterly", 
                         "highly", "incredibly", "exceptionally", "too", "so", "quite", "rather"],
                "phrases": ["a lot", "to a great extent", "by far"]
            }
        },
        "context_errors": {
            "description": "Errors in longer sentences with complex context",
            "examples": [],
            "patterns": {
                "length_threshold": 15  # Words
            }
        },
        "comparison_errors": {
            "description": "Errors involving comparison or contrast between different sentiments",
            "examples": [],
            "patterns": {
                "words": ["but", "however", "although", "though", "despite", "in spite", "nevertheless", 
                         "nonetheless", "yet", "still", "while", "whereas", "unlike", "contrary"],
                "phrases": ["on the other hand", "even though", "in contrast", "as opposed to", "rather than"]
            }
        },
        "sarcasm_errors": {
            "description": "Errors potentially involving sarcasm or irony",
            "examples": [],
            "patterns": {
                "phrases": ["yeah right", "sure", "as if", "whatever", "oh great", "big deal", "wow", 
                           "oh joy", "bravo", "how nice", "just what I needed", "good luck with that"]
            }
        },
        "ambiguity_errors": {
            "description": "Errors involving words with ambiguous or context-dependent sentiment",
            "examples": [],
            "patterns": {
                "words": ["interesting", "surprising", "impressive", "remarkable", "notable", "unusual",
                         "different", "special", "particular", "certain", "fine", "okay", "ok"]
            }
        },
        "conditional_errors": {
            "description": "Errors involving conditional statements",
            "examples": [],
            "patterns": {
                "words": ["if", "would", "could", "should", "may", "might", "can", "will", "unless"],
                "phrases": ["as long as", "provided that", "assuming that", "in case"]
            }
        },
        "other_errors": {
            "description": "Errors that don't fit into other categories",
            "examples": []
        }
    }
    
    # Function to check if a sentence contains any pattern from a list
    def contains_pattern(sentence, patterns):
        sentence = sentence.lower()
        words = sentence.split()
        
        # Check for individual words
        if "words" in patterns:
            if any(word in words or f" {word} " in sentence for word in patterns["words"]):
                return True
                
        # Check for phrases
        if "phrases" in patterns:
            if any(phrase in sentence for phrase in patterns["phrases"]):
                return True
                
        return False
    
    # Categorize each error
    for error in high_conf_errors:
        sentence = error["sentence"].lower()
        categorized = False
        
        # Check for negation errors
        if contains_pattern(sentence, error_categories["negation_errors"]["patterns"]):
            error_categories["negation_errors"]["examples"].append(error)
            categorized = True
            
        # Check for comparison errors (if not already categorized)
        elif contains_pattern(sentence, error_categories["comparison_errors"]["patterns"]):
            error_categories["comparison_errors"]["examples"].append(error)
            categorized = True
            
        # Check for intensity errors
        elif contains_pattern(sentence, error_categories["intensity_errors"]["patterns"]):
            error_categories["intensity_errors"]["examples"].append(error)
            categorized = True
            
        # Check for sarcasm errors
        elif contains_pattern(sentence, error_categories["sarcasm_errors"]["patterns"]):
            error_categories["sarcasm_errors"]["examples"].append(error)
            categorized = True
            
        # Check for ambiguity errors
        elif contains_pattern(sentence, error_categories["ambiguity_errors"]["patterns"]):
            error_categories["ambiguity_errors"]["examples"].append(error)
            categorized = True
            
        # Check for conditional errors
        elif contains_pattern(sentence, error_categories["conditional_errors"]["patterns"]):
            error_categories["conditional_errors"]["examples"].append(error)
            categorized = True
            
        # Check for context errors (longer sentences)
        elif len(sentence.split()) > error_categories["context_errors"]["patterns"]["length_threshold"]:
            error_categories["context_errors"]["examples"].append(error)
            categorized = True
            
        # Other errors
        if not categorized:
            error_categories["other_errors"]["examples"].append(error)
    
    # Create a simplified version for the return value
    result = {}
    for category, data in error_categories.items():
        result[category] = {
            "description": data["description"],
            "examples": data["examples"],
            "count": len(data["examples"])
        }
    
    return result

# Modify the analyze_model_errors callback to include error pattern analysis
@app.callback(
    [Output('error-analysis-data', 'data'),
     Output('analysis-loading-indicator', 'children')],
    Input('analyze-errors-button', 'n_clicks'),
    [State('num-samples-slider-error', 'value'),
     State('confidence-threshold-slider', 'value')],
    prevent_initial_call=True
)
def analyze_model_errors(n_clicks, num_samples, confidence_threshold):
    if not n_clicks:
        return {}, ""
    
    # Show loading indicator
    loading_indicator = html.Div([
        html.Span("Analyzing model errors... This may take a few minutes.", className="me-2"),
        html.Div(className="spinner-border spinner-border-sm", role="status")
    ], className="text-primary")
    
    try:
        # Load SST-2 dataset
        dataset = load_dataset("nyu-mll/glue", "sst2")
        validation_set = dataset['validation']
        
        # Limit to specified number of samples
        samples = validation_set.select(range(min(num_samples, len(validation_set))))
        
        # Analyze each sample
        results = []
        for i, sample in enumerate(samples):
            sentence = sample["sentence"]
            true_label = sample["label"]
            
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            try:
                # Get model prediction
                prediction = visualizer.predict_sentiment(sentence)
                pred_label = prediction["label"]
                confidence = prediction["score"]
                
                # Store result
                results.append({
                    "sentence": sentence,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "correct": true_label == pred_label,
                    "sentence_length": len(sentence.split()),
                    "has_negation": 1 if any(neg in sentence.lower() for neg in ["not", "n't", "no", "never"]) else 0,
                })
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
        
        # Analyze high confidence errors
        high_conf_errors = [r for r in results if not r["correct"] and r["confidence"] >= confidence_threshold]
        
        # Categorize errors by pattern
        error_categories = categorize_error_patterns(high_conf_errors)
        
        # Extract common words in errors
        error_sentences = [r["sentence"].lower() for r in high_conf_errors]
        all_words = []
        for sentence in error_sentences:
            # Remove punctuation and split into words
            words = re.findall(r'\b\w+\b', sentence.lower())
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Remove common stopwords
        stopwords = ["the", "a", "an", "and", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were",
                    "this", "that", "it", "i", "you", "he", "she", "they", "we", "his", "her", "their", "our", "its",
                    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
                    "can", "could", "may", "might", "must", "by", "from", "as", "about", "like", "through", "after",
                    "over", "between", "out", "against", "during", "before", "because", "if", "any", "these", "those"]
        for word in stopwords:
            if word in word_counts:
                del word_counts[word]
        
        # Get top words
        top_words = word_counts.most_common(15)
        
        # Prepare category counts and examples for the UI
        error_categories_ui = {}
        error_examples_ui = {}
        
        for category, data in error_categories.items():
            error_categories_ui[category] = data["count"]
            error_examples_ui[category] = data["examples"][:5]  # Limit to 5 examples per category
        
        return {
            "total_samples": len(results),
            "correct_count": sum(1 for r in results if r["correct"]),
            "error_count": sum(1 for r in results if not r["correct"]),
            "high_conf_error_count": len(high_conf_errors),
            "high_conf_errors": high_conf_errors[:50],  # Limit to 50 examples
            "top_error_words": top_words,
            "confidence_threshold": confidence_threshold,
            "results": results,  # All results for additional analysis
            "error_categories": error_categories_ui,
            "error_examples": error_examples_ui,
            "error_descriptions": {category: data["description"] for category, data in error_categories.items()}
        }, ""
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in error analysis: {str(e)}\n{traceback_str}")
        return {"error": str(e)}, html.Div(f"Error: {str(e)}", className="text-danger")

# Update the render_error_analysis function to display enhanced error categorization
@app.callback(
    Output('error-analysis-results', 'children'),
    Input('error-analysis-data', 'data'),
    prevent_initial_call=True
)
def render_error_analysis(data):
    if not data:
        return html.Div()
    
    if "error" in data:
        return html.Div(f"Error: {data['error']}", className="text-danger")
    
    # Create a DataFrame for analysis
    if data.get("results"):
        df = pd.DataFrame(data["results"])
    else:
        return html.Div("No results to analyze")
    
    # Calculate overall accuracy
    accuracy = data["correct_count"] / data["total_samples"] if data["total_samples"] > 0 else 0
    
    # Create summary statistics
    summary = dbc.Card([
        dbc.CardBody([
            html.H4("Summary Statistics", className="card-title"),
            html.P([
                f"Total samples analyzed: {data['total_samples']}",
                html.Br(),
                f"Correct predictions: {data['correct_count']} ({accuracy:.2%})",
                html.Br(),
                f"Incorrect predictions: {data['error_count']} ({1-accuracy:.2%})",
                html.Br(),
                f"High confidence errors (≥{data['confidence_threshold']:.2f}): {data['high_conf_error_count']}"
            ])
        ])
    ], className="mb-4")
    
    # Create confusion matrix
    confusion_data = df.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
    
    # Ensure all combinations exist
    for true_label in [0, 1]:
        for pred_label in [0, 1]:
            if not ((confusion_data['true_label'] == true_label) & 
                    (confusion_data['pred_label'] == pred_label)).any():
                confusion_data = pd.concat([confusion_data, pd.DataFrame({
                    'true_label': [true_label],
                    'pred_label': [pred_label],
                    'count': [0]
                })], ignore_index=True)
    
    confusion_fig = px.imshow(
        confusion_data.pivot(index='true_label', columns='pred_label', values='count'),
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['Negative (0)', 'Positive (1)'],
        y=['Negative (0)', 'Positive (1)'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    confusion_fig.update_layout(title="Confusion Matrix")
    
    # Create confidence distribution plot
    confidence_fig = px.histogram(
        df, 
        x="confidence", 
        color="correct",
        nbins=20,
        labels={"confidence": "Confidence Score", "correct": "Prediction"},
        color_discrete_map={True: "green", False: "red"},
        title="Confidence Distribution by Prediction Correctness",
        barmode="overlay",
        opacity=0.7
    )
    
    # Create sentence length analysis
    length_fig = px.box(
        df,
        x="correct",
        y="sentence_length",
        labels={"correct": "Prediction", "sentence_length": "Sentence Length (words)"},
        color="correct",
        color_discrete_map={True: "green", False: "red"},
        title="Sentence Length by Prediction Correctness"
    )
    
    # Create negation analysis
    negation_counts = df.groupby(['correct', 'has_negation']).size().reset_index(name='count')
    negation_fig = px.bar(
        negation_counts,
        x="correct",
        y="count",
        color="has_negation",
        barmode="group",
        labels={"correct": "Prediction", "count": "Count", "has_negation": "Contains Negation"},
        title="Impact of Negation on Prediction Correctness"
    )
    
    # Extract common words in errors
    error_sentences = [r["sentence"].lower() for r in data["results"] if not r["correct"]]
    all_words = []
    for sentence in error_sentences:
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', sentence.lower())
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Remove common stopwords
    stopwords = ["the", "a", "an", "and", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were",
                "this", "that", "it", "i", "you", "he", "she", "they", "we", "his", "her", "their", "our", "its",
                "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
                "can", "could", "may", "might", "must", "by", "from", "as", "about", "like", "through", "after",
                "over", "between", "out", "against", "during", "before", "because", "if", "any", "these", "those"]
    for word in stopwords:
        if word in word_counts:
            del word_counts[word]
    
    # Get top 15 words
    top_words = word_counts.most_common(15)
    
    # Create bar chart for top words
    if top_words:
        words, counts = zip(*top_words)
        top_words_fig = px.bar(
            x=words, 
            y=counts,
            labels={"x": "Word", "y": "Frequency"},
            title="Top 15 Words in Misclassified Sentences",
            color=counts,
            color_continuous_scale="Viridis"
        )
        top_words_fig.update_layout(xaxis={'categoryorder':'total descending'})
        
        top_words_div = html.Div([
            html.H4("Most Frequent Words in Error Cases", className="mt-4"),
            dcc.Graph(figure=top_words_fig)
        ])
    else:
        top_words_div = html.Div()
    
    # Create enhanced error pattern analysis visualization
    if "error_categories" in data:
        categories = list(data["error_categories"].keys())
        counts = list(data["error_categories"].values())
        descriptions = [data["error_descriptions"].get(cat, "") for cat in categories]
        
        # Create data for visualization
        pattern_data = pd.DataFrame({
            "category": categories,
            "count": counts,
            "description": descriptions
        })
        
        # Sort by count descending
        pattern_data = pattern_data.sort_values("count", ascending=False)
        
        # Create bar chart of error categories with hover descriptions
        pattern_fig = px.bar(
            pattern_data,
            x="category",
            y="count",
            hover_data=["description"],
            labels={"category": "Error Category", "count": "Count", "description": "Description"},
            title="Distribution of Error Patterns",
            color="count",
            color_continuous_scale="Viridis"
        )
        pattern_fig.update_layout(xaxis={'categoryorder':'total descending'})
        
        # Create examples for each error category with descriptions
        error_pattern_examples = []
        
        if "error_examples" in data:
            for category in pattern_data["category"]:
                examples = data["error_examples"].get(category, [])
                description = data["error_descriptions"].get(category, "")
                
                if examples:
                    # Format category name for display
                    category_display = " ".join(word.capitalize() for word in category.split("_"))
                    
                    # Create examples table
                    rows = []
                    for i, ex in enumerate(examples):
                        true_label = "Positive" if ex["true_label"] == 1 else "Negative"
                        pred_label = "Positive" if ex["pred_label"] == 1 else "Negative"
                        
                        row = html.Tr([
                            html.Td(i+1),
                            html.Td(ex["sentence"]),
                            html.Td(true_label),
                            html.Td(pred_label),
                            html.Td(f"{ex['confidence']:.4f}")
                        ])
                        rows.append(row)
                    
                    # Create category card with description and examples
                    card = dbc.Card([
                        dbc.CardHeader(f"{category_display} ({len(examples)} examples)"),
                        dbc.CardBody([
                            html.P(description, className="text-muted"),
                            dbc.Table([
                                html.Thead(html.Tr([
                                    html.Th("#"),
                                    html.Th("Sentence"),
                                    html.Th("True"),
                                    html.Th("Pred"),
                                    html.Th("Conf")
                                ])),
                                html.Tbody(rows)
                            ], bordered=True, size="sm")
                        ])
                    ], className="mb-3")
                    
                    error_pattern_examples.append(card)
        
        # Create tabs for different visualizations
        pattern_analysis = dbc.Tabs([
            dbc.Tab([
                html.Div([
                    html.P("This chart shows the distribution of different error categories.", className="mt-3"),
                    dcc.Graph(figure=pattern_fig)
                ])
            ], label="Error Categories Chart"),
            dbc.Tab([
                html.Div([
                    html.H4("Examples of Error Patterns", className="mt-3"),
                    html.P("Below are examples of sentences that fall into each error category."),
                    html.Div(error_pattern_examples)
                ], className="mt-3")
            ], label="Error Examples")
        ])
    else:
        pattern_analysis = html.Div("No error pattern analysis available")
    
    # Create table of high confidence errors
    if data["high_conf_errors"]:
        error_rows = []
        for i, error in enumerate(data["high_conf_errors"]):
            true_label = "Positive" if error["true_label"] == 1 else "Negative"
            pred_label = "Positive" if error["pred_label"] == 1 else "Negative"
            
            row = html.Tr([
                html.Td(i+1),
                html.Td(error["sentence"]),
                html.Td(true_label, className="fw-bold"),
                html.Td(pred_label, className="text-danger fw-bold"),
                html.Td(f"{error['confidence']:.4f}")
            ])
            error_rows.append(row)
        
        error_table = dbc.Card([
            dbc.CardHeader("High Confidence Errors"),
            dbc.CardBody([
                html.P([
                    f"Showing errors with confidence ≥ {data['confidence_threshold']:.2f}. ",
                    "These are cases where the model is confidently wrong."
                ]),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("#"),
                        html.Th("Sentence"),
                        html.Th("True Label"),
                        html.Th("Predicted"),
                        html.Th("Confidence")
                    ])),
                    html.Tbody(error_rows)
                ], bordered=True, size="sm", responsive=True)
            ])
        ], className="mb-4")
    else:
        error_table = html.Div("No high confidence errors found.")
    
    # Create tabs for different visualizations
    visualizations = dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=confusion_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=confidence_fig)
                ], md=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=length_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=negation_fig)
                ], md=6)
            ])
        ], label="Performance Analysis"),
        dbc.Tab([
            html.Div([
                html.H4("Error Pattern Analysis", className="mt-3"),
                pattern_analysis
            ])
        ], label="Error Pattern Analysis"),
        dbc.Tab([
            html.Div([
                top_words_div,
                html.H4("High Confidence Errors", className="mt-4"),
                error_table
            ])
        ], label="Error Examples")
    ])
    
    return html.Div([
        summary,
        visualizations
    ])

# Add these imports if not already present
import lime
import lime.lime_text

# Add this function after the categorize_error_patterns function
def analyze_error_with_lime(sentence, true_label, num_features=10, num_samples=1000):
    """
    Analyze an error case using LIME to understand why the model made a mistake
    
    Args:
        sentence: The sentence that was misclassified
        true_label: The true sentiment label (0=negative, 1=positive)
        num_features: Number of features to include in explanation
        num_samples: Number of samples to use for LIME
        
    Returns:
        Dictionary containing LIME explanation and analysis
    """
    try:
        # Get LIME explanation
        explanation = visualizer.explain_sentiment(sentence, num_features=num_features, num_samples=num_samples)
        
        # Get prediction details
        pred_label = explanation["prediction"]
        confidence = explanation["probability"]
        
        # Get words and their contributions
        words = explanation["words"]
        weights = explanation["weights"]
        
        # Analyze which words contributed to the incorrect prediction
        misleading_words = []
        
        # For positive sentences misclassified as negative
        if true_label == 1 and pred_label == 0:
            # Words with negative weights (pushing toward negative) are misleading
            misleading_words = [(word, weight) for word, weight in zip(words, weights) if weight < 0]
        
        # For negative sentences misclassified as positive
        elif true_label == 0 and pred_label == 1:
            # Words with positive weights (pushing toward positive) are misleading
            misleading_words = [(word, weight) for word, weight in zip(words, weights) if weight > 0]
        
        # Sort by absolute weight
        misleading_words.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            "sentence": sentence,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "words": words,
            "weights": weights,
            "misleading_words": misleading_words
        }
    except Exception as e:
        print(f"Error in LIME analysis: {str(e)}")
        return {
            "sentence": sentence,
            "error": str(e)
        }

# Add a new callback for LIME analysis of selected error examples
@app.callback(
    Output('lime-error-analysis-result', 'children'),
    Input('analyze-error-with-lime-button', 'n_clicks'),
    [State('error-example-dropdown', 'value'),
     State('error-analysis-data', 'data')],
    prevent_initial_call=True
)
def update_lime_error_analysis(n_clicks, selected_example_idx, data):
    if not n_clicks or selected_example_idx is None or not data or "high_conf_errors" not in data:
        return html.Div()
    
    try:
        # Get the selected error example
        selected_idx = int(selected_example_idx)
        if selected_idx < 0 or selected_idx >= len(data["high_conf_errors"]):
            return html.Div("Invalid example selected")
        
        error_example = data["high_conf_errors"][selected_idx]
        sentence = error_example["sentence"]
        true_label = error_example["true_label"]
        
        # Show loading indicator
        loading_div = html.Div([
            html.Span("Analyzing with LIME... This may take a moment.", className="me-2"),
            html.Div(className="spinner-border spinner-border-sm", role="status")
        ], className="text-primary my-3")
        
        # Analyze with LIME
        lime_result = analyze_error_with_lime(sentence, true_label)
        
        if "error" in lime_result:
            return html.Div(f"Error in LIME analysis: {lime_result['error']}", className="text-danger")
        
        # Create visualization of LIME results
        words = lime_result["words"]
        weights = lime_result["weights"]
        misleading_words = lime_result["misleading_words"]
        
        # Create bar chart for word importance
        fig = go.Figure()
        
        # Add bars for word weights
        fig.add_trace(go.Bar(
            y=words,
            x=weights,
            orientation='h',
            marker_color=['green' if w > 0 else 'red' for w in weights],
            name='Word Importance'
        ))
        
        fig.update_layout(
            title="Word Importance for Prediction",
            xaxis_title="Impact on Prediction (Positive → / ← Negative)",
            yaxis_title="Word",
            height=400 + (len(words) * 20),  # Adjust height based on number of words
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Highlight words in the text
        def highlight_text(text, words, weights):
            # Create a mapping of words to their weights
            word_to_weight = {word: weight for word, weight in zip(words, weights)}
            
            # Split the text into words
            text_words = re.findall(r'\b\w+\b|[^\w\s]', text)
            
            # Create highlighted spans for each word
            highlighted_words = []
            for word in text_words:
                if word.lower() in word_to_weight:
                    weight = word_to_weight[word.lower()]
                    # Scale color intensity based on weight
                    intensity = min(255, int(abs(weight) * 200))
                    
                    if weight > 0:
                        # Green for positive impact
                        color = f"rgba(0, {intensity}, 0, 0.3)"
                        border_color = f"rgba(0, {intensity}, 0, 1)"
                    else:
                        # Red for negative impact
                        color = f"rgba({intensity}, 0, 0, 0.3)"
                        border_color = f"rgba({intensity}, 0, 0, 1)"
                    
                    highlighted_words.append(
                        html.Span(
                            word + " ", 
                            style={
                                "background-color": color,
                                "border": f"1px solid {border_color}",
                                "border-radius": "3px",
                                "padding": "2px",
                                "margin": "1px"
                            }
                        )
                    )
                else:
                    highlighted_words.append(word + " ")
            
            return html.Div(highlighted_words)
        
        # Create table of misleading words
        if misleading_words:
            misleading_rows = []
            for i, (word, weight) in enumerate(misleading_words):
                row = html.Tr([
                    html.Td(i+1),
                    html.Td(word),
                    html.Td(f"{weight:.4f}", style={"color": "red" if weight < 0 else "green"})
                ])
                misleading_rows.append(row)
            
            misleading_table = dbc.Table([
                html.Thead(html.Tr([
                    html.Th("#"),
                    html.Th("Word"),
                    html.Th("Impact")
                ])),
                html.Tbody(misleading_rows)
            ], bordered=True, hover=True, size="sm")
        else:
            misleading_table = html.Div("No misleading words identified")
        
        # Create the result display
        true_label_text = "Positive" if true_label == 1 else "Negative"
        pred_label_text = "Positive" if lime_result["pred_label"] == 1 else "Negative"
        
        result_div = html.Div([
            html.H4("LIME Analysis of Error Case", className="mt-3"),
            
            dbc.Alert([
                html.P([
                    html.Strong("Sentence: "), 
                    highlight_text(sentence, words, weights)
                ]),
                html.P([
                    html.Strong("True Label: "), 
                    html.Span(true_label_text, className="badge bg-primary")
                ]),
                html.P([
                    html.Strong("Predicted Label: "), 
                    html.Span(pred_label_text, className="badge bg-danger")
                ]),
                html.P([
                    html.Strong("Confidence: "), 
                    f"{lime_result['confidence']:.4f}"
                ])
            ], color="info"),
            
            html.H5("Word Importance", className="mt-4"),
            dcc.Graph(figure=fig),
            
            html.H5("Misleading Words", className="mt-4"),
            html.P("These words contributed most to the incorrect prediction:"),
            misleading_table,
            
            html.Hr(),
            html.P([
                "This analysis shows which words influenced the model's incorrect prediction. ",
                "Words highlighted in green pushed the prediction toward positive sentiment, ",
                "while words highlighted in red pushed it toward negative sentiment."
            ])
        ])
        
        return result_div
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in LIME error analysis: {str(e)}\n{traceback_str}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

# Modify the error analysis layout to include LIME analysis section
error_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Model Error Analysis", className="text-center my-4"),
            html.P([
                "This page analyzes where the sentiment model fails on the SST-2 dataset. ",
                "It helps identify patterns in sentences where the model makes incorrect predictions."
            ], className="lead text-center")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Dataset Analysis Settings", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of samples to analyze:"),
                            dcc.Slider(
                                id="num-samples-slider-error",
                                min=100,
                                max=500,
                                step=100,
                                value=200,
                                marks={i: str(i) for i in range(100, 501, 100)},
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Confidence threshold:"),
                            dcc.Slider(
                                id="confidence-threshold-slider",
                                min=0.5,
                                max=0.95,
                                step=0.05,
                                value=0.7,
                                marks={i/100: str(i/100) for i in range(50, 96, 5)},
                            ),
                        ], width=6),
                    ], className="mb-3"),
                    dbc.Button("Analyze Model Errors", id="analyze-errors-button", color="primary", className="mt-3"),
                    html.Div(id="analysis-loading-indicator", className="mt-2")
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="error-analysis-results", className="mt-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Error Analysis Tools"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.H5("LIME Analysis of Error Cases", className="mt-3"),
                            html.P([
                                "Select an error example to analyze why the model made a mistake on this specific sentence. ",
                                "LIME will show which words influenced the incorrect prediction."
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select an error example:"),
                                    dcc.Dropdown(
                                        id="error-example-dropdown",
                                        options=[],
                                        placeholder="First run error analysis above"
                                    ),
                                ], width=8),
                                dbc.Col([
                                    dbc.Button(
                                        "Analyze with LIME", 
                                        id="analyze-error-with-lime-button", 
                                        color="info", 
                                        className="mt-4"
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            html.Div(id="lime-error-analysis-result")
                        ], label="LIME Analysis"),
                        
                        dbc.Tab([
                            html.H5("Counterfactual Testing", className="mt-3"),
                            html.P([
                                "Test how small changes to a misclassified sentence affect the model's prediction. ",
                                "This helps identify which parts of the sentence are causing the model to make errors."
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select an error example:"),
                                    dcc.Dropdown(
                                        id="counterfactual-example-dropdown",
                                        options=[],
                                        placeholder="First run error analysis above"
                                    ),
                                ], width=8),
                                dbc.Col([
                                    dbc.Button(
                                        "Test Counterfactuals", 
                                        id="test-counterfactuals-button", 
                                        color="info", 
                                        className="mt-4"
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            html.Div(id="counterfactual-results")
                        ], label="Counterfactual Testing"),
                        
                        dbc.Tab([
                            html.H5("Similarity-Based Error Analysis", className="mt-3"),
                            html.P([
                                "Find patterns in error cases using TF-IDF similarity and clustering. ",
                                "This helps identify groups of similar errors that might share common causes."
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Number of clusters:"),
                                    dbc.Input(
                                        id="num-clusters-input",
                                        type="number",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=5
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Button(
                                        "Analyze Error Similarity", 
                                        id="analyze-similarity-button", 
                                        color="info", 
                                        className="mt-4"
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            html.Div(id="similarity-analysis-results")
                        ], label="Similarity Analysis")
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Store for error analysis data
    dcc.Store(id='error-analysis-data')
], fluid=True)

# Add callback to populate error example dropdown
@app.callback(
    Output('error-example-dropdown', 'options'),
    Input('error-analysis-data', 'data'),
    prevent_initial_call=True
)
def populate_error_dropdown(data):
    if not data or "high_conf_errors" not in data:
        return []
    
    options = []
    for i, error in enumerate(data["high_conf_errors"]):
        true_label = "Positive" if error["true_label"] == 1 else "Negative"
        pred_label = "Positive" if error["pred_label"] == 1 else "Negative"
        
        # Truncate long sentences for display
        sentence = error["sentence"]
        if len(sentence) > 50:
            sentence = sentence[:47] + "..."
        
        option_label = f"{i+1}. {sentence} (True: {true_label}, Pred: {pred_label})"
        options.append({"label": option_label, "value": i})
    
    return options

# Add a function to generate counterfactual examples
def generate_counterfactuals(sentence, error_type):
    """
    Generate counterfactual examples based on the error type to test model robustness
    
    Args:
        sentence: The original sentence
        error_type: The type of error identified
        
    Returns:
        List of counterfactual examples with their transformations
    """
    counterfactuals = []
    
    # Helper function to replace a word in a sentence
    def replace_word(sentence, word, replacement):
        words = sentence.split()
        for i, w in enumerate(words):
            if w.lower() == word.lower() or w.lower() == word.lower() + '.' or w.lower() == word.lower() + ',':
                # Preserve punctuation
                punct = ''
                if not w.isalpha():
                    punct = w[-1] if not w[-1].isalnum() else ''
                words[i] = replacement + punct
                break
        return ' '.join(words)
    
    # Helper function to remove a word from a sentence
    def remove_word(sentence, word):
        words = sentence.split()
        result = [w for w in words if w.lower() != word.lower() and w.lower() != word.lower() + '.' and w.lower() != word.lower() + ',']
        return ' '.join(result)
    
    # Generate counterfactuals based on error type
    if error_type == "negation_errors":
        # Remove negation
        negation_words = ["not", "n't", "no", "never", "none", "nothing", "neither", "nor"]
        for neg in negation_words:
            if neg == "n't":
                # Handle contractions
                if "n't" in sentence.lower():
                    # Replace don't with do, won't with will, etc.
                    for contraction, replacement in [("don't", "do"), ("doesn't", "does"), 
                                                    ("isn't", "is"), ("aren't", "are"),
                                                    ("wasn't", "was"), ("weren't", "were"),
                                                    ("hasn't", "has"), ("haven't", "have"),
                                                    ("hadn't", "had"), ("wouldn't", "would"),
                                                    ("shouldn't", "should"), ("couldn't", "could"),
                                                    ("won't", "will"), ("can't", "can")]:
                        if contraction in sentence.lower():
                            new_sentence = sentence.lower().replace(contraction, replacement)
                            counterfactuals.append({
                                "type": "Remove negation",
                                "original": sentence,
                                "counterfactual": new_sentence,
                                "transformation": f"Replace '{contraction}' with '{replacement}'"
                            })
            elif neg in sentence.lower().split() or f" {neg} " in sentence.lower():
                new_sentence = remove_word(sentence, neg)
                if new_sentence != sentence:
                    counterfactuals.append({
                        "type": "Remove negation",
                        "original": sentence,
                        "counterfactual": new_sentence,
                        "transformation": f"Remove '{neg}'"
                    })
    
    elif error_type == "intensity_errors":
        # Remove intensity modifiers
        intensity_words = ["very", "extremely", "really", "absolutely", "completely", "totally", 
                          "utterly", "highly", "incredibly", "exceptionally", "too", "so", "quite", "rather"]
        for intens in intensity_words:
            if intens in sentence.lower().split() or f" {intens} " in sentence.lower():
                new_sentence = remove_word(sentence, intens)
                if new_sentence != sentence:
                    counterfactuals.append({
                        "type": "Remove intensity",
                        "original": sentence,
                        "counterfactual": new_sentence,
                        "transformation": f"Remove '{intens}'"
                    })
    
    elif error_type == "comparison_errors":
        # Split sentence at comparison word
        comparison_words = ["but", "however", "although", "though", "despite", "nevertheless", 
                           "nonetheless", "yet", "still", "while", "whereas"]
        for comp in comparison_words:
            if comp in sentence.lower().split() or f" {comp} " in sentence.lower():
                parts = re.split(f'\\b{comp}\\b', sentence, flags=re.IGNORECASE)
                if len(parts) > 1:
                    counterfactuals.append({
                        "type": "First clause only",
                        "original": sentence,
                        "counterfactual": parts[0].strip(),
                        "transformation": f"Keep only text before '{comp}'"
                    })
                    counterfactuals.append({
                        "type": "Second clause only",
                        "original": sentence,
                        "counterfactual": parts[1].strip(),
                        "transformation": f"Keep only text after '{comp}'"
                    })
    
    elif error_type == "sarcasm_errors":
        # Try removing sarcasm indicators
        sarcasm_indicators = ["yeah right", "sure", "as if", "whatever", "oh great", "big deal", 
                             "wow", "oh joy", "bravo", "how nice", "just what I needed"]
        for sarc in sarcasm_indicators:
            if sarc in sentence.lower():
                new_sentence = sentence.lower().replace(sarc, "").strip()
                if new_sentence != sentence.lower():
                    counterfactuals.append({
                        "type": "Remove sarcasm indicator",
                        "original": sentence,
                        "counterfactual": new_sentence,
                        "transformation": f"Remove '{sarc}'"
                    })
    
    elif error_type == "ambiguity_errors":
        # Try adding explicit sentiment words
        ambiguous_words = ["interesting", "surprising", "impressive", "remarkable", "notable", "unusual",
                         "different", "special", "particular", "certain", "fine", "okay", "ok"]
        for amb in ambiguous_words:
            if amb in sentence.lower().split() or f" {amb} " in sentence.lower():
                # Try replacing with positive alternative
                positive_replacements = {"interesting": "fascinating", "surprising": "delightful", 
                                        "impressive": "excellent", "remarkable": "outstanding",
                                        "notable": "exceptional", "unusual": "unique",
                                        "different": "innovative", "special": "wonderful",
                                        "particular": "specific", "certain": "definite",
                                        "fine": "good", "okay": "good", "ok": "good"}
                
                # Try replacing with negative alternative
                negative_replacements = {"interesting": "confusing", "surprising": "disappointing", 
                                        "impressive": "mediocre", "remarkable": "underwhelming",
                                        "notable": "forgettable", "unusual": "strange",
                                        "different": "odd", "special": "problematic",
                                        "particular": "peculiar", "certain": "rigid",
                                        "fine": "mediocre", "okay": "mediocre", "ok": "mediocre"}
                
                if amb in positive_replacements:
                    new_sentence = replace_word(sentence, amb, positive_replacements[amb])
                    counterfactuals.append({
                        "type": "Disambiguate positively",
                        "original": sentence,
                        "counterfactual": new_sentence,
                        "transformation": f"Replace '{amb}' with '{positive_replacements[amb]}'"
                    })
                    
                if amb in negative_replacements:
                    new_sentence = replace_word(sentence, amb, negative_replacements[amb])
                    counterfactuals.append({
                        "type": "Disambiguate negatively",
                        "original": sentence,
                        "counterfactual": new_sentence,
                        "transformation": f"Replace '{amb}' with '{negative_replacements[amb]}'"
                    })
    
    elif error_type == "conditional_errors":
        # Try removing conditional part
        conditional_words = ["if", "would", "could", "should", "may", "might", "can", "will", "unless"]
        for cond in conditional_words:
            if cond in sentence.lower().split() or f" {cond} " in sentence.lower():
                # Try to split at the conditional
                parts = re.split(f'\\b{cond}\\b', sentence, flags=re.IGNORECASE)
                if len(parts) > 1:
                    counterfactuals.append({
                        "type": "Remove conditional",
                        "original": sentence,
                        "counterfactual": parts[0].strip(),
                        "transformation": f"Keep only text before '{cond}'"
                    })
    
    # If no specific counterfactuals were generated or for context errors
    if not counterfactuals or error_type == "context_errors":
        # Try simplifying by taking first half of sentence
        words = sentence.split()
        if len(words) > 8:  # Only for longer sentences
            half_length = len(words) // 2
            first_half = ' '.join(words[:half_length])
            second_half = ' '.join(words[half_length:])
            
            counterfactuals.append({
                "type": "Simplify (first half)",
                "original": sentence,
                "counterfactual": first_half,
                "transformation": "Keep only first half of sentence"
            })
            
            counterfactuals.append({
                "type": "Simplify (second half)",
                "original": sentence,
                "counterfactual": second_half,
                "transformation": "Keep only second half of sentence"
            })
    
    return counterfactuals

# Add a callback for counterfactual testing
@app.callback(
    Output('counterfactual-results', 'children'),
    Input('test-counterfactuals-button', 'n_clicks'),
    [State('counterfactual-example-dropdown', 'value'),
     State('error-analysis-data', 'data')],
    prevent_initial_call=True
)
def test_counterfactuals(n_clicks, selected_example_idx, data):
    if not n_clicks or not selected_example_idx or not data:
        return html.Div("Select an error example and click 'Test Counterfactuals' to see how small changes affect the model's prediction.")
    
    try:
        # Get the selected error example
        high_conf_errors = data.get("high_conf_errors", [])
        if not high_conf_errors or int(selected_example_idx) >= len(high_conf_errors):
            return html.Div("Error example not found.", className="text-danger")
        
        error_example = high_conf_errors[int(selected_example_idx)]
        sentence = error_example["sentence"]
        true_label = error_example["true_label"]
        pred_label = error_example["pred_label"]
        
        # Determine error type
        error_type = "other_errors"
        for category, examples in data.get("error_examples", {}).items():
            for ex in examples:
                if ex["sentence"] == sentence:
                    error_type = category
                    break
        
        # Generate counterfactuals
        counterfactuals = generate_counterfactuals(sentence, error_type)
        
        if not counterfactuals:
            return html.Div("No counterfactual examples could be generated for this error type.")
        
        # Test each counterfactual with the model
        results = []
        for cf in counterfactuals:
            try:
                # Get model prediction
                prediction = visualizer.predict_sentiment(cf["counterfactual"])
                cf_pred_label = prediction["label"]
                cf_confidence = prediction["score"]
                
                # Check if prediction changed
                prediction_changed = cf_pred_label != pred_label
                prediction_correct = cf_pred_label == true_label
                
                results.append({
                    **cf,
                    "pred_label": cf_pred_label,
                    "confidence": cf_confidence,
                    "prediction_changed": prediction_changed,
                    "prediction_correct": prediction_correct
                })
            except Exception as e:
                print(f"Error testing counterfactual: {str(e)}")
                continue
        
        # Create a table of results
        rows = []
        for i, result in enumerate(results):
            # Format labels
            cf_label = "Positive" if result["pred_label"] == 1 else "Negative"
            
            # Determine row color based on whether prediction is now correct
            row_class = "table-success" if result["prediction_correct"] else ""
            
            row = html.Tr([
                html.Td(i+1),
                html.Td(result["type"]),
                html.Td(result["counterfactual"]),
                html.Td(result["transformation"]),
                html.Td(cf_label),
                html.Td(f"{result['confidence']:.4f}"),
                html.Td("✓" if result["prediction_changed"] else "✗"),
                html.Td("✓" if result["prediction_correct"] else "✗")
            ], className=row_class)
            rows.append(row)
        
        # Calculate summary statistics
        total = len(results)
        changed = sum(1 for r in results if r["prediction_changed"])
        corrected = sum(1 for r in results if r["prediction_correct"])
        
        # Create the results display
        return html.Div([
            html.H4("Counterfactual Testing Results"),
            html.P([
                f"Original sentence: ",
                html.Span(sentence, className="font-italic"),
                html.Br(),
                f"True label: {true_label} ({'Positive' if true_label == 1 else 'Negative'})",
                html.Br(),
                f"Original prediction: {pred_label} ({'Positive' if pred_label == 1 else 'Negative'})",
                html.Br(),
                f"Error type: {' '.join(word.capitalize() for word in error_type.split('_'))}",
                html.Br(),
                f"Generated {total} counterfactuals, {changed} changed prediction, {corrected} corrected prediction"
            ], className="mb-3"),
            
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("#"),
                    html.Th("Type"),
                    html.Th("Counterfactual"),
                    html.Th("Transformation"),
                    html.Th("Prediction"),
                    html.Th("Confidence"),
                    html.Th("Changed?"),
                    html.Th("Correct?")
                ])),
                html.Tbody(rows)
            ], bordered=True, hover=True, responsive=True)
        ])
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in counterfactual testing: {str(e)}\n{traceback_str}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

# Add a callback to populate the counterfactual dropdown
@app.callback(
    Output('counterfactual-example-dropdown', 'options'),
    Input('error-analysis-data', 'data'),
    prevent_initial_call=True
)
def populate_counterfactual_dropdown(data):
    if not data or "high_conf_errors" not in data:
        return []
    
    options = []
    for i, error in enumerate(data["high_conf_errors"]):
        true_label = "Positive" if error["true_label"] == 1 else "Negative"
        pred_label = "Positive" if error["pred_label"] == 1 else "Negative"
        
        # Create a truncated version of the sentence for the dropdown
        max_length = 50
        sentence = error["sentence"]
        if len(sentence) > max_length:
            sentence = sentence[:max_length] + "..."
            
        options.append({
            "label": f"{i+1}. {sentence} (True: {true_label}, Pred: {pred_label})",
            "value": str(i)
        })
    
    return options

# Add this function for similarity-based error exploration
def analyze_error_similarity(error_examples, n_clusters=5, n_neighbors=5):
    """
    Analyze similarity between error examples using TF-IDF and clustering
    
    Args:
        error_examples: List of error examples
        n_clusters: Number of clusters for KMeans
        n_neighbors: Number of neighbors to find for each example
        
    Returns:
        Dictionary containing similarity analysis results
    """
    if not error_examples or len(error_examples) < 5:
        return {"error": "Not enough error examples for similarity analysis"}
    
    try:
        # Extract sentences from error examples
        sentences = [ex["sentence"] for ex in error_examples]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find nearest neighbors for each example
        nearest_neighbors = {}
        for i, ex in enumerate(error_examples):
            # Get indices of top N+1 similar examples (including self)
            similar_indices = similarity_matrix[i].argsort()[::-1][:n_neighbors+1]
            # Remove self from neighbors
            similar_indices = similar_indices[similar_indices != i][:n_neighbors]
            
            neighbors = []
            for idx in similar_indices:
                neighbors.append({
                    "sentence": error_examples[idx]["sentence"],
                    "true_label": error_examples[idx]["true_label"],
                    "pred_label": error_examples[idx]["pred_label"],
                    "similarity_score": similarity_matrix[i][idx]
                })
            
            nearest_neighbors[i] = neighbors
        
        # Perform clustering
        if len(error_examples) >= n_clusters:
            kmeans = KMeans(n_clusters=min(n_clusters, len(error_examples)), random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
        else:
            # If too few examples, assign all to one cluster
            clusters = [0] * len(error_examples)
        
        # Get top terms for each cluster
        cluster_terms = {}
        if hasattr(kmeans, 'cluster_centers_'):
            for i in range(n_clusters):
                # Get indices of examples in this cluster
                cluster_examples = [idx for idx, label in enumerate(clusters) if label == i]
                if not cluster_examples:
                    continue
                    
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                ordered_terms = []
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top 10 terms for this cluster
                top_indices = cluster_center.argsort()[::-1][:10]
                for idx in top_indices:
                    ordered_terms.append({
                        "term": feature_names[idx],
                        "weight": cluster_center[idx]
                    })
                
                cluster_terms[i] = ordered_terms
        
        # Create 2D visualization using t-SNE
        if len(error_examples) >= 3:  # t-SNE needs at least 3 samples
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(error_examples)-1))
            tsne_result = tsne.fit_transform(tfidf_matrix.toarray())
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Get colormap with distinct colors
            colors = list(mcolors.TABLEAU_COLORS.values())
            if n_clusters > len(colors):
                colors = colors * (n_clusters // len(colors) + 1)
            
            # Plot points
            for i, (x, y) in enumerate(tsne_result):
                cluster_label = clusters[i]
                plt.scatter(x, y, color=colors[cluster_label], alpha=0.7, s=50)
                plt.annotate(str(i), (x, y), fontsize=9)
            
            # Add legend
            for i in range(n_clusters):
                plt.scatter([], [], color=colors[i], label=f'Cluster {i}')
            plt.legend()
            
            plt.title('t-SNE visualization of error clusters')
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
            plt.tight_layout()
            
            # Save plot to memory
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            tsne_plot = f'data:image/png;base64,{img_str}'
        else:
            tsne_plot = None
        
        # Prepare result
        result = {
            "nearest_neighbors": nearest_neighbors,
            "clusters": clusters.tolist() if hasattr(clusters, 'tolist') else clusters,
            "cluster_terms": cluster_terms,
            "tsne_plot": tsne_plot
        }
        
        return result
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in similarity analysis: {str(e)}\n{traceback_str}")
        return {"error": str(e)}

# Add callback for similarity analysis
@app.callback(
    Output('similarity-analysis-results', 'children'),
    Input('analyze-similarity-button', 'n_clicks'),
    [State('error-analysis-data', 'data'),
     State('num-clusters-input', 'value')],
    prevent_initial_call=True
)
def update_similarity_analysis(n_clicks, data, n_clusters):
    if not n_clicks or not data:
        return html.Div("Click 'Analyze Error Similarity' to find patterns in error cases.")
    
    try:
        # Get high confidence errors
        high_conf_errors = data.get("high_conf_errors", [])
        if not high_conf_errors:
            return html.Div("No high confidence errors found for similarity analysis.", className="text-danger")
        
        # Set default number of clusters if not provided
        if not n_clusters or n_clusters < 2:
            n_clusters = min(5, len(high_conf_errors))
        
        # Perform similarity analysis
        similarity_results = analyze_error_similarity(high_conf_errors, n_clusters=n_clusters)
        
        if "error" in similarity_results:
            return html.Div(f"Error in similarity analysis: {similarity_results['error']}", className="text-danger")
        
        # Create visualization of clusters
        if similarity_results["tsne_plot"]:
            cluster_viz = html.Div([
                html.H5("Error Clusters Visualization", className="mt-4"),
                html.P("This t-SNE plot shows how errors cluster together based on text similarity. Numbers indicate error indices."),
                html.Img(src=similarity_results["tsne_plot"], style={"width": "100%", "max-width": "800px"})
            ])
        else:
            cluster_viz = html.Div("Not enough examples for t-SNE visualization.")
        
        # Create table of cluster terms
        cluster_terms_cards = []
        for cluster_id, terms in similarity_results.get("cluster_terms", {}).items():
            if not terms:
                continue
                
            term_rows = []
            for term_data in terms:
                term_rows.append(html.Tr([
                    html.Td(term_data["term"]),
                    html.Td(f"{term_data['weight']:.4f}")
                ]))
            
            cluster_card = dbc.Card([
                dbc.CardHeader(f"Cluster {cluster_id} Key Terms"),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Term"),
                            html.Th("Weight")
                        ])),
                        html.Tbody(term_rows)
                    ], bordered=True, size="sm")
                ])
            ], className="mb-3")
            
            cluster_terms_cards.append(cluster_card)
        
        # Create nearest neighbors explorer
        neighbors_cards = []
        for i, neighbors in similarity_results.get("nearest_neighbors", {}).items():
            if i >= len(high_conf_errors) or not neighbors:
                continue
                
            example = high_conf_errors[i]
            true_label = "Positive" if example["true_label"] == 1 else "Negative"
            pred_label = "Positive" if example["pred_label"] == 1 else "Negative"
            cluster_id = similarity_results["clusters"][i]
            
            neighbor_rows = []
            for j, neighbor in enumerate(neighbors):
                n_true_label = "Positive" if neighbor["true_label"] == 1 else "Negative"
                n_pred_label = "Positive" if neighbor["pred_label"] == 1 else "Negative"
                
                neighbor_rows.append(html.Tr([
                    html.Td(j+1),
                    html.Td(neighbor["sentence"]),
                    html.Td(n_true_label),
                    html.Td(n_pred_label),
                    html.Td(f"{neighbor['similarity_score']:.4f}")
                ]))
            
            card = dbc.Card([
                dbc.CardHeader([
                    f"Error #{i+1} (Cluster {cluster_id})",
                    dbc.Badge("Similar Errors", color="info", className="ms-2")
                ]),
                dbc.CardBody([
                    html.P([
                        html.Strong("Sentence: "),
                        example["sentence"]
                    ]),
                    html.P([
                        html.Strong("True label: "),
                        true_label,
                        html.Strong(" Predicted: "),
                        pred_label,
                        html.Strong(" Confidence: "),
                        f"{example['confidence']:.4f}"
                    ]),
                    html.H6("Similar Error Cases:", className="mt-3"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("#"),
                            html.Th("Sentence"),
                            html.Th("True"),
                            html.Th("Pred"),
                            html.Th("Similarity")
                        ])),
                        html.Tbody(neighbor_rows)
                    ], bordered=True, size="sm", responsive=True)
                ])
            ], className="mb-4")
            
            neighbors_cards.append(card)
        
        # Limit to first 5 examples to avoid overwhelming the UI
        if len(neighbors_cards) > 5:
            neighbors_cards = neighbors_cards[:5]
            neighbors_cards.append(html.P("Showing only first 5 examples. Run analysis with fewer samples for more detailed results."))
        
        # Create tabs for different visualizations
        similarity_tabs = dbc.Tabs([
            dbc.Tab([
                html.Div([
                    html.P("This visualization shows how error cases cluster together based on text similarity.", className="mt-3"),
                    cluster_viz,
                    html.Div(cluster_terms_cards, className="mt-4")
                ])
            ], label="Error Clusters"),
            dbc.Tab([
                html.Div([
                    html.P("This view shows similar error cases for each example, helping identify common patterns.", className="mt-3"),
                    html.Div(neighbors_cards)
                ], className="mt-3")
            ], label="Nearest Neighbors")
        ])
        
        return html.Div([
            html.H4("Similarity-Based Error Analysis"),
            html.P([
                f"Analyzed {len(high_conf_errors)} error cases and found {len(similarity_results.get('cluster_terms', {}))} clusters. ",
                "This analysis helps identify groups of similar errors that might share common causes."
            ], className="mb-3"),
            similarity_tabs
        ])
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in similarity analysis: {str(e)}\n{traceback_str}")
        return html.Div(f"Error: {str(e)}", className="text-danger")


# === Main entry point ===
if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Start the app
    logging.info(f"Starting server on port {port}")
    app.run_server(debug=False, host='0.0.0.0', port=port)
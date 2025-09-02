"""
NER entity details visualization page.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

from models.api import model_api

def create_layout():
    """
    Create the layout for the NER entity details visualization page.
    
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
                            html.I(className="fas fa-tags me-2 text-primary"),
                            "Entity Details Visualization"
                        ], className="mb-2"),
                        html.P("Analyze named entities in text with detailed information and confidence scores", 
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
                                "Text Input"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Label("Enter text to analyze for named entities:", className="fw-bold mb-2"),
                            dbc.Textarea(
                                id="ner-entity-input", 
                                value="John Smith works at Google in New York and was born on January 15, 1985.",
                                className="mb-4",
                                style={"height": "120px"},
                                placeholder="Enter your text here..."
                            ),
                            
                            # Generate button
                            dbc.Button([
                                html.I(className="fas fa-search me-2"),
                                "Analyze Entities"
                            ], id="ner-entity-analyze-button", color="primary", size="lg", className="w-100"),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            # Results Section
            dbc.Row([
                dbc.Col([
                    dbc.Spinner([
                        html.Div(id="ner-entity-error-output", className="text-danger mb-3"),
                        html.Div(id="ner-entity-results")
                    ], color="primary", type="border", fullscreen=False),
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
                                    "About Named Entity Recognition"
                                ], className="mb-0")
                            ]),
                            dbc.CardBody([
                                html.P([
                                    html.Strong("Named Entity Recognition (NER)"), 
                                    " identifies and classifies named entities in text into predefined categories."
                                ]),
                                html.Ul([
                                    html.Li([html.Strong("PERSON:"), " Names of people"]),
                                    html.Li([html.Strong("ORG:"), " Organizations, companies, agencies"]),
                                    html.Li([html.Strong("LOC:"), " Locations, countries, cities"]),
                                    html.Li([html.Strong("MISC:"), " Miscellaneous entities"]),
                                ]),
                                html.P([
                                    "The visualization shows identified entities with their types, confidence scores, ",
                                    "and positions in the text. Higher confidence scores indicate more certain predictions."
                                ], className="text-muted mb-0")
                            ]),
                        ], className="border-primary"),
                    ], id="ner-entity-info-collapse", is_open=False),
                    
                    dbc.Button([
                        html.I(className="fas fa-question-circle me-2"),
                        "Learn More About NER"
                    ], id="ner-entity-info-toggle", color="outline-primary", size="sm", className="mt-3")
                ], width=12),
            ], className="mt-4"),
            
        ], fluid=True, className="p-3"),
    ])
    
    return layout

@callback(
    [Output("ner-entity-results", "children"),
     Output("ner-entity-error-output", "children")],
    [Input("ner-entity-analyze-button", "n_clicks")],
    [State("ner-entity-input", "value")],
    prevent_initial_call=True
)
def update_ner_entity_analysis(n_clicks, input_text):
    if not n_clicks:
        return "", ""
    
    if not input_text or not input_text.strip():
        return "", "Please enter some text to analyze."

    try:
        # Check if model is loaded
        if not model_api.selected_model_path:
            return "", "Error: No model selected. Please select a model from the main interface first."
        
        # Get entities from the model
        result = model_api.analyze_entities(input_text)
        
        if not result or "entities" not in result:
            return "", "Error: Could not extract entities from the model."

        entities = result["entities"]
        
        if not entities:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle fa-2x text-info mb-3"),
                    html.H5("No Entities Found", className="text-info"),
                    html.P("No named entities were detected in the provided text.")
                ], className="text-center py-5")
            ]), ""

        # Create highlighted text
        highlighted_text = create_highlighted_text(input_text, entities)
        
        # Create entity table
        entity_table = create_entity_table(entities)
        
        # Create entity statistics
        entity_stats = create_entity_statistics(entities)
        
        # Create entity chart
        entity_chart = create_entity_chart(entities)
        
        results = html.Div([
            # Highlighted text section
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-highlighter me-2"),
                        "Text with Highlighted Entities"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    html.Div(highlighted_text, className="entity-highlighted-text"),
                    html.Hr(),
                    html.Small([
                        "Entities are highlighted with different colors based on their types. ",
                        "Hover over highlighted text to see entity details."
                    ], className="text-muted")
                ])
            ], className="mb-4"),
            
            # Statistics section
            dbc.Row([
                dbc.Col([entity_stats], width=12, md=6),
                dbc.Col([entity_chart], width=12, md=6)
            ], className="mb-4"),
            
            # Entity table section
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-table me-2"),
                        "Detailed Entity Information"
                    ], className="mb-0")
                ]),
                dbc.CardBody([entity_table])
            ])
        ])
        
        return results, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if "meta tensor" in str(e).lower():
            error_msg = "Error: Model loading issue. Please try reloading the model or use a different model."
        return "", error_msg

def create_highlighted_text(text, entities):
    """Create highlighted text with entity markup."""
    if not entities:
        return html.P(text)
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))
    
    # Color mapping for entity types
    entity_colors = {
        "PERSON": "primary",
        "ORG": "success", 
        "LOC": "info",
        "MISC": "warning",
        "PER": "primary",
        "B-PER": "primary",
        "I-PER": "primary"
    }
    
    highlighted_parts = []
    last_end = 0
    
    for entity in sorted_entities:
        start = entity.get("start", 0)
        end = entity.get("end", start + len(entity.get("word", "")))
        entity_text = entity.get("word", "")
        entity_label = entity.get("entity", "MISC")
        confidence = entity.get("score", 0)
        
        # Add text before entity
        if start > last_end:
            highlighted_parts.append(text[last_end:start])
        
        # Add highlighted entity
        color = entity_colors.get(entity_label.replace("B-", "").replace("I-", ""), "secondary")
        highlighted_parts.append(
            html.Mark(
                entity_text,
                className=f"entity-highlight entity-{color}",
                title=f"{entity_label}: {confidence:.3f}",
                style={
                    "backgroundColor": f"var(--bs-{color})",
                    "color": "white" if color in ["primary", "success"] else "black",
                    "padding": "2px 4px",
                    "borderRadius": "3px",
                    "margin": "0 1px"
                }
            )
        )
        
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        highlighted_parts.append(text[last_end:])
    
    return html.P(highlighted_parts, style={"fontSize": "1.1rem", "lineHeight": "1.8"})

def create_entity_table(entities):
    """Create a table showing entity details."""
    rows = []
    for i, entity in enumerate(entities, 1):
        entity_text = entity.get("word", "")
        entity_label = entity.get("entity", "MISC")
        confidence = entity.get("score", 0)
        start = entity.get("start", 0)
        end = entity.get("end", start + len(entity_text))
        
        rows.append(
            html.Tr([
                html.Td(i),
                html.Td(entity_text),
                html.Td(entity_label.replace("B-", "").replace("I-", "")),
                html.Td(f"{confidence:.3f}"),
                html.Td(f"{start}-{end}")
            ])
        )
    
    return dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("#"),
                    html.Th("Entity"),
                    html.Th("Type"),
                    html.Th("Confidence"),
                    html.Th("Position")
                ])
            ),
            html.Tbody(rows)
        ],
        bordered=True,
        hover=True,
        striped=True,
        responsive=True
    )

def create_entity_statistics(entities):
    """Create entity statistics card."""
    if not entities:
        return html.Div()
    
    # Count entities by type
    entity_counts = {}
    total_confidence = 0
    
    for entity in entities:
        entity_type = entity.get("entity", "MISC").replace("B-", "").replace("I-", "")
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        total_confidence += entity.get("score", 0)
    
    avg_confidence = total_confidence / len(entities) if entities else 0
    
    stats_items = []
    for entity_type, count in entity_counts.items():
        stats_items.append(
            html.Li([
                html.Strong(f"{entity_type}: "),
                f"{count} entities"
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H6([
                html.I(className="fas fa-chart-bar me-2"),
                "Entity Statistics"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P([
                html.Strong("Total Entities: "),
                f"{len(entities)}"
            ]),
            html.P([
                html.Strong("Average Confidence: "),
                f"{avg_confidence:.3f}"
            ]),
            html.P([html.Strong("Entity Types:")]),
            html.Ul(stats_items)
        ])
    ])

def create_entity_chart(entities):
    """Create entity type distribution chart."""
    if not entities:
        return html.Div()
    
    # Count entities by type
    entity_counts = {}
    for entity in entities:
        entity_type = entity.get("entity", "MISC").replace("B-", "").replace("I-", "")
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    if not entity_counts:
        return html.Div()
    
    # Create pie chart
    fig = px.pie(
        values=list(entity_counts.values()),
        names=list(entity_counts.keys()),
        title="Entity Type Distribution"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H6([
                html.I(className="fas fa-chart-pie me-2"),
                "Entity Distribution"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    ])

# Callback for info toggle
@callback(
    Output("ner-entity-info-collapse", "is_open"),
    Input("ner-entity-info-toggle", "n_clicks"),
    State("ner-entity-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_ner_entity_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open
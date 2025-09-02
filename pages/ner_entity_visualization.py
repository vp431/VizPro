"""
NER entity visualization page with enhanced entity marking and details.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import logging

from models.api import model_api

logger = logging.getLogger(__name__)

def create_layout():
    """
    Create modern layout for NER entity visualization inspired by sentiment_lime.py.
    Professional design with enhanced styling and better user experience.
    
    Returns:
        A Dash layout object
    """
    layout = html.Div([
        # Main NER Container with modern styling
        html.Div([
            # Header Section - Modern Design
            html.Div([
                html.H4([
                    html.I(className="fas fa-tags", style={"marginRight": "0.5rem"}),
                    "Named Entity Recognition"
                ]),
                html.P("Identify and visualize entities in text with advanced NER models", 
                       className="ner-header-subtitle")
            ], className="ner-header"),
            
            # Input Section - Modern Design
            html.Div([
                html.Label("Enter text to analyze:", className="ner-input-label"),
                dcc.Textarea(
                    id="ner-input", 
                    value="Apple Inc. is planning to open a new store in New York City next month. CEO Tim Cook announced this during his visit to Berlin, Germany.",
                    className="ner-textarea",
                    placeholder="Enter your text here..."
                ),
                
                # Generate Button - Modern
                dbc.Button([
                    html.I(className="fas fa-search me-2"),
                    "Analyze Entities"
                ], id="ner-analyze-button", color="primary", size="lg", className="w-100"),
                
                # Error Output
                html.Div(id="ner-error-output", className="ner-error")
            ], className="ner-input-section"),
            
            # Results Section - Modern Layout
            html.Div([
                html.Div(id="ner-loading-container", children=[], className="ner-loading"),
                
                # Highlighted Text Results
                html.Div([
                    html.Div("Entity Analysis Results", className="ner-result-header"),
                    html.Div([
                        html.Div([
                            html.H5("Highlighted Text", className="ner-section-title"),
                            html.Div(id="ner-highlighted-text", className="ner-highlighted-container")
                        ], className="ner-text-section"),
                        
                        # Analytics Row - Side by Side
                        dbc.Row([
                            dbc.Col([
                                html.H5("Entity Distribution", className="ner-section-title"),
                                html.Div(id="ner-entity-pie-chart", className="ner-chart-container")
                            ], width=6, className="ner-chart-col"),
                            
                            dbc.Col([
                                html.H5("Entity Summary", className="ner-section-title"),
                                html.Div(id="ner-entity-table", className="ner-table-container")
                            ], width=6, className="ner-table-col")
                        ], className="ner-analytics-row", id="ner-analytics-section", style={"display": "none"})
                        
                    ], className="ner-result-body")
                ], id="ner-result-card", className="ner-result-card", style={"display": "none"}),
            ], className="ner-results-container"),
            
            # Info Section - Modern Design
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "Learn More About Named Entity Recognition"
                ], id="ner-info-toggle", color="outline-primary", size="sm", className="mt-3"),
                
                dbc.Collapse([
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle", style={"marginRight": "0.5rem"}),
                            "About Named Entity Recognition"
                        ]),
                        html.P([
                            html.Strong("Named Entity Recognition (NER)"), 
                            " identifies and classifies named entities in text into predefined categories."
                        ]),
                        html.Div([
                            html.Span("PERSON", className="entity-per me-2"),
                            html.Span("ORGANIZATION", className="entity-org me-2"),
                            html.Span("LOCATION", className="entity-loc me-2"),
                            html.Span("MISCELLANEOUS", className="entity-misc")
                        ], className="entity-legend-examples mb-3"),
                        html.P([
                            "Each entity type is highlighted with distinct colors. Hover over entities to see confidence scores. ",
                            "The visualization includes distribution charts and detailed analysis."
                        ], className="ner-explanation")
                    ], className="ner-info-content")
                ], id="ner-info-collapse", is_open=False),
            ], className="ner-info-section"),
            
        ], className="ner-modal-container"),
    ])
    
    return layout

def create_entity_marked_text(text, entities):
    """
    Create text with highlighted entities.
    
    Args:
        text: Original text
        entities: List of entity dictionaries
        
    Returns:
        HTML elements with marked entities
    """
    if not entities:
        return html.P(text)
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))
    
    # Entity type colors
    entity_colors = {
        "PERSON": "success",
        "PER": "success",
        "ORG": "primary", 
        "LOC": "info",
        "MISC": "warning",
        "GPE": "info",  # Geopolitical entity
        "DATE": "secondary",
        "TIME": "secondary",
        "MONEY": "warning",
        "PERCENT": "warning"
    }
    
    marked_elements = []
    last_end = 0
    
    try:
        for entity in sorted_entities:
            start = entity.get("start", 0)
            end = entity.get("end", start)
            entity_text = entity.get("word", entity.get("text", ""))
            entity_label = entity.get("entity", entity.get("label", "MISC"))
            confidence = entity.get("confidence", entity.get("score", 0))
            
            # Add text before entity
            if start > last_end:
                marked_elements.append(text[last_end:start])
            
            # Add highlighted entity
            color_class = entity_colors.get(entity_label, "secondary")
            marked_elements.append(
                html.Mark([
                    entity_text,
                    html.Sup(f" {entity_label}", className="small text-muted")
                ], 
                className=f"bg-{color_class} bg-opacity-25 border border-{color_class} rounded px-1",
                title=f"{entity_label}: {confidence:.2f}")
            )
            
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            marked_elements.append(text[last_end:])
            
    except Exception as e:
        logger.error(f"Error creating marked text: {str(e)}")
        return html.P(text)
    
    return html.P(marked_elements, className="fs-5 lh-lg")

def create_compact_entity_marked_text(text, entities):
    """Create improved highlighted text for entities with better visibility."""
    if not entities:
        return html.Span(text, style={"fontSize": "0.85rem", "color": "#495057"})
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))
    
    marked_elements = []
    last_end = 0
    
    try:
        for entity in sorted_entities:
            start = entity.get("start", 0)
            end = entity.get("end", start)
            entity_text = entity.get("word", entity.get("text", ""))
            entity_label = entity.get("entity", entity.get("label", "MISC"))
            confidence = entity.get("confidence", entity.get("score", 0))
            
            # Add text before entity
            if start > last_end:
                marked_elements.append(
                    html.Span(text[last_end:start], style={"color": "#495057"})
                )
            
            # Add highlighted entity with enhanced visibility
            marked_elements.append(
                html.Span([
                    html.Span(
                        entity_text,
                        className=f"entity-{entity_label.lower()}",
                        title=f"{entity_label}: {confidence:.2f}",
                        style={
                            "padding": "3px 8px",
                            "borderRadius": "6px",
                            "fontWeight": "600",
                            "fontSize": "0.85rem",
                            "marginRight": "4px",
                            "marginLeft": "1px",
                            "display": "inline-block",
                            "boxShadow": "0 1px 3px rgba(0,0,0,0.2)",
                            "cursor": "pointer"
                        }
                    ),
                    html.Sup(
                        entity_label,
                        style={
                            "fontSize": "0.6rem",
                            "color": "#495057",
                            "fontWeight": "500",
                            "marginLeft": "2px"
                        }
                    )
                ], title=f"{entity_label}: {confidence:.2f}")
            )
            
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            marked_elements.append(
                html.Span(text[last_end:], style={"color": "#495057"})
            )
            
    except Exception as e:
        logger.error(f"Error creating compact marked text: {str(e)}")
        return html.Span(text, style={"fontSize": "0.85rem", "color": "#495057"})
    
    return html.Div(marked_elements, style={
        "fontSize": "0.85rem", 
        "lineHeight": "1.8",
        "wordSpacing": "1px",
        "letterSpacing": "0.3px"
    })

def create_compact_entity_table(entities):
    """Create compact entity table."""
    if not entities:
        return html.Div()
    
    # Group entities by type for summary
    entity_groups = {}
    for entity in entities:
        entity_type = entity.get("entity", entity.get("label", "MISC"))
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)
    
    # Create compact summary rows
    summary_rows = []
    for entity_type, type_entities in entity_groups.items():
        count = len(type_entities)
        avg_confidence = sum(e.get("confidence", e.get("score", 0)) for e in type_entities) / count
        
        summary_rows.append(
            html.Tr([
                html.Td(html.Span(entity_type, className=f"entity-{entity_type.lower()}", 
                                style={"fontSize": "0.7rem", "padding": "1px 4px"})),
                html.Td(count, style={"fontSize": "0.75rem", "textAlign": "center"}),
                html.Td(f"{avg_confidence:.2f}", style={"fontSize": "0.75rem", "textAlign": "center"})
            ], style={"height": "1.8rem"})
        )
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Type", style={"fontSize": "0.75rem", "padding": "0.2rem"}),
                html.Th("Count", style={"fontSize": "0.75rem", "padding": "0.2rem", "textAlign": "center"}),
                html.Th("Avg", style={"fontSize": "0.75rem", "padding": "0.2rem", "textAlign": "center"})
            ])
        ]),
        html.Tbody(summary_rows)
    ], style={"width": "100%", "border": "1px solid #dee2e6", "borderRadius": "3px"})

def create_modern_entity_table(entities):
    """Create modern entity table with better styling."""
    if not entities:
        return html.Div([
            html.I(className="fas fa-info-circle text-info me-2"),
            html.Span("No entities detected.")
        ], className="text-center text-muted py-3")
    
    # Group entities by type for summary
    entity_groups = {}
    for entity in entities:
        entity_type = entity.get("entity", entity.get("label", "MISC"))
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)
    
    # Create modern summary cards
    summary_cards = []
    for entity_type, type_entities in entity_groups.items():
        count = len(type_entities)
        avg_confidence = sum(e.get("confidence", e.get("score", 0)) for e in type_entities) / count
        
        card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(entity_type, className=f"entity-{entity_type.lower()}", 
                            style={"fontSize": "0.75rem", "fontWeight": "600"}),
                ], className="mb-2"),
                html.Div([
                    html.H6(count, className="mb-1 text-primary"),
                    html.Small(f"Avg: {avg_confidence:.2f}", className="text-muted")
                ])
            ], className="text-center py-2")
        ], className="h-100 border-0 shadow-sm")
        summary_cards.append(card)
    
    return html.Div([
        dbc.Row([
            dbc.Col(card, width=6 if len(summary_cards) <= 2 else 4, className="mb-2")
            for card in summary_cards
        ])
    ])

def create_entity_pie_chart(entities):
    """Create compact pie chart for entity distribution."""
    if not entities:
        return html.Div()
    
    # Group entities by type
    entity_counts = {}
    for entity in entities:
        entity_type = entity.get("entity", entity.get("label", "MISC"))
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Create pie chart
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame([
        {"Entity Type": entity_type, "Count": count}
        for entity_type, count in entity_counts.items()
    ])
    
    # Use dark colors for consistency
    colors = {
        "PERSON": "#1e3a8a", "PER": "#1e3a8a",
        "ORG": "#166534", "ORGANIZATION": "#166534",
        "LOC": "#991b1b", "LOCATION": "#991b1b", 
        "MISC": "#581c87", "MISCELLANEOUS": "#581c87"
    }
    
    fig = px.pie(df, values="Count", names="Entity Type", 
                 color="Entity Type", color_discrete_map=colors)
    
    fig.update_layout(
        height=200,
        margin=dict(l=15, r=15, t=25, b=15),
        font=dict(size=10),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(size=9))
    )
    fig.update_traces(textinfo='label+percent', textfont_size=9)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={"height": "200px"})

def create_entity_details_table(entities):
    """
    Create a detailed table of entities.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        HTML table with entity details
    """
    if not entities:
        return html.Div([
            html.I(className="fas fa-info-circle text-info me-2"),
            html.P("No entities found in the text.")
        ])
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        entity_type = entity.get("entity", entity.get("label", "MISC"))
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)
    
    # Create summary cards
    summary_cards = []
    for entity_type, type_entities in entity_groups.items():
        count = len(type_entities)
        avg_confidence = sum(e.get("confidence", e.get("score", 0)) for e in type_entities) / count
        
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(count, className="text-primary mb-1"),
                    html.P(entity_type, className="mb-1"),
                    html.Small(f"Avg: {avg_confidence:.2f}", className="text-muted")
                ], className="text-center")
            ], className="border-primary")
        ], width=6, md=3, className="mb-3")
        summary_cards.append(card)
    
    # Create detailed table
    table_rows = []
    for i, entity in enumerate(entities, 1):
        entity_text = entity.get("word", entity.get("text", ""))
        entity_type = entity.get("entity", entity.get("label", "MISC"))
        confidence = entity.get("confidence", entity.get("score", 0))
        start = entity.get("start", 0)
        end = entity.get("end", start)
        
        table_rows.append(
            html.Tr([
                html.Td(i),
                html.Td(entity_text),
                html.Td([
                    dbc.Badge(entity_type, color="primary", className="me-1"),
                ]),
                html.Td(f"{confidence:.3f}"),
                html.Td(f"{start}-{end}")
            ])
        )
    
    detailed_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("#"),
                html.Th("Entity"),
                html.Th("Type"),
                html.Th("Confidence"),
                html.Th("Position")
            ])
        ]),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, striped=True, responsive=True)
    
    return html.Div([
        # Summary cards
        dbc.Row(summary_cards, className="mb-4"),
        
        # Detailed table
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-table me-2"),
                    "Entity Details"
                ], className="mb-0")
            ]),
            dbc.CardBody([detailed_table])
        ])
    ])

@callback(
    [Output("ner-highlighted-text", "children"),
     Output("ner-entity-table", "children"),
     Output("ner-entity-pie-chart", "children"),
     Output("ner-result-card", "style"),
     Output("ner-analytics-section", "style"),
     Output("ner-error-output", "children"),
     Output("ner-loading-container", "children")],
    [Input("ner-analyze-button", "n_clicks")],
    [State("ner-input", "value")],
    prevent_initial_call=True
)
def analyze_ner_entities_modern(n_clicks, input_text):
    """Analyze entities in the input text with modern layout."""
    if not n_clicks:
        return "", "", "", {"display": "none"}, {"display": "none"}, "", ""
    
    if not input_text or not input_text.strip():
        return "", "", "", {"display": "none"}, {"display": "none"}, "Please enter some text to analyze.", ""

    # Show loading
    loading = dbc.Spinner(size="lg", color="primary")

    try:
        # Check if model is loaded
        if not model_api.selected_model_path:
            return "", "", "", {"display": "none"}, {"display": "none"}, "Error: No model selected. Please select a model from the main interface first.", ""
        
        # Get entities from the model
        result = model_api.analyze_entities(input_text)
        
        if not result or "entities" not in result:
            return "", "", "", {"display": "none"}, {"display": "none"}, "Error: Could not extract entities from the model.", ""

        entities = result["entities"]
        
        if not entities:
            no_entities_msg = html.Div([
                html.I(className="fas fa-info-circle text-info me-2"),
                html.Span("No entities found in the text.")
            ], className="text-center py-4")
            return no_entities_msg, "", "", {"display": "block"}, {"display": "none"}, "", ""

        # Create modern highlighted text
        highlighted_text = create_compact_entity_marked_text(input_text, entities)
        
        # Create modern entity table
        entity_table = create_modern_entity_table(entities)
        
        # Create pie chart
        pie_chart = create_entity_pie_chart(entities)
        
        return (highlighted_text, entity_table, pie_chart, 
                {"display": "block"}, {"display": "block"}, "", "")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if "meta tensor" in str(e).lower():
            error_msg = "Error: Model loading issue. Please try reloading the model or use a different model."
        return "", "", "", {"display": "none"}, {"display": "none"}, error_msg, ""

# Callback for info toggle
@callback(
    Output("ner-info-collapse", "is_open"),
    Input("ner-info-toggle", "n_clicks"),
    State("ner-info-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_ner_info(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open
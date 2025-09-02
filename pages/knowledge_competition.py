"""
Knowledge Competition Analysis Page
Implements fact vs counterfact knowledge competition visualization
"""
import logging
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
from models.knowledge_competition import knowledge_competition_analyzer
from components.counterfactual_visualizations import (
    create_knowledge_competition_dashboard,
    create_interactive_drill_down
)

logger = logging.getLogger(__name__)

def create_knowledge_competition_layout():
    """Create the layout for knowledge competition analysis."""
    
    # Default fact-counterfact pairs for examples
    example_pairs = knowledge_competition_analyzer.generate_fact_counterfact_pairs("capital", 3)
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4("Knowledge Competition Analysis", className="mb-0"),
                html.Small("Analyze how transformer models handle competing factual vs counterfactual information", 
                          className="text-muted")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Factual Statement:", className="fw-bold"),
                        dcc.Textarea(
                            id="fact-input",
                            placeholder="Enter a factual statement (e.g., 'The capital of France is Paris.')",
                            value=example_pairs[0][0] if example_pairs else "The capital of France is Paris.",
                            style={"width": "100%", "height": "80px"},
                            className="mb-3"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Counterfactual Statement:", className="fw-bold"),
                        dcc.Textarea(
                            id="counterfact-input",
                            placeholder="Enter a counterfactual statement (e.g., 'The capital of Italy is Paris.')",
                            value=example_pairs[0][1] if example_pairs else "The capital of Italy is Paris.",
                            style={"width": "100%", "height": "80px"},
                            className="mb-3"
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Example Pairs:", className="fw-bold"),
                        dcc.Dropdown(
                            id="example-pairs-dropdown",
                            options=[
                                {"label": f"{pair[0]} vs {pair[1]}", "value": i}
                                for i, pair in enumerate(example_pairs)
                            ],
                            placeholder="Select an example pair...",
                            className="mb-3"
                        )
                    ], width=8),
                    dbc.Col([
                        dbc.Button(
                            "Analyze Competition",
                            id="analyze-competition-btn",
                            color="primary",
                            size="lg",
                            className="w-100",
                            disabled=False
                        )
                    ], width=4)
                ])
            ])
        ], className="mb-4"),
        
        # Store for analysis results data
        dcc.Store(id="competition-data-store"),
        
        # Loading component
        dcc.Loading(
            id="competition-loading",
            type="cube",
            color="#1f77b4",
            children=[
                html.Div(id="competition-analysis-results", className="mt-4")
            ]
        )
    ])


@callback(
    [Output("fact-input", "value"),
     Output("counterfact-input", "value")],
    [Input("example-pairs-dropdown", "value")]
)
def update_inputs_from_example(selected_example):
    """Update input fields when an example is selected."""
    if selected_example is None:
        return no_update, no_update
    
    example_pairs = knowledge_competition_analyzer.generate_fact_counterfact_pairs("capital", 5)
    if selected_example < len(example_pairs):
        fact, counterfact = example_pairs[selected_example]
        return fact, counterfact
    
    return no_update, no_update


@callback(
    [Output("competition-analysis-results", "children"),
     Output("competition-data-store", "data")],
    [Input("analyze-competition-btn", "n_clicks")],
    [State("fact-input", "value"),
     State("counterfact-input", "value")]
)
def perform_competition_analysis(n_clicks, fact_text, counterfact_text):
    """Perform knowledge competition analysis and display results."""
    if not n_clicks:
        return html.Div([
            dbc.Alert(
                "Enter factual and counterfactual statements above, then click 'Analyze Competition' to begin analysis.",
                color="info",
                className="mt-4"
            )
        ]), {}
    
    if not fact_text or not counterfact_text:
        return html.Div([
            dbc.Alert(
                "Please enter both factual and counterfactual statements.",
                color="warning",
                className="mt-4"
            )
        ]), {}
    
    try:
        # Import model API to use for analysis
        from models.api import model_api
        
        # Perform competition analysis using model API
        analysis_results = model_api.analyze_knowledge_competition(
            fact_text.strip(), counterfact_text.strip()
        )
        
        if "error" in analysis_results:
            return html.Div([
                dbc.Alert(
                    f"Error in analysis: {analysis_results['error']}",
                    color="danger",
                    className="mt-4"
                )
            ]), {}
        
        # Create the complete dashboard
        dashboard = create_knowledge_competition_dashboard(analysis_results)
        
        # Return both dashboard and raw data for callbacks
        return dashboard, analysis_results
        
    except Exception as e:
        logger.error(f"Error in competition analysis callback: {e}")
        return html.Div([
            dbc.Alert(
                f"An error occurred during analysis: {str(e)}",
                color="danger",
                className="mt-4"
            )
        ]), {}


@callback(
    Output("drill-down-chart", "figure"),
    [Input("drill-down-layer-slider", "value")],
    [State("competition-data-store", "data")]
)
def update_drill_down_chart(selected_layer, stored_data):
    """Update the drill-down chart when layer selection changes."""
    if selected_layer is None or not stored_data:
        return create_interactive_drill_down({}, 0)
    
    # Extract layerwise competition data
    layerwise_data = stored_data.get("layerwise_competition", {})
    
    return create_interactive_drill_down(layerwise_data, selected_layer)


def handle_knowledge_competition_analysis(fact_text: str, counterfact_text: str):
    """
    Handle knowledge competition analysis request.
    
    Args:
        fact_text: Factual statement
        counterfact_text: Counterfactual statement
        
    Returns:
        Analysis results dictionary
    """
    try:
        # Perform the analysis
        results = knowledge_competition_analyzer.analyze_fact_counterfact_competition(
            fact_text, counterfact_text
        )
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in knowledge competition analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }

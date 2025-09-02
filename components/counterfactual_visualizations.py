"""
Counterfactual Analysis Visualization Functions
Implements Panel A-F for knowledge competition analysis
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_input_setup_panel(fact_text: str, counterfact_text: str, current_mode: str = "fact") -> html.Div:
    """
    Panel A - Input & Setup: Fact vs. counterfactual prompts with toggle switch.
    
    Args:
        fact_text: Factual statement
        counterfact_text: Counterfactual statement
        current_mode: Current mode ("fact" or "counterfact")
        
    Returns:
        HTML div component
    """
    # Identify different words between fact and counterfact
    fact_words = fact_text.split()
    counterfact_words = counterfact_text.split()
    
    # Create highlighted text displays
    fact_display = []
    counterfact_display = []
    
    # Simple word-by-word comparison
    max_len = max(len(fact_words), len(counterfact_words))
    
    for i in range(max_len):
        fact_word = fact_words[i] if i < len(fact_words) else ""
        counterfact_word = counterfact_words[i] if i < len(counterfact_words) else ""
        
        # Highlight different words
        if fact_word != counterfact_word and fact_word and counterfact_word:
            fact_display.append(html.Span(fact_word + " ", style={
                "background-color": "#e3f2fd", 
                "padding": "2px 4px", 
                "border-radius": "3px",
                "border": "1px solid #2196f3"
            }))
            counterfact_display.append(html.Span(counterfact_word + " ", style={
                "background-color": "#ffebee", 
                "padding": "2px 4px", 
                "border-radius": "3px",
                "border": "1px solid #f44336"
            }))
        else:
            fact_display.append(fact_word + " ")
            counterfact_display.append(counterfact_word + " ")
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5("Panel A - Input & Setup", className="mb-0"),
                html.Small("Compare factual vs counterfactual statements", className="text-muted")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Factual Statement:", className="text-primary"),
                        html.Div(fact_display, className="p-3 border rounded mb-3", 
                                style={"background-color": "#f8f9fa", "min-height": "60px"})
                    ], width=6),
                    dbc.Col([
                        html.H6("Counterfactual Statement:", className="text-danger"),
                        html.Div(counterfact_display, className="p-3 border rounded mb-3",
                                style={"background-color": "#f8f9fa", "min-height": "60px"})
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("View Mode:", className="fw-bold"),
                        dbc.Switch(
                            id="fact-counterfact-toggle",
                            label="Fact ↔ Counterfact",
                            value=current_mode == "counterfact",
                            className="mb-2"
                        ),
                        html.Small([
                            html.Span("Blue = Factual", className="text-primary me-3"),
                            html.Span("Red = Counterfactual", className="text-danger")
                        ], className="text-muted")
                    ], width=12)
                ])
            ])
        ], className="mb-4")
    ])


def create_layerwise_competition_heatmap(competition_data: dict) -> go.Figure:
    """
    Panel B - Layerwise Competition Heatmap: Layer/head competition visualization.
    
    Args:
        competition_data: Dictionary containing competition matrix and labels
        
    Returns:
        Plotly figure object
    """
    if not competition_data or "competition_matrix" not in competition_data:
        return go.Figure().add_annotation(text="No competition data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    matrix = np.array(competition_data["competition_matrix"])
    layer_labels = competition_data.get("layer_labels", [f"Layer {i+1}" for i in range(matrix.shape[0])])
    head_labels = competition_data.get("head_labels", [f"Head {i+1}" for i in range(matrix.shape[1])])
    
    # Create custom colorscale: blue for fact dominance, red for counterfact dominance
    colorscale = [
        [0.0, "#d32f2f"],   # Strong counterfact (red)
        [0.25, "#ffcdd2"],  # Weak counterfact (light red)
        [0.5, "#ffffff"],   # Neutral (white)
        [0.75, "#bbdefb"],  # Weak fact (light blue)
        [1.0, "#1976d2"]    # Strong fact (blue)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=head_labels,
        y=layer_labels,
        colorscale=colorscale,
        zmid=0,  # Center colorscale at zero
        colorbar=dict(
            title="Dominance",
            tickmode="array",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Strong\nCounterfact", "Weak\nCounterfact", "Neutral", "Weak\nFact", "Strong\nFact"]
        ),
        hovertemplate="Layer: %{y}<br>Head: %{x}<br>Dominance: %{z:.3f}<br>" +
                     "<i>Blue = Fact Wins, Red = Counterfact Wins</i><extra></extra>"
    ))
    
    fig.update_layout(
        title="Panel B - Layerwise Competition Heatmap",
        xaxis_title="Attention Heads",
        yaxis_title="Transformer Layers",
        height=400,
        width=700
    )
    
    return fig


def create_counterfactual_trails_sankey(trails_data: dict, mode: str = "fact") -> go.Figure:
    """
    Panel C - Counterfactual Trails: Sankey-style flow visualization.
    
    Args:
        trails_data: Dictionary containing trail information
        mode: "fact" or "counterfact" or "both"
        
    Returns:
        Plotly figure object
    """
    if not trails_data or "trails" not in trails_data:
        return go.Figure().add_annotation(text="No trails data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    trails = trails_data["trails"]
    tokens = trails_data.get("tokens", [])
    
    # Create node and link data for Sankey diagram
    nodes = []
    links = []
    node_colors = []
    
    # Create nodes for each token at each layer
    node_map = {}
    node_id = 0
    
    num_layers = len(trails)
    for layer_idx in range(num_layers + 1):  # +1 for input layer
        for token_idx, token in enumerate(tokens):
            node_label = f"L{layer_idx}:{token}" if layer_idx > 0 else token
            nodes.append(node_label)
            node_map[(layer_idx, token_idx)] = node_id
            
            # Color nodes based on importance
            if layer_idx == 0:
                node_colors.append("rgba(100, 100, 100, 0.8)")  # Input tokens
            else:
                node_colors.append("rgba(70, 130, 180, 0.6)")   # Layer tokens
            
            node_id += 1
    
    # Create links based on attention flows
    for layer_idx, trail in enumerate(trails):
        if layer_idx >= num_layers - 1:  # Skip last layer
            continue
            
        for head in trail["heads"]:
            for flow in head["flows"]:
                if flow["strength"] > 0.1:  # Only strong flows
                    source_node = node_map.get((layer_idx, flow["from_pos"]))
                    target_node = node_map.get((layer_idx + 1, flow["to_pos"]))
                    
                    if source_node is not None and target_node is not None:
                        links.append({
                            "source": source_node,
                            "target": target_node,
                            "value": flow["strength"] * 10,  # Scale for visibility
                            "color": "rgba(70, 130, 180, 0.4)" if mode == "fact" else "rgba(220, 53, 69, 0.4)"
                        })
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links], 
            value=[link["value"] for link in links],
            color=[link["color"] for link in links]
        )
    )])
    
    color_desc = "Blue flows = Factual pathways" if mode == "fact" else "Red flows = Counterfactual pathways"
    
    fig.update_layout(
        title=f"Panel C - Information Flow Trails ({mode.title()})",
        font_size=10,
        height=500,
        annotations=[
            dict(text=color_desc, x=0.02, y=0.98, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12))
        ]
    )
    
    return fig


def create_trail_comparison_view(fact_trails: dict, counterfact_trails: dict) -> html.Div:
    """
    Panel D - Trail Comparison: Side-by-side fact vs counterfactual trails.
    
    Args:
        fact_trails: Factual trail data
        counterfact_trails: Counterfactual trail data
        
    Returns:
        HTML div component with side-by-side comparison
    """
    
    # Create side-by-side Sankey diagrams
    fact_fig = create_counterfactual_trails_sankey(fact_trails, "fact")
    counterfact_fig = create_counterfactual_trails_sankey(counterfact_trails, "counterfact")
    
    # Update titles for comparison view
    fact_fig.update_layout(title="Factual Information Trails", height=400)
    counterfact_fig.update_layout(title="Counterfactual Information Trails", height=400)
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5("Panel D - Trail Comparison", className="mb-0"),
                html.Small("Side-by-side comparison of information pathways", className="text-muted")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fact_fig, config={"displayModeBar": True})
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=counterfact_fig, config={"displayModeBar": True})
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Key Differences:", className="text-info"),
                            html.Ul([
                                html.Li("Blue flows show how factual information propagates"),
                                html.Li("Red flows show how counterfactual information propagates"),
                                html.Li("Thickness indicates attention strength"),
                                html.Li("Divergent pathways reveal where competition occurs")
                            ], className="small text-muted")
                        ])
                    ], width=12)
                ])
            ])
        ], className="mb-4")
    ])


def create_interactive_drill_down(competition_data: dict, selected_layer: int = 0) -> go.Figure:
    """
    Panel E - Interactive Drill-Down: Bar chart of head contributions.
    
    Args:
        competition_data: Dictionary containing competition matrix
        selected_layer: Layer to drill down into
        
    Returns:
        Plotly figure object
    """
    if not competition_data or "competition_matrix" not in competition_data:
        return go.Figure().add_annotation(text="No competition data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    matrix = np.array(competition_data["competition_matrix"])
    if selected_layer >= matrix.shape[0]:
        selected_layer = 0
    
    layer_data = matrix[selected_layer, :]
    head_labels = [f"Head {i+1}" for i in range(len(layer_data))]
    
    # Create bar colors based on dominance
    colors = []
    for value in layer_data:
        if value > 0:
            # Fact dominance - blue scale
            intensity = min(abs(value), 1.0)
            colors.append(f"rgba(25, 118, 210, {0.4 + intensity * 0.6})")
        else:
            # Counterfact dominance - red scale  
            intensity = min(abs(value), 1.0)
            colors.append(f"rgba(211, 47, 47, {0.4 + intensity * 0.6})")
    
    fig = go.Figure(data=[
        go.Bar(
            x=head_labels,
            y=layer_data,
            marker_color=colors,
            hovertemplate="Head: %{x}<br>Dominance: %{y:.3f}<br>" +
                         "<i>Positive = Fact Dominant, Negative = Counterfact Dominant</i><extra></extra>"
        )
    ])
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Panel E - Head Contributions for Layer {selected_layer + 1}",
        xaxis_title="Attention Heads",
        yaxis_title="Dominance Score",
        height=400,
        annotations=[
            dict(text="Positive = Fact Dominant", x=0.02, y=0.98, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="Negative = Counterfact Dominant", x=0.02, y=0.92, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red"))
        ]
    )
    
    return fig


def create_difference_map(difference_data: dict) -> go.Figure:
    """
    Panel F - Difference Map: How contributions diverge across layers.
    
    Args:
        difference_data: Dictionary containing layer-wise differences
        
    Returns:
        Plotly figure object
    """
    if not difference_data or "layer_differences" not in difference_data:
        return go.Figure().add_annotation(text="No difference data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    layer_diffs = difference_data["layer_differences"]
    
    layers = [f"Layer {d['layer'] + 1}" for d in layer_diffs]
    fact_attention = [np.mean(d["fact_attention"]) for d in layer_diffs]
    counterfact_attention = [np.mean(d["counterfact_attention"]) for d in layer_diffs]
    differences = [np.mean(d["difference"]) for d in layer_diffs]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Attention Strength Comparison", "Difference (Fact - Counterfact)"),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Top plot: Comparison of fact vs counterfact attention
    fig.add_trace(
        go.Scatter(x=layers, y=fact_attention, mode='lines+markers', 
                  name='Fact Attention', line=dict(color='blue', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=layers, y=counterfact_attention, mode='lines+markers',
                  name='Counterfact Attention', line=dict(color='red', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Bottom plot: Difference
    colors = ['blue' if d > 0 else 'red' for d in differences]
    fig.add_trace(
        go.Bar(x=layers, y=differences, name='Difference', 
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # Add horizontal line at y=0 for difference plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title="Panel F - Layer-wise Difference Map",
        height=600,
        showlegend=True,
        annotations=[
            dict(text="Positive = Fact Stronger", x=0.02, y=0.35, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="blue")),
            dict(text="Negative = Counterfact Stronger", x=0.02, y=0.30, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=10, color="red"))
        ]
    )
    
    fig.update_xaxes(title_text="Transformer Layers", row=2, col=1)
    fig.update_yaxes(title_text="Mean Attention", row=1, col=1)
    fig.update_yaxes(title_text="Difference Score", row=2, col=1)
    
    return fig


def create_knowledge_competition_dashboard(analysis_results: dict) -> html.Div:
    """
    Create a complete dashboard combining all counterfactual analysis panels.
    
    Args:
        analysis_results: Complete analysis results from knowledge competition analyzer
        
    Returns:
        HTML div component with all panels
    """
    if not analysis_results or "error" in analysis_results:
        error_msg = analysis_results.get("error", "Unknown error") if analysis_results else "No data provided"
        return html.Div([
            dbc.Alert(f"Error in knowledge competition analysis: {error_msg}", color="danger")
        ])
    
    # Extract data
    fact_text = analysis_results.get("fact_text", "")
    counterfact_text = analysis_results.get("counterfact_text", "")
    layerwise_competition = analysis_results.get("layerwise_competition", {})
    fact_trails = analysis_results.get("fact_trails", {})
    counterfact_trails = analysis_results.get("counterfact_trails", {})
    difference_maps = analysis_results.get("difference_maps", {})
    
    # Create panels
    panel_a = create_input_setup_panel(fact_text, counterfact_text)
    panel_b_fig = create_layerwise_competition_heatmap(layerwise_competition)
    panel_d = create_trail_comparison_view(fact_trails, counterfact_trails)
    panel_e_fig = create_interactive_drill_down(layerwise_competition, 0)
    panel_f_fig = create_difference_map(difference_maps)
    
    return html.Div([
        # Panel A - Input Setup
        panel_a,
        
        # Panel B - Competition Heatmap
        dbc.Card([
            dbc.CardHeader([
                html.H5("Panel B - Layerwise Competition Heatmap", className="mb-0"),
                html.Small("Bird's-eye view of fact vs counterfact competition across network", className="text-muted")
            ]),
            dbc.CardBody([
                dcc.Graph(figure=panel_b_fig, config={"displayModeBar": True})
            ])
        ], className="mb-4"),
        
        # Panel D - Trail Comparison
        panel_d,
        
        # Panel E - Interactive Drill-Down
        dbc.Card([
            dbc.CardHeader([
                html.H5("Panel E - Interactive Drill-Down", className="mb-0"),
                html.Small("Detailed analysis of head contributions", className="text-muted")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Layer for Drill-Down:", className="fw-bold"),
                        dcc.Slider(
                            id="drill-down-layer-slider",
                            min=0,
                            max=len(layerwise_competition.get("competition_matrix", [[]])) - 1,
                            value=0,
                            marks={i: f"L{i+1}" for i in range(len(layerwise_competition.get("competition_matrix", [[]])))},
                            step=1
                        )
                    ], width=6)
                ], className="mb-3"),
                dcc.Graph(id="drill-down-chart", figure=panel_e_fig, config={"displayModeBar": True})
            ])
        ], className="mb-4"),
        
        # Panel F - Difference Map  
        dbc.Card([
            dbc.CardHeader([
                html.H5("Panel F - Difference Map", className="mb-0"),
                html.Small("How contributions diverge across layers", className="text-muted")
            ]),
            dbc.CardBody([
                dcc.Graph(figure=panel_f_fig, config={"displayModeBar": True})
            ])
        ], className="mb-4"),
        
        # Summary Statistics
        dbc.Card([
            dbc.CardHeader(html.H5("Analysis Summary", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Network Architecture:", className="text-info"),
                        html.P([
                            f"Layers: {analysis_results.get('metadata', {}).get('num_layers', 'N/A')}, ",
                            f"Heads: {analysis_results.get('metadata', {}).get('num_heads', 'N/A')}"
                        ])
                    ], width=4),
                    dbc.Col([
                        html.H6("Competition Statistics:", className="text-info"),
                        html.P([
                            f"Total Divergence: {difference_maps.get('summary', {}).get('total_divergence', 0):.3f}, ",
                            f"Peak Layer: {difference_maps.get('summary', {}).get('max_divergence_layer', 'N/A') + 1 if difference_maps.get('summary', {}).get('max_divergence_layer') is not None else 'N/A'}"
                        ])
                    ], width=4),
                    dbc.Col([
                        html.H6("Pathway Analysis:", className="text-info"),
                        html.P([
                            f"Common Tokens: {len(analysis_results.get('competing_pathways', {}).get('common_tokens', []))}, ",
                            f"High Competition: {len(analysis_results.get('competing_pathways', {}).get('competition_summary', {}).get('high_competition_tokens', []))}"
                        ])
                    ], width=4)
                ])
            ])
        ], className="mb-4")
    ])

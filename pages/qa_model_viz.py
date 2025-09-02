"""
QA Model Visualization Page
Provides comprehensive model visualization with overview + detail views using Plotly and D3.
"""
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, no_update, ctx
import dash
import torch
from models.api import model_api

logger = logging.getLogger(__name__)


def create_layout(default_context="", default_question=""):
    """Create the QA model visualization layout."""
    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "Interactive Model Visualization with Overview + Detail views. Explore attention patterns, layer interactions, and model architecture.",
        ], color="info"),
        
        # Input Controls
        dbc.Card([
            dbc.CardHeader([html.I(className="fas fa-cog me-2"), "Input Configuration"]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Context", className="fw-bold mb-2"),
                        dbc.Textarea(
                            id="model-viz-context", 
                            value=default_context or "The iPhone was developed by Apple Inc. It revolutionized the smartphone industry.",
                            rows=3, 
                            className="mb-2"
                        )
                    ], width=7),
                    dbc.Col([
                        html.Label("Question", className="fw-bold mb-2"),
                        dbc.Input(
                            id="model-viz-question", 
                            value=default_question or "Who developed the iPhone?",
                            className="mb-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-play me-2"),
                            "Generate Visualization"
                        ], id="generate-model-viz", color="primary", className="w-100")
                    ], width=5)
                ])
            ])
        ], className="mb-4"),
        
        # Visualization Container
        html.Div(id="model-viz-container")
    ])

def get_qa_model_data(context, question, max_length=128):
    """Extract attention data from QA model."""
    try:
        qa_model = model_api.get_qa_model()
        model = qa_model.model
        tokenizer = qa_model.tokenizer
        
        # Ensure output attentions
        model.config.output_attentions = True
        
        # Tokenize inputs
        inputs = tokenizer.encode_plus(
            context, question,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract attentions
        attentions = outputs.attentions  # tuple of tensors
        input_ids = inputs['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Convert to numpy
        attention_data = []
        for layer_idx, layer_attn in enumerate(attentions):
            layer_attn_np = layer_attn[0].cpu().numpy()  # Remove batch dim
            attention_data.append(layer_attn_np)
        
        return {
            'attentions': attention_data,
            'tokens': tokens,
            'num_layers': len(attention_data),
            'num_heads': attention_data[0].shape[0] if attention_data else 12
        }
    except Exception as e:
        logger.error(f"Error extracting model data: {e}")
        return None

def create_3d_model_architecture(model_data):
    """Create 3D interactive model architecture showing data flow through layers."""
    if not model_data:
        return go.Figure()
    
    num_layers = model_data['num_layers']
    num_heads = model_data['num_heads']
    tokens = model_data['tokens']
    seq_len = len(tokens)
    
    fig = go.Figure()
    
    # Define 3D positions and dimensions
    layer_spacing = 3.0  # Distance between layers
    component_width = 2.0
    component_height = 1.5
    component_depth = 0.5
    
    # Colors for different components
    colors = {
        'embedding': '#FF6B6B',
        'attention': '#4ECDC4', 
        'feedforward': '#45B7D1',
        'output': '#96CEB4',
        'data_flow': '#FFD93D'
    }
    
    # Helper function to create 3D box
    def create_3d_box(x_center, y_center, z_center, width, height, depth, color, name, opacity=0.8):
        # Define the 8 vertices of a box
        x = [x_center - width/2, x_center + width/2, x_center + width/2, x_center - width/2,
             x_center - width/2, x_center + width/2, x_center + width/2, x_center - width/2]
        y = [y_center - height/2, y_center - height/2, y_center + height/2, y_center + height/2,
             y_center - height/2, y_center - height/2, y_center + height/2, y_center + height/2]
        z = [z_center - depth/2, z_center - depth/2, z_center - depth/2, z_center - depth/2,
             z_center + depth/2, z_center + depth/2, z_center + depth/2, z_center + depth/2]
        
        # Define the 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Create mesh3d for solid box
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5],
            j=[1, 3, 4, 2, 5, 3, 6, 7, 5, 6, 7, 6],
            k=[2, 4, 7, 6, 6, 6, 7, 4, 6, 7, 5, 7],
            color=color,
            opacity=opacity,
            name=name,
            hovertemplate=f"<b>{name}</b><br>Position: ({x_center:.1f}, {y_center:.1f}, {z_center:.1f})<extra></extra>"
        ))
        
        return x_center, y_center, z_center
    
    # Helper function to create data flow arrows
    def create_flow_arrow(start_pos, end_pos, color='#FFD93D'):
        x_start, y_start, z_start = start_pos
        x_end, y_end, z_end = end_pos
        
        fig.add_trace(go.Scatter3d(
            x=[x_start, x_end],
            y=[y_start, y_end],
            z=[z_start, z_end],
            mode='lines+markers',
            line=dict(color=color, width=8),
            marker=dict(size=[3, 8], color=color, symbol=['circle', 'arrow']),
            name='Data Flow',
            showlegend=False,
            hovertemplate="Data Flow<extra></extra>"
        ))
    
    # 1. Input Embeddings Layer
    embed_pos = create_3d_box(0, 0, 0, component_width, component_height, component_depth, 
                             colors['embedding'], f"Input Embeddings ({seq_len} tokens)")
    
    # Store positions for flow arrows
    layer_positions = [embed_pos]
    
    # 2. Transformer Layers
    for layer_idx in range(num_layers):
        z_pos = (layer_idx + 1) * layer_spacing
        
        # Multi-Head Attention block
        attention_pos = create_3d_box(-1.5, 0, z_pos, component_width, component_height, component_depth,
                                    colors['attention'], f"Multi-Head Attention L{layer_idx} ({num_heads} heads)")
        
        # Feed Forward Network block
        ffn_pos = create_3d_box(1.5, 0, z_pos, component_width, component_height, component_depth,
                               colors['feedforward'], f"Feed Forward Network L{layer_idx}")
        
        # Add residual connection visualization (curved path)
        residual_x = np.linspace(-1.5, 1.5, 20)
        residual_y = np.sin(np.linspace(0, np.pi, 20)) * 0.5
        residual_z = np.full(20, z_pos)
        
        fig.add_trace(go.Scatter3d(
            x=residual_x, y=residual_y, z=residual_z,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.6)', width=4, dash='dash'),
            name=f'Residual Connection L{layer_idx}',
            showlegend=False,
            hovertemplate=f"Residual Connection Layer {layer_idx}<extra></extra>"
        ))
        
        layer_positions.append(attention_pos)
        layer_positions.append(ffn_pos)
    
    # 3. Output Layer
    output_z = (num_layers + 1) * layer_spacing
    output_pos = create_3d_box(0, 0, output_z, component_width, component_height, component_depth,
                              colors['output'], "QA Output (Start/End Predictions)")
    layer_positions.append(output_pos)
    
    # 4. Create data flow arrows between major components
    # Input to first layer
    create_flow_arrow(embed_pos, (-1.5, 0, layer_spacing))
    
    # Between transformer layers
    for layer_idx in range(num_layers - 1):
        current_z = (layer_idx + 1) * layer_spacing
        next_z = (layer_idx + 2) * layer_spacing
        
        # Attention to FFN within layer
        create_flow_arrow((-1.5, 0, current_z), (1.5, 0, current_z))
        
        # FFN to next layer attention
        create_flow_arrow((1.5, 0, current_z), (-1.5, 0, next_z))
    
    # Last layer to output
    last_layer_z = num_layers * layer_spacing
    create_flow_arrow((1.5, 0, last_layer_z), output_pos)
    
    # 5. Add animated data particles flowing through the model
    # Create particle trail showing data movement
    particle_path_x = []
    particle_path_y = []
    particle_path_z = []
    
    # Path through the model
    for i in range(0, (num_layers + 2) * 10):
        t = i / 10.0
        if t <= num_layers + 1:
            particle_path_x.append(0 if t % 1 < 0.5 else (1.5 if t % 1 < 0.75 else -1.5))
            particle_path_y.append(0)
            particle_path_z.append(t * layer_spacing)
    
    # Add particle trail
    fig.add_trace(go.Scatter3d(
        x=particle_path_x[:20],  # Show only part of the trail
        y=particle_path_y[:20],
        z=particle_path_z[:20],
        mode='markers',
        marker=dict(
            size=[2 + i*0.5 for i in range(20)],
            color=colors['data_flow'],
            opacity=[0.3 + i*0.035 for i in range(20)]
        ),
        name='Data Flow Particles',
        showlegend=False,
        hovertemplate="Data flowing through model<extra></extra>"
    ))
    
    # 6. Add model information as 3D text
    fig.add_trace(go.Scatter3d(
        x=[4], y=[0], z=[output_z/2],
        mode='text',
        text=[f"""Model: DistilBERT-QA
Layers: {num_layers}
Heads: {num_heads}
Sequence: {seq_len} tokens
        
Data Flow:
Input → Embeddings
↓
{num_layers}x Transformer Layers
(Attention + FFN + Residual)
↓
QA Output Layer
↓
Start/End Predictions"""],
        textfont=dict(size=12, color='black'),
        name='Model Info',
        showlegend=False,
        hovertemplate="Model Architecture Information<extra></extra>"
    ))
    
    # Update layout for 3D visualization
    fig.update_layout(
        title="3D QA Model Architecture - Interactive Data Flow Visualization",
        scene=dict(
            xaxis=dict(title="Width", showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Depth", showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title="Layer Progression", showgrid=True, gridcolor='lightgray'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=output_z/2)
            ),
            bgcolor='rgba(240,240,240,0.1)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.8, z=2)
        ),
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        font=dict(size=12)
    )
    
    return fig

def create_3d_model_visualization(model_data):
    """Create a 3D visualization of the transformer model architecture."""
    if not model_data:
        return go.Figure()
    
    attentions = model_data['attentions']
    num_layers = model_data['num_layers']
    num_heads = model_data['num_heads']
    
    fig = go.Figure()
    
    # Model architecture components
    layer_height = 8
    component_spacing = 3
    
    for layer_idx in range(num_layers):
        layer_z = layer_idx * layer_height
        
        # Multi-Head Attention block
        attention_x = []
        attention_y = []
        attention_z = []
        attention_colors = []
        attention_sizes = []
        attention_texts = []
        
        # Arrange attention heads in a grid
        heads_per_row = int(np.sqrt(num_heads))
        for head_idx in range(num_heads):
            row = head_idx // heads_per_row
            col = head_idx % heads_per_row
            
            x = (col - heads_per_row/2) * 2
            y = (row - heads_per_row/2) * 2
            z = layer_z + 1
            
            # Get attention intensity
            if layer_idx < len(attentions):
                intensity = np.mean(attentions[layer_idx][head_idx])
            else:
                intensity = 0.1
            
            attention_x.append(x)
            attention_y.append(y)
            attention_z.append(z)
            attention_colors.append(intensity)
            attention_sizes.append(max(8, intensity * 30))
            attention_texts.append(f"Layer {layer_idx}<br>Attention Head {head_idx}<br>Intensity: {intensity:.3f}")
        
        # Add Multi-Head Attention
        fig.add_trace(go.Scatter3d(
            x=attention_x,
            y=attention_y,
            z=attention_z,
            mode='markers',
            marker=dict(
                size=attention_sizes,
                color=attention_colors,
                colorscale='Plasma',
                opacity=0.8,
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            text=attention_texts,
            hovertemplate='%{text}<extra></extra>',
            name=f'Multi-Head Attention L{layer_idx}',
            showlegend=False
        ))
        
        # Feed Forward Network (FFN)
        ffn_x = [0]
        ffn_y = [0]
        ffn_z = [layer_z + 3]
        ffn_size = [25]
        ffn_color = [0.5]
        ffn_text = [f"Layer {layer_idx}<br>Feed Forward Network<br>Hidden Size: 3072"]
        
        fig.add_trace(go.Scatter3d(
            x=ffn_x,
            y=ffn_y,
            z=ffn_z,
            mode='markers',
            marker=dict(
                size=ffn_size,
                color=ffn_color,
                colorscale='Viridis',
                opacity=0.7,
                symbol='square',
                line=dict(width=2, color='white')
            ),
            text=ffn_text,
            hovertemplate='%{text}<extra></extra>',
            name=f'FFN L{layer_idx}',
            showlegend=False
        ))
        
        # Layer Normalization (before attention)
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[layer_z + 0.5],
            mode='markers',
            marker=dict(
                size=[15],
                color=['lightblue'],
                opacity=0.6,
                symbol='diamond',
                line=dict(width=1, color='blue')
            ),
            text=[f"Layer {layer_idx}<br>Layer Normalization<br>(Pre-Attention)"],
            hovertemplate='%{text}<extra></extra>',
            name=f'LayerNorm1 L{layer_idx}',
            showlegend=False
        ))
        
        # Layer Normalization (before FFN)
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[layer_z + 2.5],
            mode='markers',
            marker=dict(
                size=[15],
                color=['lightgreen'],
                opacity=0.6,
                symbol='diamond',
                line=dict(width=1, color='green')
            ),
            text=[f"Layer {layer_idx}<br>Layer Normalization<br>(Pre-FFN)"],
            hovertemplate='%{text}<extra></extra>',
            name=f'LayerNorm2 L{layer_idx}',
            showlegend=False
        ))
        
        # Residual connections (lines)
        if layer_idx > 0:
            # Connection from previous layer
            fig.add_trace(go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[layer_z - layer_height + 3, layer_z + 0.5],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.4)', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Internal connections within layer
        # Attention to FFN
        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[layer_z + 1.5, layer_z + 2.5],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.6)', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add input embedding layer
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[-2],
        mode='markers',
        marker=dict(
            size=[30],
            color=['gold'],
            opacity=0.8,
            symbol='diamond-open',
            line=dict(width=3, color='orange')
        ),
        text=["Input Embeddings<br>+ Positional Encoding"],
        hovertemplate='%{text}<extra></extra>',
        name='Input Layer',
        showlegend=False
    ))
    
    # Add output layer
    final_z = (num_layers - 1) * layer_height + 5
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[final_z],
        mode='markers',
        marker=dict(
            size=[30],
            color=['red'],
            opacity=0.8,
            symbol='diamond-open',
            line=dict(width=3, color='darkred')
        ),
        text=["Output Layer<br>Linear + Softmax"],
        hovertemplate='%{text}<extra></extra>',
        name='Output Layer',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Transformer Model Architecture",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Depth", 
            zaxis_title="Layer Stack",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0.5)
            ),
            bgcolor='rgba(248, 249, 250, 0.1)',
            xaxis=dict(
                backgroundcolor="rgba(255, 255, 255, 0.9)",
                gridcolor="rgba(200, 200, 200, 0.3)",
                showbackground=True,
                range=[-8, 8]
            ),
            yaxis=dict(
                backgroundcolor="rgba(255, 255, 255, 0.9)",
                gridcolor="rgba(200, 200, 200, 0.3)",
                showbackground=True,
                range=[-8, 8]
            ),
            zaxis=dict(
                backgroundcolor="rgba(255, 255, 255, 0.9)",
                gridcolor="rgba(200, 200, 200, 0.3)",
                showbackground=True
            )
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Add loading component to layout first
def create_loading_component():
    return html.Div([
        html.Div([
            html.I(className="fas fa-spinner fa-spin fa-3x text-primary mb-3"),
            html.H5("Generating Model Visualization...", className="text-primary"),
            html.P("Extracting attention patterns from the QA model", className="text-muted"),
            html.Div([
                html.Div(className="progress-bar progress-bar-striped progress-bar-animated", 
                        style={"width": "100%"})
            ], className="progress mb-3", style={"height": "8px"}),
            html.Small("This may take a few seconds depending on the model size", className="text-muted")
        ], className="text-center p-4", style={
            "background": "rgba(255, 255, 255, 0.95)",
            "border-radius": "15px",
            "box-shadow": "0 4px 20px rgba(0, 0, 0, 0.1)"
        })
    ], className="d-flex justify-content-center align-items-center", 
       style={"min-height": "300px"})

# Loading callback to show immediate feedback
@callback(
    Output("model-viz-container", "children", allow_duplicate=True),
    Input("generate-model-viz", "n_clicks"),
    prevent_initial_call=True
)
def show_loading_on_click(n_clicks):
    if n_clicks:
        return create_loading_component()
    return no_update

# Main visualization callback
@callback(
    [Output("model-viz-container", "children"),
     Output("generate-model-viz", "disabled")],
    [Input("generate-model-viz", "n_clicks")],
    [State("model-viz-context", "value"),
     State("model-viz-question", "value")],
    prevent_initial_call=True
)
def generate_model_visualization(n_clicks, context, question):
    if not n_clicks:
        return html.Div(), False
    
    if not context or not context.strip():
        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide a context for visualization."
        ], color="warning")
        return error_alert, False
    
    if not question or not question.strip():
        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide a question for visualization."
        ], color="warning")
        return error_alert, False
    
    try:
        logger.info(f"Starting visualization generation for context: '{context[:50]}...' and question: '{question}'")
        
        # Get model data
        model_data = get_qa_model_data(context.strip(), question.strip())
        
        if not model_data:
            error_alert = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Failed to extract model data. Please check your inputs and ensure the QA model is loaded."
            ], color="danger")
            return error_alert, False
        
        logger.info("Model data extracted successfully, creating visualizations...")
        
        # Create visualizations
        architecture_fig = create_3d_model_architecture(model_data)
        overview_fig = create_model_overview(model_data)
        flow_fig = create_attention_flow_viz(model_data)
        layer_fig = create_layer_comparison(model_data)
        
        # Store model data for detail view
        import json
        model_data_json = {
            'attentions': [layer.tolist() for layer in model_data['attentions']],
            'tokens': model_data['tokens'],
            'num_layers': model_data['num_layers'],
            'num_heads': model_data['num_heads']
        }
        
        success_div = html.Div([
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Successfully generated visualization for {model_data['num_layers']} layers and {model_data['num_heads']} heads!"
            ], color="success", className="mb-4"),
            
            # Model Architecture - Full Width
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-sitemap me-2"),
                    "Model Architecture & Data Flow"
                ]),
                dbc.CardBody([
                    dcc.Graph(figure=architecture_fig),
                    html.P("This diagram shows how data flows through the QA model from input embeddings to final predictions.", 
                         className="text-muted text-center small")
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-th me-2"),
                            "Attention Heatmap"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(figure=overview_fig, id="overview-graph"),
                            html.P("Click on any cell to see detailed view", 
                                 className="text-muted text-center small")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Token Attention Flow"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(figure=flow_fig)
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-line me-2"),
                    "Layer-wise Analysis"
                ]),
                dbc.CardBody([
                    dcc.Graph(figure=layer_fig)
                ])
            ], className="mb-4"),
            
            # Store data for detail modal
            html.Div(id="stored-model-data", 
                    children=json.dumps(model_data_json), 
                    style={"display": "none"})
        ])
        
        return success_div, False
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        error_alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error: {str(e)}"
        ], color="danger")
        return error_alert, False

@callback(
    Output("detail-modal", "is_open"),
    Output("detail-viz-container", "children"),
    Input("overview-graph", "clickData"),
    Input("close-detail-modal", "n_clicks"),
    Input("detail-layer-slider", "value"),
    Input("detail-head-slider", "value"),
    State("detail-modal", "is_open"),
    State("stored-model-data", "children"),
    prevent_initial_call=True
)
def handle_detail_view(click_data, close_clicks, layer_val, head_val, is_open, stored_data):
    if not ctx.triggered_id:
        return is_open, no_update
    
    if ctx.triggered_id == "close-detail-modal":
        return False, no_update
    
    if ctx.triggered_id == "overview-graph" and click_data:
        # Extract clicked layer and head
        point = click_data['points'][0]
        layer_idx = point['y']
        head_idx = point['x']
        layer_num = int(layer_idx.split()[-1])
        head_num = int(head_idx.split()[-1])
        
        if stored_data:
            import json
            model_data = json.loads(stored_data)
            model_data['attentions'] = [np.array(layer) for layer in model_data['attentions']]
            
            detail_fig = create_detail_view(model_data, layer_num, head_num)
            return True, dcc.Graph(figure=detail_fig)
    
    if ctx.triggered_id in ["detail-layer-slider", "detail-head-slider"] and is_open:
        if stored_data:
            import json
            model_data = json.loads(stored_data)
            model_data['attentions'] = [np.array(layer) for layer in model_data['attentions']]
            
            detail_fig = create_detail_view(model_data, layer_val, head_val)
            return is_open, dcc.Graph(figure=detail_fig)
    
    return is_open, no_update

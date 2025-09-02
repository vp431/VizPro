"""
QA Counterfactual Data Flow Visualization
Shows how data flows through the model when processing factual vs counterfactual information.
Provides overview + detail views of layer-by-layer value changes.
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
    """Create the counterfactual data flow layout."""
    
    return html.Div([
        # Header
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "Analyze how factual vs counterfactual information flows through the model layers. ",
            "The overview shows layer-by-layer changes, click any layer to see detailed transitions."
        ], color="info"),
        
        # Input Configuration
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-balance-scale me-2"),
                "Counterfactual vs Factual Analysis Setup"
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Factual Context:", className="fw-bold mb-2"),
                        dbc.Textarea(
                            id="cf-flow-factual-context",
                            placeholder="Enter factual information...",
                            value=default_context or "The iPhone was developed by Apple Inc. It revolutionized smartphones.",
                            rows=3,
                            className="mb-2"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Counterfactual Context:", className="fw-bold mb-2"),
                        dbc.Textarea(
                            id="cf-flow-counterfactual-context",
                            placeholder="Enter counterfactual information...",
                            value="The iPhone was developed by Google Inc. It revolutionized smartphones.",
                            rows=3,
                            className="mb-2"
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Question:", className="fw-bold mb-2"),
                        dbc.Input(
                            id="cf-flow-question",
                            placeholder="What question to analyze?",
                            value=default_question or "Who developed the iPhone?",
                            className="mb-3"
                        )
                    ], width=8),
                    dbc.Col([
                        html.Label("Analysis Type:", className="fw-bold mb-2"),
                        dbc.Select(
                            id="cf-flow-analysis-type",
                            options=[
                                {"label": "🔍 Attention Flow", "value": "attention"},
                                {"label": "🧠 Hidden States", "value": "hidden"},
                                {"label": "📊 Logit Evolution", "value": "logits"}
                            ],
                            value="attention",
                            className="mb-3"
                        )
                    ], width=4)
                ]),
                html.Div([
                dbc.Button([
                    html.I(className="fas fa-play me-2"),
                    "Analyze Counterfactual Flow"
                    ], id="run-cf-flow-analysis", color="warning", size="md")
                ], className="text-center mt-3")
            ])
        ], className="mb-4"),
        
        # Results Container
        html.Div(id="cf-flow-results"),
        

    ], className="counterfactual-flow-container")

def extract_counterfactual_flow_data(factual_context, counterfactual_context, question, analysis_type="attention"):
    """Extract data flow information for factual vs counterfactual contexts."""
    try:
        qa_model = model_api.get_qa_model()
        model = qa_model.model
        tokenizer = qa_model.tokenizer
        
        # Ensure output attentions and hidden states
        model.config.output_attentions = True
        model.config.output_hidden_states = True
        
        # Process factual context
        factual_inputs = tokenizer.encode_plus(
            factual_context, question,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Process counterfactual context
        cf_inputs = tokenizer.encode_plus(
            counterfactual_context, question,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Get model outputs
        with torch.no_grad():
            factual_outputs = model(**factual_inputs)
            cf_outputs = model(**cf_inputs)
        
        # Extract relevant data based on analysis type
        if analysis_type == "attention":
            factual_data = [attn[0].cpu().numpy() for attn in factual_outputs.attentions]
            cf_data = [attn[0].cpu().numpy() for attn in cf_outputs.attentions]
        elif analysis_type == "hidden":
            factual_data = [hidden[0].cpu().numpy() for hidden in factual_outputs.hidden_states]
            cf_data = [hidden[0].cpu().numpy() for hidden in cf_outputs.hidden_states]
        else:  # logits
            # For logits, we need to get intermediate representations
            factual_data = [hidden[0].cpu().numpy() for hidden in factual_outputs.hidden_states]
            cf_data = [hidden[0].cpu().numpy() for hidden in cf_outputs.hidden_states]
        
        # Get tokens for reference
        factual_tokens = tokenizer.convert_ids_to_tokens(factual_inputs['input_ids'][0])
        cf_tokens = tokenizer.convert_ids_to_tokens(cf_inputs['input_ids'][0])
        
        return {
            'factual_data': factual_data,
            'counterfactual_data': cf_data,
            'factual_tokens': factual_tokens,
            'cf_tokens': cf_tokens,
            'num_layers': len(factual_data),
            'analysis_type': analysis_type,
            'factual_context': factual_context,
            'cf_context': counterfactual_context,
            'question': question
        }
        
    except Exception as e:
        logger.error(f"Error extracting counterfactual flow data: {e}")
        return None

def create_overview_visualization(flow_data):
    """Create overview visualization showing layer-by-layer differences."""
    if not flow_data:
        return go.Figure()
    
    factual_data = flow_data['factual_data']
    cf_data = flow_data['counterfactual_data']
    num_layers = flow_data['num_layers']
    analysis_type = flow_data['analysis_type']
    
    # Calculate layer-wise differences
    layer_differences = []
    factual_magnitudes = []
    cf_magnitudes = []
    
    for layer_idx in range(num_layers):
        if analysis_type == "attention":
            # Average attention across heads and sequence
            factual_avg = np.mean(factual_data[layer_idx])
            cf_avg = np.mean(cf_data[layer_idx])
        else:  # hidden states
            # Average hidden state magnitude
            factual_avg = np.mean(np.abs(factual_data[layer_idx]))
            cf_avg = np.mean(np.abs(cf_data[layer_idx]))
        
        factual_magnitudes.append(factual_avg)
        cf_magnitudes.append(cf_avg)
        layer_differences.append(abs(cf_avg - factual_avg))
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{analysis_type.title()} Magnitude by Layer",
            "Factual vs Counterfactual Difference"
        ),
        vertical_spacing=0.1
    )
    
    # Top plot: Factual vs Counterfactual magnitudes
    fig.add_trace(
        go.Scatter(
            x=list(range(num_layers)),
            y=factual_magnitudes,
            mode='lines+markers',
            name='Factual',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8),
            hovertemplate='Layer %{x}<br>Factual: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(num_layers)),
            y=cf_magnitudes,
            mode='lines+markers',
            name='Counterfactual',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=8),
            hovertemplate='Layer %{x}<br>Counterfactual: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Bottom plot: Differences (clickable)
    fig.add_trace(
        go.Bar(
            x=list(range(num_layers)),
            y=layer_differences,
            name='Difference',
            marker_color='#F18F01',
            hovertemplate='Layer %{x}<br>Difference: %{y:.4f}<br><b>Click for details</b><extra></extra>',
            customdata=list(range(num_layers))  # Store layer indices for click events
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Counterfactual vs Factual Data Flow Overview",
        showlegend=True,
        hovermode='closest',
        clickmode='event+select'
    )
    
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer (Click for Detail)", row=2, col=1)
    fig.update_yaxes(title_text=f"{analysis_type.title()} Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Difference", row=2, col=1)
    
    return fig

def create_enhanced_layer_detail(flow_data, layer_idx):
    """Create detailed transformer mechanics analysis for a specific layer."""
    if not flow_data or layer_idx >= flow_data['num_layers']:
        return dbc.Alert("Invalid layer selection", color="warning")
    
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    factual_data = flow_data['factual_data']
    cf_data = flow_data['counterfactual_data']
    analysis_type = flow_data['analysis_type']
    factual_tokens = flow_data['factual_tokens']
    cf_tokens = flow_data['cf_tokens']
    
    try:
        return create_transformer_mechanics_analysis(
            factual_data, cf_data, layer_idx, 
            factual_tokens, cf_tokens, analysis_type
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error creating layer detail: {str(e)}"
        ], color="danger")

def create_transformer_mechanics_analysis(factual_data, cf_data, layer_idx, factual_tokens, cf_tokens, analysis_type):
    """Detailed analysis of transformer mechanics at a specific layer."""
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Get layer data with proper shape handling
    factual_layer = factual_data[layer_idx]
    cf_layer = cf_data[layer_idx]
    
    print(f"[DEBUG] Layer {layer_idx} Analysis - Factual: {factual_layer.shape}, CF: {cf_layer.shape}")
    
    if analysis_type == "attention":
        return create_attention_mechanics_analysis(
            factual_layer, cf_layer, layer_idx, factual_tokens, cf_tokens
        )
    else:
        return create_hidden_state_mechanics_analysis(
            factual_layer, cf_layer, layer_idx, factual_tokens, cf_tokens
        )

def create_attention_mechanics_analysis(factual_attn, cf_attn, layer_idx, factual_tokens, cf_tokens):
    """Analyze attention mechanics: Q, K, V interactions and attention flow."""
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Handle different sequence lengths
    fact_heads, fact_seq, _ = factual_attn.shape
    cf_heads, cf_seq, _ = cf_attn.shape
    
    min_heads = min(fact_heads, cf_heads)
    min_seq = min(fact_seq, cf_seq)
    
    # Trim to compatible dimensions
    fact_trimmed = factual_attn[:min_heads, :min_seq, :min_seq]
    cf_trimmed = cf_attn[:min_heads, :min_seq, :min_seq]
    tokens_display = factual_tokens[:min_seq]
    
    print(f"[DEBUG] Attention analysis - Compatible shape: {fact_trimmed.shape}")
    
    # === TRANSFORMER MECHANICS ANALYSIS ===
    
    # 1. ATTENTION PATTERN ANALYSIS (What each head focuses on)
    attention_patterns = []
    for head in range(min_heads):
        fact_head = fact_trimmed[head]
        cf_head = cf_trimmed[head]
        
        # Analyze attention distribution
        fact_focus = np.max(fact_head, axis=1)  # Max attention per query position
        cf_focus = np.max(cf_head, axis=1)     # Max attention per query position
        focus_change = cf_focus - fact_focus
        
        # Attention entropy (how spread out the attention is)
        fact_entropy = -np.sum(fact_head * np.log(fact_head + 1e-12), axis=1)
        cf_entropy = -np.sum(cf_head * np.log(cf_head + 1e-12), axis=1)
        entropy_change = cf_entropy - fact_entropy
        
        # Attention shift (which tokens get more/less attention)
        attention_shift = np.sum(cf_head - fact_head, axis=0)  # Net change per key position
        
        attention_patterns.append({
            'head': head,
            'focus_change': focus_change,
            'entropy_change': entropy_change,
            'attention_shift': attention_shift,
            'fact_pattern': fact_head,
            'cf_pattern': cf_head
        })
    
    # 2. QUERY-KEY-VALUE INTERACTIONS SIMULATION
    # Simulate what happens in transformer attention mechanism
    
    # Query strength (how much each position queries for information)
    query_strength_fact = np.mean(np.sum(fact_trimmed, axis=2), axis=0)  # Average across heads
    query_strength_cf = np.mean(np.sum(cf_trimmed, axis=2), axis=0)
    query_change = query_strength_cf - query_strength_fact
    
    # Key attractiveness (how much each position attracts attention)  
    key_attractiveness_fact = np.mean(np.sum(fact_trimmed, axis=1), axis=0)  # Average across heads
    key_attractiveness_cf = np.mean(np.sum(cf_trimmed, axis=1), axis=0)
    key_change = key_attractiveness_cf - key_attractiveness_fact
    
    # Information flow intensity (overall attention flow)
    flow_intensity_fact = np.mean(fact_trimmed, axis=0)
    flow_intensity_cf = np.mean(cf_trimmed, axis=0)
    flow_change = flow_intensity_cf - flow_intensity_fact
    
    # 3. ATTENTION LOGITS ANALYSIS (Pre-softmax scores)
    # Simulate attention logits before softmax normalization
    # Higher values = stronger Query-Key similarity
    
    # Average attention weights per head (post-softmax)
    avg_attention_fact = np.mean(fact_trimmed, axis=0)
    avg_attention_cf = np.mean(cf_trimmed, axis=0)
    
    # Simulate logit differences (what would cause these attention changes)
    # This shows which Query-Key pairs became more/less compatible
    logit_changes = np.log(avg_attention_cf + 1e-12) - np.log(avg_attention_fact + 1e-12)
    
    # === CREATE TRANSFORMER MECHANICS VISUALIZATION ===
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "🔍 Query Strength Changes",
            "🔑 Key Attractiveness Changes", 
            "🌊 Attention Flow Heatmap",
            "📊 Head-wise Pattern Changes",
            "⚡ Attention Logit Changes",
            "🎯 Focus vs Entropy Analysis"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Row 1: Query and Key Analysis
    max_tokens_display = min(10, len(tokens_display))
    clean_tokens = [token.replace('Ġ', '').replace('##', '') for token in tokens_display[:max_tokens_display]]
    
    # Query Strength (how much each token seeks information)
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=query_change[:max_tokens_display],
            name='Query Change',
            marker_color=['red' if x < 0 else 'blue' for x in query_change[:max_tokens_display]],
            text=[f"{x:+.3f}" for x in query_change[:max_tokens_display]],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Key Attractiveness (how much each token attracts attention)
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=key_change[:max_tokens_display],
            name='Key Change',
            marker_color=['red' if x < 0 else 'green' for x in key_change[:max_tokens_display]],
            text=[f"{x:+.3f}" for x in key_change[:max_tokens_display]],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # Row 2: Flow Analysis and Head Patterns
    
    # Attention Flow Heatmap (limited size for readability)
    flow_display_size = min(8, max_tokens_display)
    fig.add_trace(
        go.Heatmap(
            z=flow_change[:flow_display_size, :flow_display_size],
            x=clean_tokens[:flow_display_size],
            y=clean_tokens[:flow_display_size],
            colorscale='RdBu',
            zmid=0,
            name='Flow Change',
            showscale=True
        ),
        row=2, col=1
    )
    
    # Head-wise Pattern Changes
    head_pattern_changes = [np.mean(np.abs(attention_patterns[h]['attention_shift'])) for h in range(min_heads)]
    fig.add_trace(
        go.Bar(
            x=[f"Head {i+1}" for i in range(min_heads)],
            y=head_pattern_changes,
            name='Pattern Change',
            marker_color='#9B59B6',
            text=[f"{x:.3f}" for x in head_pattern_changes],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # Row 3: Logits and Focus Analysis
    
    # Attention Logit Changes Heatmap
    fig.add_trace(
        go.Heatmap(
            z=logit_changes[:flow_display_size, :flow_display_size],
            x=clean_tokens[:flow_display_size],
            y=clean_tokens[:flow_display_size],
            colorscale='Viridis',
            name='Logit Change',
            showscale=True
        ),
        row=3, col=1
    )
    
    # Focus vs Entropy Analysis
    head_focus_changes = [np.mean(np.abs(attention_patterns[h]['focus_change'])) for h in range(min_heads)]
    head_entropy_changes = [np.mean(np.abs(attention_patterns[h]['entropy_change'])) for h in range(min_heads)]
    
    fig.add_trace(
        go.Scatter(
            x=head_focus_changes,
            y=head_entropy_changes,
            mode='markers+text',
            text=[f"H{i+1}" for i in range(min_heads)],
            textposition='middle center',
            marker=dict(size=12, color='#E74C3C'),
            name='Focus vs Entropy'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"🔧 Layer {layer_idx + 1} Transformer Mechanics Analysis",
        showlegend=True,
        title_font_size=16
    )
    
    # Add annotations explaining what each metric means
    explanations = {
        'Query': 'How much each token seeks information from others',
        'Key': 'How much each token attracts attention from others',
        'Flow': 'Attention flow changes between token pairs',
        'Heads': 'Pattern changes per attention head',
        'Logits': 'Pre-softmax compatibility changes',
        'Focus': 'Relationship between attention focus and spread'
    }
    
    return create_mechanics_analysis_card(fig, layer_idx, explanations, attention_patterns, query_change, key_change)

def create_hidden_state_mechanics_analysis(factual_hidden, cf_hidden, layer_idx, factual_tokens, cf_tokens):
    """Analyze hidden state mechanics: representations, transformations, and information encoding."""
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Handle different sequence lengths for hidden states
    print(f"[DEBUG] Hidden states shapes: factual={factual_hidden.shape}, cf={cf_hidden.shape}")
    
    # Remove batch dimension if present
    if len(factual_hidden.shape) > 2:
        factual_hidden = factual_hidden[0]
        cf_hidden = cf_hidden[0]
    
    fact_seq, fact_hidden_dim = factual_hidden.shape
    cf_seq, cf_hidden_dim = cf_hidden.shape
    
    min_seq = min(fact_seq, cf_seq)
    min_hidden_dim = min(fact_hidden_dim, cf_hidden_dim)
    
    # Trim to compatible dimensions
    fact_trimmed = factual_hidden[:min_seq, :min_hidden_dim]
    cf_trimmed = cf_hidden[:min_seq, :min_hidden_dim]
    tokens_display = factual_tokens[:min_seq]
    
    print(f"[DEBUG] Hidden states trimmed: {fact_trimmed.shape}")
    
    # === HIDDEN STATE MECHANICS ANALYSIS ===
    
    # 1. REPRESENTATION MAGNITUDE CHANGES
    fact_magnitudes = np.linalg.norm(fact_trimmed, axis=1)  # L2 norm per token
    cf_magnitudes = np.linalg.norm(cf_trimmed, axis=1)
    magnitude_changes = cf_magnitudes - fact_magnitudes
    
    # 2. DIMENSIONAL ACTIVATION PATTERNS
    fact_activations = np.mean(np.abs(fact_trimmed), axis=0)  # Average activation per dimension
    cf_activations = np.mean(np.abs(cf_trimmed), axis=0)
    activation_changes = cf_activations - fact_activations
    
    # 3. TOKEN SIMILARITY CHANGES
    # How similar tokens are to each other (cosine similarity)
    def cosine_similarity_matrix(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / (norms + 1e-12)
        return np.dot(normalized, normalized.T)
    
    fact_similarity = cosine_similarity_matrix(fact_trimmed)
    cf_similarity = cosine_similarity_matrix(cf_trimmed)
    similarity_changes = cf_similarity - fact_similarity
    
    # 4. INFORMATION CONTENT ANALYSIS
    # Estimate information content using activation variance
    fact_info_content = np.var(fact_trimmed, axis=1)  # Variance per token
    cf_info_content = np.var(cf_trimmed, axis=1)
    info_changes = cf_info_content - fact_info_content
    
    # === CREATE HIDDEN STATE VISUALIZATION ===
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "🎯 Representation Magnitude Changes",
            "⚡ Dimensional Activation Changes",
            "🔗 Token Similarity Matrix",
            "📊 Information Content Changes"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    max_tokens_display = min(12, len(tokens_display))
    clean_tokens = [token.replace('Ġ', '').replace('##', '') for token in tokens_display[:max_tokens_display]]
    
    # Magnitude Changes
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=magnitude_changes[:max_tokens_display],
            name='Magnitude Change',
            marker_color=['red' if x < 0 else 'blue' for x in magnitude_changes[:max_tokens_display]],
            text=[f"{x:+.3f}" for x in magnitude_changes[:max_tokens_display]],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Top Dimensional Changes
    top_dims = np.argsort(np.abs(activation_changes))[-20:][::-1]
    fig.add_trace(
        go.Bar(
            x=[f"Dim {d}" for d in top_dims],
            y=[activation_changes[d] for d in top_dims],
            name='Activation Change',
            marker_color=['red' if activation_changes[d] < 0 else 'green' for d in top_dims]
        ),
        row=1, col=2
    )
    
    # Similarity Matrix
    similarity_display_size = min(8, max_tokens_display)
    fig.add_trace(
        go.Heatmap(
            z=similarity_changes[:similarity_display_size, :similarity_display_size],
            x=clean_tokens[:similarity_display_size],
            y=clean_tokens[:similarity_display_size],
            colorscale='RdBu',
            zmid=0,
            name='Similarity Change'
        ),
        row=2, col=1
    )
    
    # Information Content Changes
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=info_changes[:max_tokens_display],
            name='Info Change',
            marker_color=['red' if x < 0 else 'purple' for x in info_changes[:max_tokens_display]],
            text=[f"{x:+.3f}" for x in info_changes[:max_tokens_display]],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text=f"🧠 Layer {layer_idx + 1} Hidden State Mechanics Analysis",
        showlegend=True,
        title_font_size=16
    )
    
    explanations = {
        'Magnitude': 'Overall strength of token representations',
        'Dimensions': 'Which hidden dimensions changed most',
        'Similarity': 'How token relationships changed',
        'Information': 'Information content per token representation'
    }
    
    return create_mechanics_analysis_card(fig, layer_idx, explanations, None, magnitude_changes, activation_changes)

def create_mechanics_analysis_card(fig, layer_idx, explanations, attention_patterns, primary_changes, secondary_changes):
    """Create the final card with mechanics analysis and insights."""
    
    # Calculate key statistics
    total_change = np.sum(np.abs(primary_changes)) if primary_changes is not None else 0
    max_change_idx = np.argmax(np.abs(primary_changes)) if primary_changes is not None and len(primary_changes) > 0 else 0
    avg_change = np.mean(np.abs(secondary_changes)) if secondary_changes is not None else 0
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-cogs me-2"),
            f"Layer {layer_idx + 1} Transformer Mechanics"
        ]),
        dbc.CardBody([
            # Main visualization
            dcc.Graph(figure=fig, style={'height': '700px'}),
            
            # Mechanics Insights
            html.Hr(),
            html.Div([
                html.H6([
                    html.I(className="fas fa-gear me-2"),
                    "Transformer Mechanics Insights"
                ], className="text-info mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Change", className="card-title text-primary"),
                                html.H4(f"{total_change:.4f}", className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-primary mb-2")
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Max Change Position", className="card-title text-warning"),
                                html.H6(f"Position {max_change_idx + 1}", className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-warning mb-2")
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Average Change", className="card-title text-success"),
                                html.H6(f"{avg_change:.4f}", className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-success mb-2")
                    ], width=4)
                ])
            ], className="mechanics-insights")
        ])
    ], className="layer-detail-card")
            fact_trimmed = factual_layer_raw[:min_heads, :min_seq, :min_seq]
            cf_trimmed = cf_layer_raw[:min_heads, :min_seq, :min_seq]
            
            print(f"[DEBUG] Trimmed shapes: fact={fact_trimmed.shape}, cf={cf_trimmed.shape}")
            
            # === ANALYSIS 1: HEAD-WISE COMPARISON ===
            head_analysis = []
            
            for head_idx in range(min_heads):
                fact_head = fact_trimmed[head_idx]  # [seq, seq]
                cf_head = cf_trimmed[head_idx]      # [seq, seq]
                
                # Calculate metrics for this head
                pattern_diff = np.mean(np.abs(cf_head - fact_head))
                
                # Attention entropy (how focused vs diffuse)
                fact_entropy = -np.sum(fact_head * np.log(fact_head + 1e-12))
                cf_entropy = -np.sum(cf_head * np.log(cf_head + 1e-12))
                entropy_change = cf_entropy - fact_entropy
                
                # Max attention (peak focus)
                fact_max = np.max(fact_head)
                cf_max = np.max(cf_head)
                focus_change = cf_max - fact_max
                
                head_analysis.append({
                    'head': head_idx,
                    'pattern_diff': pattern_diff,
                    'entropy_change': entropy_change,
                    'focus_change': focus_change,
                    'fact_entropy': fact_entropy,
                    'cf_entropy': cf_entropy
                })
            
            # === ANALYSIS 2: TOKEN ATTENTION PATTERNS ===
            # How much attention each token receives (averaged across heads and from-positions)
            fact_token_received = np.mean(fact_trimmed, axis=(0, 1))  # [seq] - attention TO each token
            cf_token_received = np.mean(cf_trimmed, axis=(0, 1))      # [seq] - attention TO each token
            token_received_diff = cf_token_received - fact_token_received
            
            # How much attention each token gives out (averaged across heads and to-positions)
            fact_token_given = np.mean(fact_trimmed, axis=(0, 2))     # [seq] - attention FROM each token  
            cf_token_given = np.mean(cf_trimmed, axis=(0, 2))         # [seq] - attention FROM each token
            token_given_diff = cf_token_given - fact_token_given
            
            # === ANALYSIS 3: ATTENTION FLOW CHANGES ===
            # Overall flow intensity per position
            fact_flow_intensity = np.sum(fact_trimmed, axis=0)  # [seq, seq] - total flow between positions
            cf_flow_intensity = np.sum(cf_trimmed, axis=0)      # [seq, seq] - total flow between positions
            flow_diff_matrix = cf_flow_intensity - fact_flow_intensity
            
            # Create multi-panel visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Head-wise Attention Differences",
                    "Attention Entropy Comparison", 
                    "Token Attention Flow",
                    "Most Affected Tokens"
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Top Left: Head pattern differences
            head_indices = [f"H{h['head']+1}" for h in head_analysis]
            pattern_diffs = [h['pattern_diff'] for h in head_analysis]
            
            fig.add_trace(
                go.Bar(x=head_indices, 
                       y=pattern_diffs, 
                       marker_color='#FF6B6B',
                       name='Pattern Diff',
                       showlegend=False),
                row=1, col=1
            )
            
            # Top Right: Entropy comparison
            fact_entropies = [h['fact_entropy'] for h in head_analysis]
            cf_entropies = [h['cf_entropy'] for h in head_analysis]
            
            fig.add_trace(
                go.Scatter(x=head_indices,
                          y=fact_entropies,
                          mode='markers+lines',
                          name='Factual',
                          marker_color='#4ECDC4',
                          line=dict(width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=head_indices,
                          y=cf_entropies,
                          mode='markers+lines', 
                          name='Counterfactual',
                          marker_color='#FFE66D',
                          line=dict(width=2)),
                row=1, col=2
            )
            
            # Bottom Left: Token attention received (how much attention each token gets)
            trimmed_tokens = factual_tokens[:min_seq]
            max_tokens = min(12, len(trimmed_tokens))  # Limit for readability
            display_tokens = [token.replace('Ġ', '').replace('##', '') for token in trimmed_tokens[:max_tokens]]
            
            fig.add_trace(
                go.Bar(x=display_tokens,
                       y=fact_token_received[:max_tokens],
                       name='Factual',
                       marker_color='#4ECDC4',
                       opacity=0.8),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=display_tokens,
                       y=cf_token_received[:max_tokens],
                       name='Counterfactual',
                       marker_color='#FFE66D',
                       opacity=0.7),
                row=2, col=1
            )
            
            # Bottom Right: Most affected tokens (biggest changes in received attention)
            if len(token_received_diff) > 0:
                # Get absolute differences for ranking
                abs_token_diffs = np.abs(token_received_diff)
                top_affected_indices = np.argsort(abs_token_diffs)[-8:][::-1]  # Top 8 most affected
                top_tokens = [trimmed_tokens[i].replace('Ġ', '').replace('##', '') for i in top_affected_indices if i < len(trimmed_tokens)]
                top_diffs = [token_received_diff[i] for i in top_affected_indices if i < len(token_received_diff)]
            else:
                top_tokens = []
                top_diffs = []
            
            fig.add_trace(
                go.Bar(x=top_tokens,
                       y=top_diffs,
                       marker_color='#FF8A5B',
                       name='Token Impact',
                       showlegend=False),
                row=2, col=2
            )
            
            # Calculate statistics for attention
            total_change = np.sum(pattern_diffs) if pattern_diffs else 0
            max_head = np.argmax(pattern_diffs) if pattern_diffs else 0
            avg_entropy_diff = np.mean([abs(h['entropy_change']) for h in head_analysis]) if head_analysis else 0
            
        else:  # hidden states analysis
            factual_layer = factual_data[layer_idx]
            cf_layer = cf_data[layer_idx]
            
            print(f"[DEBUG] Hidden states - original shapes: factual={factual_layer.shape}, cf={cf_layer.shape}")
            
            # Flatten to work with hidden states
            if len(factual_layer.shape) > 2:
                factual_layer = factual_layer[0]  # Remove batch dimension
                cf_layer = cf_layer[0]
            
            print(f"[DEBUG] Hidden states - after flattening: factual={factual_layer.shape}, cf={cf_layer.shape}")
            
            # Handle different sequence lengths for hidden states
            min_seq_len = min(factual_layer.shape[0], cf_layer.shape[0])
            min_hidden_dim = min(factual_layer.shape[1], cf_layer.shape[1])
            
            factual_layer_trimmed = factual_layer[:min_seq_len, :min_hidden_dim]
            cf_layer_trimmed = cf_layer[:min_seq_len, :min_hidden_dim]
            
            print(f"[DEBUG] Hidden states - after trimming: factual={factual_layer_trimmed.shape}, cf={cf_layer_trimmed.shape}")
            
            # Ensure shapes match exactly
            assert factual_layer_trimmed.shape == cf_layer_trimmed.shape, f"Shape mismatch: {factual_layer_trimmed.shape} vs {cf_layer_trimmed.shape}"
            
            # Token-wise analysis
            token_diffs = np.mean(np.abs(cf_layer_trimmed - factual_layer_trimmed), axis=1)
            
            # Dimension-wise analysis  
            dim_diffs = np.mean(np.abs(cf_layer_trimmed - factual_layer_trimmed), axis=0)
            
            # Create visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Token-wise Hidden State Changes", "Dimension-wise Changes"),
                horizontal_spacing=0.1
            )
            
            max_tokens = min(20, len(token_diffs))
            trimmed_tokens_hidden = factual_tokens[:min_seq_len]
            display_tokens = [token.replace('Ġ', '').replace('##', '') for token in trimmed_tokens_hidden[:max_tokens]]
            
            fig.add_trace(
                go.Bar(x=display_tokens,
                       y=token_diffs[:max_tokens],
                       marker_color='#A8E6CF',
                       name='Token Changes'),
                row=1, col=1
            )
            
            # Show top changing dimensions
            top_dims = np.argsort(dim_diffs)[-20:][::-1]  # Top 20 dimensions
            fig.add_trace(
                go.Bar(x=[f"Dim {d}" for d in top_dims],
                       y=[dim_diffs[d] for d in top_dims],
                       marker_color='#FFB6C1',
                       name='Dimension Changes'),
                row=1, col=2
            )
            
            # Calculate statistics for hidden states
            total_change = np.sum(token_diffs)
            max_token_idx = np.argmax(token_diffs) if len(token_diffs) > 0 else 0
            avg_entropy_diff = 0  # Not applicable for hidden states
        
        # Update layout
        fig.update_layout(
            height=500,
            title_text=f"Layer {layer_idx + 1} Detailed Analysis - {analysis_type.title()}",
            showlegend=True,
            title_font_size=16,
            font_size=12,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Create comprehensive detail card
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-microscope me-2"),
                f"Layer {layer_idx + 1} Deep Dive Analysis"
            ]),
            dbc.CardBody([
                # Statistics Row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Change", className="card-title text-primary"),
                                html.H4(f"{total_change:.4f}", className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-primary")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Most Affected", className="card-title text-warning"),
                                html.H6(f"Head {max_head + 1}" if analysis_type == "attention" else f"Token {max_token_idx + 1}", 
                                        className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-warning")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Analysis Type", className="card-title text-info"),
                                html.H6(analysis_type.title(), className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-info")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Layer Position", className="card-title text-success"),
                                html.H6(f"{layer_idx + 1} / {flow_data['num_layers']}", className="text-dark mb-0")
                            ], className="text-center")
                        ], className="border-success")
                    ], width=3)
                ], className="mb-4"),
                
                # Main visualization
                dcc.Graph(figure=fig, config={"displayModeBar": True}),
                
                # Insights section
                html.Hr(),
                html.Div([
                    html.H6([
                        html.I(className="fas fa-lightbulb me-2"),
                        "Key Insights"
                    ], className="text-info mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Strong("🎯 Competition Hotspot: "),
                                f"Head {max_head + 1} shows the strongest difference " if analysis_type == "attention" 
                                else f"Token '{factual_tokens[max_token_idx].replace('Ġ', '')}' most affected" if 'max_token_idx' in locals() else "Token analysis completed",
                                f" with {pattern_diffs[max_head]:.4f} change magnitude." if analysis_type == "attention" and pattern_diffs
                                else f" with {token_diffs[max_token_idx]:.4f} change magnitude." if 'token_diffs' in locals() and 'max_token_idx' in locals()
                                else ""
                            ], className="mb-2")
                        ], width=12),
                        dbc.Col([
                            html.Div([
                                html.Strong("📊 Layer Impact: "),
                                f"This layer contributes {(total_change/np.sum([np.sum(np.abs(cf_data[i] - factual_data[i])) for i in range(len(factual_data))])*100):.1f}% ",
                                "of total model difference between factual and counterfactual processing."
                            ], className="mb-2")
                        ], width=12),
                        dbc.Col([
                            html.Div([
                                html.Strong("🧠 Processing Style: "),
                                f"Average entropy difference of {avg_entropy_diff:.4f} indicates " if analysis_type == "attention" else "Hidden state analysis shows ",
                                "focused vs diffuse attention patterns." if analysis_type == "attention" and avg_entropy_diff > 0.1 
                                else "distributed processing changes." if analysis_type == "attention"
                                else "significant representational shifts in the hidden space."
                            ])
                        ], width=12)
                    ])
                ], className="insights-section")
            ])
        ], className="layer-detail-card")
        
    except Exception as e:
        logger.error(f"Error creating enhanced layer detail: {e}")
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error creating layer detail: {str(e)}"
        ], color="danger")

# Callbacks
@callback(
    Output("cf-flow-results", "children"),
    Input("run-cf-flow-analysis", "n_clicks"),
    [State("cf-flow-factual-context", "value"),
     State("cf-flow-counterfactual-context", "value"),
     State("cf-flow-question", "value"),
     State("cf-flow-analysis-type", "value")],
    prevent_initial_call=True
)
def run_counterfactual_flow_analysis(n_clicks, factual_context, cf_context, question, analysis_type):
    """Run the counterfactual flow analysis."""
    if not n_clicks:
        return html.Div()
    
    # Validate inputs
    if not all([factual_context, cf_context, question]):
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please provide factual context, counterfactual context, and question."
        ], color="warning")
    
    try:
        # Extract flow data
        flow_data = extract_counterfactual_flow_data(
            factual_context.strip(),
            cf_context.strip(), 
            question.strip(),
            analysis_type
        )
        
        if not flow_data:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Failed to extract flow data. Please check your inputs."
            ], color="danger")
        
        # Create overview visualization
        overview_fig = create_overview_visualization(flow_data)
        
        # Store flow data for detail modal
        import json
        flow_data_json = {
            'factual_data': [layer.tolist() for layer in flow_data['factual_data']],
            'counterfactual_data': [layer.tolist() for layer in flow_data['counterfactual_data']],
            'factual_tokens': flow_data['factual_tokens'],
            'cf_tokens': flow_data['cf_tokens'],
            'num_layers': flow_data['num_layers'],
            'analysis_type': flow_data['analysis_type'],
            'factual_context': flow_data['factual_context'],
            'cf_context': flow_data['cf_context'],
            'question': flow_data['question']
        }
        
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Successfully analyzed {flow_data['num_layers']} layers using {analysis_type} analysis!"
            ], color="success", className="mb-4"),
            
            # Main Overview Card
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-line me-2"),
                    "Data Flow Overview"
                ]),
                dbc.CardBody([
                    dcc.Graph(
                        figure=overview_fig,
                        id="cf-flow-overview-graph",
                        style={"height": "500px"},
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                        }
                    ),
                    html.Hr(className="my-4"),
                    html.Div([
                        html.H6([
                            html.I(className="fas fa-info-circle me-2"),
                            "Click any layer in the chart above to see detailed analysis below"
                        ], className="text-muted text-center mb-3"),
                        html.Div([
                            dbc.Button("Test Layer 0", id="test-layer-btn", color="outline-secondary", size="sm"),
                            html.Small(" (Debug: Click to test layer detail)", className="text-muted ms-2")
                        ], className="text-center")
                    ])
                ])
            ], className="mb-4"),
            
            # Integrated Layer Detail View (replaces modal)
            html.Div(
                id="cf-flow-layer-detail-section",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-mouse-pointer fa-2x text-muted mb-3"),
                                html.H5("Click any layer above to see detailed analysis", className="text-muted"),
                                html.P("Interactive layer-by-layer breakdown will appear here", className="text-muted")
                            ], className="text-center py-4")
                        ])
                    ], className="border-dashed", style={"border-style": "dashed", "border-color": "#dee2e6"})
                ]
            ),
            
            # Store data for callbacks
            html.Div(
                id="stored-cf-flow-data",
                children=json.dumps(flow_data_json),
                style={"display": "none"}
            )
        ])
        
    except Exception as e:
        logger.error(f"Error in counterfactual flow analysis: {e}")
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error: {str(e)}"
        ], color="danger")

@callback(
    Output("cf-flow-layer-detail-section", "children"),
    [Input("cf-flow-overview-graph", "clickData"),
     Input("test-layer-btn", "n_clicks")],
    State("stored-cf-flow-data", "children"),
    prevent_initial_call=False
)
def update_layer_detail_section(click_data, test_btn_clicks, stored_data):
    """Update the integrated layer detail section when a layer is clicked."""
    
    # Check if test button was clicked
    if test_btn_clicks and stored_data:
        try:
            import json
            flow_data = json.loads(stored_data)
            flow_data['factual_data'] = [np.array(layer) for layer in flow_data['factual_data']]
            flow_data['counterfactual_data'] = [np.array(layer) for layer in flow_data['counterfactual_data']]
            
            # Test with layer 0
            detail_content = create_enhanced_layer_detail(flow_data, 0)
            return detail_content
        except Exception as e:
            return dbc.Alert(f"Test error: {str(e)}", color="danger")
    
    # Handle graph clicks
    if not click_data:
        return html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-mouse-pointer fa-2x text-muted mb-3"),
                        html.H5("Click any layer above to see detailed analysis", className="text-muted"),
                        html.P("Interactive layer-by-layer breakdown will appear here", className="text-muted")
                    ], className="text-center py-4")
                ])
            ], className="border-dashed", style={"border-style": "dashed", "border-color": "#dee2e6"})
        ])
    
    # Debug: Show that click was detected
    print(f"[DEBUG] Click detected! Click data: {click_data}")
    
    if not stored_data:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "No analysis data available. Please run the analysis first."
        ], color="warning")
    
    try:
            import json
            flow_data = json.loads(stored_data)
        
        # Debug logging
        print(f"[DEBUG] Flow data keys: {list(flow_data.keys())}")
        
            # Convert back to numpy arrays
            flow_data['factual_data'] = [np.array(layer) for layer in flow_data['factual_data']]
            flow_data['counterfactual_data'] = [np.array(layer) for layer in flow_data['counterfactual_data']]
            
        # Get clicked layer - handle both trace clicks
            point = click_data['points'][0]
        layer_idx = int(point['x'])
        
        print(f"[DEBUG] Processing layer {layer_idx} detail")
        
        # Call the enhanced layer detail analysis
        return create_enhanced_layer_detail(stored_data, layer_idx)
        
    except Exception as e:
        print(f"[DEBUG] Error in callback: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error loading layer detail: {str(e)}"
        ], color="danger")
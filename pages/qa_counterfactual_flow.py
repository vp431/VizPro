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
    
    # 3. VALUE INFORMATION FLOW ANALYSIS
    # Simulate value information flow (what information is being passed)
    value_flow_fact = np.sum(fact_trimmed, axis=1)  # Sum across to-positions (how much info each position sends)
    value_flow_cf = np.sum(cf_trimmed, axis=1)
    value_flow_change = np.mean(value_flow_cf - value_flow_fact, axis=0)  # Average across heads
    
    # 4. ATTENTION LOGITS ANALYSIS (Pre-softmax scores)
    # Simulate attention logits before softmax normalization
    avg_attention_fact = np.mean(fact_trimmed, axis=0)
    avg_attention_cf = np.mean(cf_trimmed, axis=0)
    logit_changes = np.log(avg_attention_cf + 1e-12) - np.log(avg_attention_fact + 1e-12)
    
    # 5. ATTENTION ENTROPY PER HEAD
    head_entropies_fact = []
    head_entropies_cf = []
    for h in range(min_heads):
        ent_f = -np.sum(fact_trimmed[h] * np.log(fact_trimmed[h] + 1e-12), axis=1)
        ent_cf = -np.sum(cf_trimmed[h] * np.log(cf_trimmed[h] + 1e-12), axis=1)
        head_entropies_fact.append(np.mean(ent_f))
        head_entropies_cf.append(np.mean(ent_cf))
    
    # 6. FLOW DIRECTIONALITY (incoming vs outgoing attention)
    incoming_attention_fact = np.mean(np.sum(fact_trimmed, axis=1), axis=0)  # How much attention each token receives
    outgoing_attention_fact = np.mean(np.sum(fact_trimmed, axis=2), axis=0)  # How much attention each token gives
    incoming_attention_cf = np.mean(np.sum(cf_trimmed, axis=1), axis=0)
    outgoing_attention_cf = np.mean(np.sum(cf_trimmed, axis=2), axis=0)
    
    attention_balance_change = (incoming_attention_cf - outgoing_attention_cf) - (incoming_attention_fact - outgoing_attention_fact)
    
    # 7. QUERY-KEY COMPATIBILITY MATRIX
    qk_compatibility_fact = np.corrcoef(avg_attention_fact)
    qk_compatibility_cf = np.corrcoef(avg_attention_cf)
    qk_compatibility_change = qk_compatibility_cf - qk_compatibility_fact
    
    # 8. HEAD SPECIALIZATION (how different each head is from others)
    head_specialization_fact = []
    head_specialization_cf = []
    for h in range(min_heads):
        # Calculate how different this head is from the average
        avg_pattern_fact = np.mean(fact_trimmed, axis=0)
        avg_pattern_cf = np.mean(cf_trimmed, axis=0)
        spec_f = np.mean(np.abs(fact_trimmed[h] - avg_pattern_fact))
        spec_cf = np.mean(np.abs(cf_trimmed[h] - avg_pattern_cf))
        head_specialization_fact.append(spec_f)
        head_specialization_cf.append(spec_cf)
    
    # 9. COMPETITION DYNAMICS (which tokens compete for attention)
    competition_matrix_fact = np.std(fact_trimmed, axis=0)  # Variance in attention allocation
    competition_matrix_cf = np.std(cf_trimmed, axis=0)
    competition_change = competition_matrix_cf - competition_matrix_fact
    
    # === CREATE COMPREHENSIVE TRANSFORMER MECHANICS VISUALIZATION ===
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=(
            "🔍 Query Strength Changes",
            "🔑 Key Attractiveness Changes", 
            "💎 Value Information Flow",
            "⚡ Pre-Softmax Logits",
            "🌊 Post-Softmax Attention",
            "📊 Head-wise Patterns",
            "🎯 Attention Entropy",
            "🔄 Flow Directionality",
            "📈 Query-Key Compatibility",
            "🧠 Head Specialization",
            "⚖️ Attention Balance",
            "🎪 Competition Dynamics"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )
    
    # Prepare display data
    max_tokens_display = min(8, len(tokens_display))
    clean_tokens = [token.replace('Ġ', '').replace('##', '') for token in tokens_display[:max_tokens_display]]
    
    # === ROW 1: QUERY, KEY, VALUE, LOGITS ===
    
    # 1. Query Strength Changes
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=query_change[:max_tokens_display],
            name='Query Change',
            marker_color=['red' if x < 0 else 'blue' for x in query_change[:max_tokens_display]],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Key Attractiveness Changes
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=key_change[:max_tokens_display],
            name='Key Change',
            marker_color=['red' if x < 0 else 'green' for x in key_change[:max_tokens_display]],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Value Information Flow
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=value_flow_change[:max_tokens_display],
            name='Value Flow',
            marker_color=['red' if x < 0 else 'purple' for x in value_flow_change[:max_tokens_display]],
            showlegend=False
        ),
        row=1, col=3
    )
    
    # 4. Pre-Softmax Logits Heatmap
    logit_size = min(6, max_tokens_display)
    fig.add_trace(
        go.Heatmap(
            z=logit_changes[:logit_size, :logit_size],
            x=clean_tokens[:logit_size],
            y=clean_tokens[:logit_size],
            colorscale='RdBu',
            zmid=0,
            showscale=False
        ),
        row=1, col=4
    )
    
    # === ROW 2: ATTENTION FLOW, HEADS, ENTROPY, DIRECTIONALITY ===
    
    # 5. Post-Softmax Attention Flow Heatmap
    flow_size = min(6, max_tokens_display)
    fig.add_trace(
        go.Heatmap(
            z=flow_change[:flow_size, :flow_size],
            x=clean_tokens[:flow_size],
            y=clean_tokens[:flow_size],
            colorscale='RdBu',
            zmid=0,
            showscale=False
        ),
        row=2, col=1
    )
    
    # 6. Head-wise Pattern Changes
    head_pattern_changes = [np.mean(np.abs(attention_patterns[h]['attention_shift'])) for h in range(min_heads)]
    fig.add_trace(
        go.Bar(
            x=[f"H{i+1}" for i in range(min_heads)],
            y=head_pattern_changes,
            marker_color='#9B59B6',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 7. Attention Entropy per Head
    fig.add_trace(
        go.Scatter(
            x=[f"H{i+1}" for i in range(min_heads)],
            y=head_entropies_fact,
            mode='markers+lines',
            name='Factual',
            marker_color='#4ECDC4',
            line=dict(width=2)
        ),
        row=2, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=[f"H{i+1}" for i in range(min_heads)],
            y=head_entropies_cf,
            mode='markers+lines',
            name='CF',
            marker_color='#FFE66D',
            line=dict(width=2)
        ),
        row=2, col=3
    )
    
    # 8. Flow Directionality (Attention Balance)
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=attention_balance_change[:max_tokens_display],
            marker_color=['red' if x < 0 else 'orange' for x in attention_balance_change[:max_tokens_display]],
            showlegend=False
        ),
        row=2, col=4
    )
    
    # === ROW 3: COMPATIBILITY, SPECIALIZATION, BALANCE, COMPETITION ===
    
    # 9. Query-Key Compatibility Changes
    comp_size = min(6, max_tokens_display)
    fig.add_trace(
        go.Heatmap(
            z=qk_compatibility_change[:comp_size, :comp_size],
            x=clean_tokens[:comp_size],
            y=clean_tokens[:comp_size],
            colorscale='Viridis',
            showscale=False
        ),
        row=3, col=1
    )
    
    # 10. Head Specialization Changes
    head_spec_changes = [head_specialization_cf[i] - head_specialization_fact[i] for i in range(min_heads)]
    fig.add_trace(
        go.Scatter(
            x=head_specialization_fact,
            y=head_specialization_cf,
            mode='markers+text',
            text=[f"H{i+1}" for i in range(min_heads)],
            textposition='middle center',
            marker=dict(size=10, color='#E74C3C'),
            showlegend=False
        ),
        row=3, col=2
    )
    
    # 11. Attention Balance Changes
    fig.add_trace(
        go.Bar(
            x=clean_tokens,
            y=attention_balance_change[:max_tokens_display],
            marker_color=['red' if x < 0 else 'cyan' for x in attention_balance_change[:max_tokens_display]],
            showlegend=False
        ),
        row=3, col=3
    )
    
    # 12. Competition Dynamics
    fig.add_trace(
        go.Heatmap(
            z=competition_change[:comp_size, :comp_size],
            x=clean_tokens[:comp_size],
            y=clean_tokens[:comp_size],
            colorscale='Plasma',
            showscale=False
        ),
        row=3, col=4
    )
    
    # Update layout for comprehensive analysis
    fig.update_layout(
        height=1200,
        title_text=f"🔧 Layer {layer_idx + 1} Comprehensive Transformer Mechanics",
        showlegend=True,
        title_font_size=18,
        margin=dict(t=120, b=60, l=60, r=60),
        font=dict(size=10)
    )
    
    # Make all subplot areas more square-like
    fig.update_xaxes(tickangle=45, title_font_size=10)
    fig.update_yaxes(title_font_size=10)
    
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
        vertical_spacing=0.18,
        horizontal_spacing=0.12
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
        height=800,
        title_text=f"🧠 Layer {layer_idx + 1} Hidden State Mechanics Analysis",
        showlegend=True,
        title_font_size=16,
        margin=dict(t=100, b=50, l=50, r=50)
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
            # Main visualization using full width
            dcc.Graph(
                figure=fig, 
                style={
                    'height': '1200px', 
                    'width': '100%'
                },
                config={'displayModeBar': True, 'responsive': True}
            )
        ], style={'width': '100%', 'padding': '10px'})
    ], className="layer-detail-card", style={'width': '100%'})

# Register callbacks for this page
@callback(
    Output("cf-flow-results", "children"),
    [Input("cf-flow-analyze-btn", "n_clicks")],
    [State("cf-flow-factual-context", "value"),
     State("cf-flow-counterfactual-context", "value"),
     State("cf-flow-question", "value")]
)
def run_counterfactual_flow_analysis(n_clicks, factual_context, cf_context, question):
    """Run the counterfactual flow analysis when button is clicked."""
    if not n_clicks:
        return html.Div()
    
    if not factual_context or not cf_context or not question:
        return dbc.Alert("Please provide factual context, counterfactual context, and a question.", color="warning")
    
    try:
        from models.api import model_api
        
        # Ensure QA model is loaded
        if not model_api.is_model_loaded("qa"):
            model_api.load_model("qa")
        
        # Run QA for both contexts
        factual_result = model_api.answer_question(factual_context, question)
        cf_result = model_api.answer_question(cf_context, question)
        
        # Get attention weights for both
        factual_attention_data = model_api.get_attention_weights(factual_context + " " + question, "qa")
        cf_attention_data = model_api.get_attention_weights(cf_context + " " + question, "qa")
        
        if not factual_attention_data or not cf_attention_data:
            return dbc.Alert("Could not extract attention weights from the model.", color="danger")
        
        # Prepare flow data
        flow_data = {
            'factual_data': factual_attention_data['attention_weights'],
            'counterfactual_data': cf_attention_data['attention_weights'],
            'factual_tokens': factual_attention_data['tokens'],
            'cf_tokens': cf_attention_data['tokens'],
            'num_layers': len(factual_attention_data['attention_weights']),
            'analysis_type': 'attention'
        }
        
        # Create overview visualization
        overview_fig = create_overview_visualization(flow_data)
        
        # Return complete analysis layout
        return html.Div([
            # QA Results Comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📝 Factual Answer"),
                        dbc.CardBody([
                            html.P(factual_result.get('answer', 'No answer found'), className="lead"),
                            html.Small(f"Confidence: {factual_result.get('score', 0):.3f}", className="text-muted")
                        ])
                    ], color="success", outline=True)
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔄 Counterfactual Answer"),
                        dbc.CardBody([
                            html.P(cf_result.get('answer', 'No answer found'), className="lead"),
                            html.Small(f"Confidence: {cf_result.get('score', 0):.3f}", className="text-muted")
                        ])
                    ], color="warning", outline=True)
                ], width=6)
            ], className="mb-4"),
            
            # Overview Chart
            dbc.Card([
                dbc.CardHeader("🌊 Attention Flow Overview"),
                dbc.CardBody([
                    dcc.Store(id="cf-flow-data-store", data=flow_data),
                    dcc.Graph(
                        id="cf-flow-overview-graph",
                        figure=overview_fig,
                        config={'displayModeBar': True},
                        clickmode='event+select'
                    )
                ])
            ], className="mb-4"),
            
            # Layer Detail Section (updated by clicking)
            html.Div(id="cf-flow-layer-detail-section", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-click text-muted fa-2x mb-3"),
                            html.H5("Click on a layer above to see detailed analysis", className="text-muted"),
                            html.P("Each layer shows transformer mechanics: Q/K/V interactions, attention patterns, and information flow changes.", className="text-muted")
                        ], className="text-center py-4")
                    ])
                ], color="light")
            ])
        ])
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Analysis failed: {str(e)}", color="danger")

@callback(
    Output("cf-flow-layer-detail-section", "children"),
    [Input("cf-flow-overview-graph", "clickData"),
     Input("test-layer-btn", "n_clicks")],
    [State("cf-flow-data-store", "data")]
)
def update_layer_detail_section(click_data, test_btn, stored_data):
    """Update the layer detail section when a layer is clicked."""
    import json
    import numpy as np
    
    print(f"[DEBUG] Click detected! Click data: {click_data}, Test btn: {test_btn}")
    
    if not stored_data:
        return dbc.Alert("No analysis data available", color="warning")
    
    try:
        # Parse stored data (convert from JSON back to arrays)
        flow_data = {}
        for key, value in stored_data.items():
            if key in ['factual_data', 'counterfactual_data']:
                flow_data[key] = np.array(value)
            else:
                flow_data[key] = value
        
        # Determine layer index
        layer_idx = 0  # Default
        
        if test_btn and test_btn > 0:
            layer_idx = 0  # Test with layer 0
            print(f"[DEBUG] Test button clicked - using layer {layer_idx}")
        elif click_data and 'points' in click_data and len(click_data['points']) > 0:
            # Get clicked layer - handle both trace clicks
            point = click_data['points'][0]
            layer_idx = int(point['x'])
            print(f"[DEBUG] Layer clicked from graph - layer {layer_idx}, point data: {point}")
        else:
            print(f"[DEBUG] No valid click data, using default layer {layer_idx}")
        
        print(f"[DEBUG] Processing layer {layer_idx} detail (will show as Layer {layer_idx + 1} in UI)")
        
        # Call the enhanced layer detail analysis
        return create_enhanced_layer_detail(flow_data, layer_idx)
        
    except Exception as e:
        print(f"[DEBUG] Error in callback: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error loading layer detail: {str(e)}"
        ], color="danger")

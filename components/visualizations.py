"""
Reusable visualization components for the TinyBERT visualization app.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from dash import html, dcc
import dash_bootstrap_components as dbc
from config import VISUALIZATION_CONFIG
import plotly.figure_factory as ff

def create_attention_heatmap_matrix(tokens, attentions, layer_idx=0, head_idx=0, height=300):
    """
    Create matrix/heatmap style attention visualization
    
    Args:
        tokens: List of tokens
        attentions: Attention weights tensor
        layer_idx: Index of the layer to visualize
        head_idx: Index of the attention head to visualize
        height: Height of the visualization (default 300px)
        
    Returns:
        Plotly figure object
    """
    # Extract attention weights for the specified layer and head
    # Convert from list (if from JSON) back to numpy array
    if isinstance(attentions[0], list):
        attention_map = np.array(attentions[layer_idx])[0, head_idx]
    else:
        attention_map = attentions[layer_idx][0, head_idx]
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=attention_map,
        x=tokens,
        y=tokens,
        colorscale=VISUALIZATION_CONFIG["colorscale"],
        colorbar=dict(title='Attention Weight')
    ))
    
    fig.update_layout(
        title=f"Layer {layer_idx}, Head {head_idx}",
        xaxis_title="Target Tokens",
        yaxis_title="Source Tokens",
        height=height,
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=10),
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )
    
    return fig

def create_attention_heatmap_lines(tokens, attentions, layer_idx=0, head_idx=0):
    """
    Create attention visualization with lines
    
    Args:
        tokens: List of tokens
        attentions: Attention weights tensor
        layer_idx: Index of the layer to visualize
        head_idx: Index of the attention head to visualize
    
    Returns:
        Plotly figure with attention lines
    """
    # Handle different input formats - convert to numpy if needed
    if isinstance(attentions[0], list):
        attn_layer = np.array(attentions[layer_idx])[0, head_idx]
    else:
        # If it's already a tensor, convert to numpy
        try:
            attn_layer = attentions[layer_idx][0, head_idx].cpu().numpy()
        except:
            attn_layer = attentions[layer_idx][0, head_idx]
    
    # Define consistent line color
    line_color = "rgb(0,150,220)"
    
    # Create token positions (left side and right side)
    pos_left = [(0, i) for i in range(len(tokens))]
    pos_right = [(1, i) for i in range(len(tokens))]
    
    # Create figure
    fig = go.Figure()
    
    # Add lines between tokens (for attention)
    max_weight = attn_layer.max()
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            weight = attn_layer[i, j]
            
            # Skip very weak connections
            if weight < 0.03:
                continue
            
            width = weight / max_weight * 10
            opacity = min(weight + 0.3, 0.95)
            
            # Draw line
            fig.add_trace(
                go.Scatter(
                    x=[pos_left[i][0], pos_right[j][0]],
                    y=[pos_left[i][1], pos_right[j][1]],
                    mode="lines",
                    line=dict(color=line_color, width=width),
                    opacity=opacity,
                    hoverinfo="text",
                    text=f"{tokens[i]} → {tokens[j]}: {weight:.3f}",
                    showlegend=False
                )
            )
    
    # Add token markers
    for side, positions in [("Source", pos_left), ("Target", pos_right)]:
        for i, (x, y) in enumerate(positions):
            # Add token marker
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=8, color="lightgray", line=dict(color="black", width=1)),
                hoverinfo="text",
                text=f"{side}: {tokens[i]}",
                showlegend=False
            ))
            
            # Add token label
            fig.add_annotation(
                x=x, y=y,
                text=tokens[i][:8] + "..." if len(tokens[i]) > 8 else tokens[i],  # Truncate long tokens
                showarrow=False,
                font=dict(color="black", size=9),  # Smaller font for compact view
                xanchor="right" if side == "Source" else "left",
                yanchor="middle",
                xshift=-5 if side == "Source" else 5
            )
    
    # Update layout
    fig.update_layout(
        title=f"Layer {layer_idx}, Head {head_idx}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, len(tokens)]),
        height=300,  # Smaller height to fit in the area
        autosize=True,   # Use available width
        paper_bgcolor="white",  # Changed to white background
        plot_bgcolor="white",   # Changed to white background
        font=dict(color="black", size=10),  # Changed to black text
        margin=dict(l=40, r=40, t=40, b=40)  # Smaller margins
    )
    
    return fig

def create_attention_all_heads_grid(tokens, attentions, layer_idx):
    """
    Create a grid visualization of all attention heads for a single layer using line visualization
    
    Args:
        tokens: List of tokens
        attentions: Attention weights tensor
        layer_idx: Index of the layer to visualize
        
    Returns:
        Plotly figure with grid of attention visualizations
    """
    # Handle different input formats - convert to numpy if needed
    if isinstance(attentions[0], list):
        attention_layer = np.array(attentions[layer_idx])
    else:
        # If it's already a tensor, convert to numpy
        try:
            attention_layer = attentions[layer_idx].cpu().numpy()
        except:
            attention_layer = attentions[layer_idx]
    
    # Make sure we have a 3D array (batch, heads, seq_len, seq_len)
    if len(attention_layer.shape) == 2:
        # If we have a 2D array, reshape to add head dimension
        attention_layer = attention_layer.reshape(1, 1, *attention_layer.shape)
    elif len(attention_layer.shape) == 3:
        # If we have a 3D array (heads, seq_len, seq_len), add batch dimension
        attention_layer = attention_layer.reshape(1, *attention_layer.shape)
    
    # Get number of attention heads
    num_heads = attention_layer.shape[1]
    
    # Use 2×6 grid layout (2 rows, 6 columns)
    cols = 4
    rows = math.ceil(num_heads / cols)
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {h}" for h in range(num_heads)],
        vertical_spacing=0.05,
        horizontal_spacing=0.01  # Reduced spacing for more compact layout
    )
    
    # Use the same line color for consistency
    line_color = "rgb(0,150,220)"
    
    for head_idx in range(num_heads):
        row = head_idx // cols + 1
        col = head_idx % cols + 1
        
        attn_head = attention_layer[0, head_idx]
        
        # Add token positions for this subplot
        max_weight = attn_head.max()
        
        # Draw attention lines for this subplot
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                weight = attn_head[i, j]
                
                # Skip weak connections
                if weight < 0.05:
                    continue
                    
                # Use consistent line styling
                width = weight / max_weight * 5  # Slightly thinner for grid view
                opacity = min(weight + 0.3, 0.95)
                
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[i, j],
                        mode="lines",
                        line=dict(color=line_color, width=width),
                        opacity=opacity,
                        showlegend=False,
                        hoverinfo="text",
                        text=f"{tokens[i]} → {tokens[j]}: {weight:.3f}"
                    ),
                    row=row, col=col
                )
        
        # Add token labels for this subplot
        for i, token in enumerate(tokens):
            # Left side token labels (truncate long tokens)
            short_token = token[:5] + "..." if len(token) > 5 else token
            
            # Add left side label
            fig.add_annotation(
                x=-0.02, 
                y=i,
                text=short_token,
                showarrow=False,
                font=dict(color="white", size=8),
                xanchor="right",
                yanchor="middle",
                xref=f"x{head_idx+1}" if head_idx > 0 else "x",
                yref=f"y{head_idx+1}" if head_idx > 0 else "y"
            )
            
            # Add right side label
            fig.add_annotation(
                x=1.02, 
                y=i,
                text=short_token,
                showarrow=False,
                font=dict(color="white", size=8),
                xanchor="left",
                yanchor="middle",
                xref=f"x{head_idx+1}" if head_idx > 0 else "x",
                yref=f"y{head_idx+1}" if head_idx > 0 else "y"
            )
                
        # Update axes for this subplot
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1],
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, len(tokens) - 0.5],
            row=row, col=col
        )
    
    # Update overall layout
    fig.update_layout(
        height=400 * rows,  # Responsive height based on rows
        autosize=True,     # Enable responsive sizing
        paper_bgcolor="#0a1929",
        plot_bgcolor="#0a1929",
        margin=dict(l=10, r=10, t=40, b=10),  # Reduced margins
        font=dict(color="white"),
        title=f"All Attention Heads for Layer {layer_idx}"
    )
    
    return fig

def create_attention_all_matrices_grid(attentions, tokens, layer_idx=0):
    """
    Create a grid of heatmap matrices showing all attention heads in a layer
    
    Args:
        attentions: Attention weights tensor
        tokens: List of tokens
        layer_idx: Index of the layer to visualize
    
    Returns:
        Plotly figure showing all attention heads in the specified layer as matrices
    """
    # Handle different input formats - convert to numpy if needed
    if isinstance(attentions[0], list):
        attention_layer = np.array(attentions[layer_idx])
    else:
        # If it's already a tensor, convert to numpy
        try:
            attention_layer = attentions[layer_idx].cpu().numpy()
        except:
            attention_layer = attentions[layer_idx]
    
    # Make sure we have a 3D array (batch, heads, seq_len, seq_len)
    if len(attention_layer.shape) == 2:
        # If we have a 2D array, reshape to add head dimension
        attention_layer = attention_layer.reshape(1, 1, *attention_layer.shape)
    elif len(attention_layer.shape) == 3:
        # If we have a 3D array (heads, seq_len, seq_len), add batch dimension
        attention_layer = attention_layer.reshape(1, *attention_layer.shape)
    
    num_heads = attention_layer.shape[1]
    
    # Use 2×6 grid layout (2 rows, 6 columns)
    cols = 4
    rows = math.ceil(num_heads / cols)
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {h}" for h in range(num_heads)],
        vertical_spacing=0.05,
        horizontal_spacing=0.01  # Reduced spacing for more compact layout
    )
    
    # Use the same colorscale as in the single matrix view for consistency
    colorscale = VISUALIZATION_CONFIG["colorscale"]
    
    for head_idx in range(num_heads):
        row = head_idx // cols + 1
        col = head_idx % cols + 1
        
        attn_head = attention_layer[0, head_idx]
        
        # Create heatmap trace
        heatmap = go.Heatmap(
            z=attn_head,
            x=tokens,
            y=tokens,
            colorscale=colorscale,
            showscale=False,  # No color scale to save space
            hoverongaps=False,
            hoverinfo="text",
            text=[[f"{tokens[i]} → {tokens[j]}: {attn_head[i][j]:.3f}" 
                   for j in range(len(tokens))] 
                  for i in range(len(tokens))]
        )
        
        fig.add_trace(heatmap, row=row, col=col)
        
        # Update axes for this subplot
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=row, col=col
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=400 * rows,  # Responsive height based on rows
        autosize=True,     # Enable responsive sizing
        paper_bgcolor="#0a1929",
        plot_bgcolor="#0a1929",
        margin=dict(l=10, r=10, t=40, b=10),  # Reduced margins
        font=dict(color="white"),
        title=f"All Attention Matrices for Layer {layer_idx}"
    )
    
    return fig

def create_clickable_entropy_heatmap(entropy):
    """
    Create heatmap visualization of attention entropy with click functionality
    
    Args:
        entropy: 2D numpy array of entropy values [layers, heads]
        
    Returns:
        Plotly figure object
    """
    # Convert to numpy array if not already
    entropy = np.array(entropy)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=entropy,
        x=[f"Head {i}" for i in range(entropy.shape[1])],
        y=[f"Layer {i}" for i in range(entropy.shape[0])],
        colorscale="Viridis",
        colorbar=dict(title="Entropy"),
        hovertemplate='Layer: %{y}<br>Head: %{x}<br>Entropy: %{z:.3f}<extra>Click to view attention matrix</extra>'
    ))
    
    fig.update_layout(
        title="Attention Entropy by Layer and Head",
        xaxis_title="Attention Head",
        yaxis_title="Layer",
        height=400,
    )
    
    # Add custom data to support click events
    fig.update_traces(
        customdata=np.stack((
            np.tile(np.arange(entropy.shape[0]).reshape(-1, 1), (1, entropy.shape[1])),  # layer indices
            np.tile(np.arange(entropy.shape[1]), (entropy.shape[0], 1))  # head indices
        ), axis=-1)
    )
    
    return fig

def create_embedding_plot(tokens, token_embeddings, method="tsne"):
    """
    Create visualization of token embeddings
    
    Args:
        tokens: List of tokens
        token_embeddings: Token embeddings array
        method: Dimensionality reduction method ("tsne" or "pca")
        
    Returns:
        Plotly figure object
    """
    # Convert to numpy array if not already
    token_embeddings = np.array(token_embeddings)
    
    # Choose dimensionality reduction method
    if method == "pca" or len(tokens) <= 3:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(token_embeddings)
        method_name = "PCA"
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"Explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%}"
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, len(tokens) - 1))
        try:
            # Try with max_iter (newer scikit-learn versions)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        except TypeError:
            # Fall back to n_iter (older scikit-learn versions)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        embeddings_2d = reducer.fit_transform(token_embeddings)
        method_name = "t-SNE"
        subtitle = f"Perplexity: {perplexity}"
    
    # Create scatter plot with better styling
    fig = go.Figure(data=go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode="markers+text",
        marker=dict(
            size=12,
            color=list(range(len(tokens))),
            colorscale="Viridis",
            colorbar=dict(title="Token Position"),
            line=dict(width=1, color="white")
        ),
        text=tokens,
        textposition="top center",
        textfont=dict(size=10, color="black"),
        hovertemplate="<b>%{text}</b><br>Position: %{marker.color}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Token Embeddings Visualization ({method_name})<br><sub>{subtitle}</sub>",
        xaxis_title=f"{method_name} Dimension 1",
        yaxis_title=f"{method_name} Dimension 2",
        height=500,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
    )
    
    return fig

def highlight_lime_text(text, words, weights):
    """
    Highlight text based on LIME weights
    
    Args:
        text: Text string
        words: List of words from LIME explanation
        weights: List of weights from LIME explanation
        
    Returns:
        HTML component with highlighted text
    """
    # Create mapping of words to weights
    word_weights = {}
    for word, weight in zip(words, weights):
        if " " not in word:  # Only highlight single words
            word_weights[word.strip()] = weight
    
    # Split text by spaces but keep spaces
    words = []
    for word in text.split(" "):
        if word:
            words.append(word)
            words.append(" ")
    if words and words[-1] == " ":
        words = words[:-1]  # Remove trailing space
    
    # Create highlighted spans
    highlighted_text = []
    for word in words:
        if word.strip().lower() in word_weights:
            weight = word_weights[word.strip().lower()]
            if weight > 0:
                # Stronger green highlighting for positive weights
                alpha = min(0.9, abs(weight) * 3)  # Increased multiplier and max alpha
                color = f"rgba(40, 167, 69, {alpha})"  # Darker green
                border_color = "rgba(40, 167, 69, 1)"
            else:
                # Stronger red highlighting for negative weights
                alpha = min(0.9, abs(weight) * 3)  # Increased multiplier and max alpha
                color = f"rgba(220, 53, 69, {alpha})"  # Darker red
                border_color = "rgba(220, 53, 69, 1)"
            
            highlighted_text.append(html.Span(
                word, 
                style={
                    "background-color": color,
                    "border": f"2px solid {border_color}",
                    "border-radius": "4px",
                    "padding": "3px 6px",
                    "margin": "2px",
                    "font-weight": "bold",
                    "color": "white" if alpha > 0.6 else "black"
                }
            ))
        else:
            highlighted_text.append(word)
    
    return highlighted_text

def highlight_text(text, words, weights):
    """
    Highlight words in text based on their importance weights
    
    Args:
        text: Text string
        words: List of words
        weights: List of weights
        
    Returns:
        HTML div component with highlighted text
    """
    import re
    
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

def create_lime_bar_chart(words, weights):
    """
    Create a bar chart for LIME word importance
    
    Args:
        words: List of words from LIME explanation
        weights: List of weights from LIME explanation
        
    Returns:
        Plotly figure object
    """
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
    
    return fig

def create_entity_visualization(tokens, entity_labels):
    """
    Create visualization of named entities
    
    Args:
        tokens: List of tokens
        entity_labels: List of entity labels
        
    Returns:
        HTML div component with highlighted entities
    """
    # Map entity labels to colors and readable names
    entity_colors = {
        "O": "transparent",
        "B-PER": "#8dd3c7",
        "I-PER": "#8dd3c7",
        "B-ORG": "#fb8072",
        "I-ORG": "#fb8072",
        "B-LOC": "#80b1d3",
        "I-LOC": "#80b1d3",
        "B-MISC": "#fdb462",
        "I-MISC": "#fdb462"
    }
    
    entity_types = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Miscellaneous"
    }
    
    # Count entities by type
    entity_counts = {}
    for label in entity_labels:
        if label != "O":
            entity_type = label.split("-")[1]
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += 1
    
    # Create visualization
    entity_spans = []
    current_entity = None
    entity_words = []
    
    for token, label in zip(tokens, entity_labels):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"] or (token.startswith('[') and token.endswith(']')):
            continue
            
        if label.startswith("B-"):
            # End previous entity if there was one
            if current_entity:
                color = entity_colors.get(current_entity, "transparent")
                entity_spans.append(html.Span(" ".join(entity_words), 
                                            style={"background-color": color,
                                                  "padding": "2px 5px",
                                                  "margin": "2px",
                                                  "border-radius": "3px"}))
                entity_spans.append(" ")
                entity_words = []
            
            # Start new entity
            current_entity = label
            entity_words = [token.replace("##", "")]
            
        elif label.startswith("I-"):
            if current_entity:
                entity_words.append(token.replace("##", ""))
            else:
                entity_spans.append(token.replace("##", ""))
                entity_spans.append(" ")
                
        elif label == "O":
            # End previous entity if there was one
            if current_entity:
                color = entity_colors.get(current_entity, "transparent")
                entity_spans.append(html.Span(" ".join(entity_words), 
                                            style={"background-color": color,
                                                  "padding": "2px 5px",
                                                  "margin": "2px", 
                                                  "border-radius": "3px"}))
                entity_spans.append(" ")
                current_entity = None
                entity_words = []
            
            # Just add the token
            entity_spans.append(token.replace("##", ""))
            entity_spans.append(" ")
    
    # Add the final entity if there is one
    if current_entity:
        color = entity_colors.get(current_entity, "transparent")
        entity_spans.append(html.Span(" ".join(entity_words), 
                                    style={"background-color": color,
                                          "padding": "2px 5px",
                                          "margin": "2px", 
                                          "border-radius": "3px"}))
    
    # Create legend
    legend_items = []
    for entity_type, readable_name in entity_types.items():
        if entity_type in entity_counts:
            color = entity_colors.get(f"B-{entity_type}", "transparent")
            legend_items.append(html.Div([
                html.Span("   ", style={"background-color": color, "padding": "2px 10px", "margin-right": "5px"}),
                f"{readable_name} ({entity_counts.get(entity_type, 0)})"
            ], style={"margin-right": "15px", "display": "inline-block"}))
    
    # Create entity table
    entity_table = []
    current_entity = None
    entity_text = ""
    
    for token, label in zip(tokens, entity_labels):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"] or (token.startswith('[') and token.endswith(']')):
            continue
            
        if label.startswith("B-"):
            # End previous entity
            if current_entity:
                entity_type = current_entity.split("-")[1]
                entity_table.append(html.Tr([
                    html.Td(entity_text),
                    html.Td(entity_types.get(entity_type, entity_type))
                ]))
            
            # Start new entity
            current_entity = label
            entity_text = token.replace("##", "")
            
        elif label.startswith("I-"):
            if current_entity:
                entity_text += " " + token.replace("##", "")
                
        elif label == "O":
            # End previous entity
            if current_entity:
                entity_type = current_entity.split("-")[1]
                entity_table.append(html.Tr([
                    html.Td(entity_text),
                    html.Td(entity_types.get(entity_type, entity_type))
                ]))
                current_entity = None
    
    # Add the final entity if there is one
    if current_entity:
        entity_type = current_entity.split("-")[1]
        entity_table.append(html.Tr([
            html.Td(entity_text),
            html.Td(entity_types.get(entity_type, entity_type))
        ]))
    
    # Create result component
    result = html.Div([
        html.H5("Identified Entities in Text:"),
        html.Div(legend_items, className="mb-3"),
        html.Div(entity_spans, className="entity-text p-3 border rounded mb-4"),
    ])
    
    # Add entity table if we have entities
    if entity_table:
        result.children.append(
            html.Div([
                html.H5("Extracted Entities", className="mt-4"),
                dbc.Table(
                    [
                        html.Thead([
                            html.Tr([
                                html.Th("Entity Text"),
                                html.Th("Entity Type")
                            ])
                        ]),
                        html.Tbody(entity_table)
                    ],
                    bordered=True,
                    hover=True,
                    striped=True,
                    className="mb-4"
                )
            ])
        )
    
    return result

def create_heatmap(tokens, attention_weights, colorscale='Viridis'):
    """
    Create an attention heatmap visualization.
    
    Args:
        tokens: List of tokens
        attention_weights: 2D numpy array of attention weights
        colorscale: Plotly colorscale name
        
    Returns:
        Plotly figure object
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens,
        y=tokens,
        colorscale=colorscale,
        colorbar=dict(title="Attention Weight"),
    ))
    
    # Update layout
    fig.update_layout(
        title="Attention Weights Visualization",
        xaxis=dict(
            title="Target Tokens",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Source Tokens",
        ),
        height=600,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    return fig

def create_token_bar_chart(tokens, values, title, color='blue'):
    """
    Create a bar chart for token values (e.g. attributions).
    
    Args:
        tokens: List of tokens
        values: List of values for each token
        title: Chart title
        color: Bar color
        
    Returns:
        Plotly figure object
    """
    # Create bar chart
    fig = go.Figure(data=go.Bar(
        x=tokens,
        y=values,
        marker_color=color,
        text=[f"{v:.3f}" for v in values],
        textposition='outside',
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Tokens",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Value",
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    return fig

def create_entity_visualization(entities, text):
    """
    Create visualization of named entities.
    
    Args:
        entities: List of dictionaries, each representing an entity with 'text', 'label', 'start_idx', 'end_idx', 'score'.
        text: The original text string.
        
    Returns:
        HTML div component with highlighted entities and a table summary.
    """
    # Create a marked-up text display
    text_with_entities = []
    last_end = 0
    
    # Sort entities by start index
    sorted_entities = sorted(entities, key=lambda e: e.get("start_idx", 0))
    
    for entity in sorted_entities:
        # Add text before this entity
        if entity.get("start_idx", 0) > last_end:
            text_with_entities.append(text[last_end:entity.get("start_idx", last_end)])
        
        # Add the entity with appropriate styling
        entity_span = html.Mark(
            entity.get("text", ""),
            className=f"entity-{entity.get('label', '').lower()}",
            title=f"{entity.get('label', '')}: {entity.get('score', 0):.2f}"
        )
        text_with_entities.append(entity_span)
        
        # Update last end position
        last_end = entity.get("end_idx", last_end)
    
    # Add any remaining text
    if last_end < len(text):
        text_with_entities.append(text[last_end:])
    
    # Create a table to display the entities
    table_header = [
        html.Thead(html.Tr([
            html.Th("Entity Text"),
            html.Th("Type"),
            html.Th("Confidence"),
            html.Th("Position")
        ]))
    ]
    
    rows = []
    for entity in entities:
        entity_type = entity["label"]
        confidence = f"{entity['score']*100:.1f}%"
        position = f"{entity['start_idx']}-{entity['end_idx']}"
        
        rows.append(html.Tr([
            html.Td(entity["text"]),
            html.Td(html.Mark(entity_type, className=f"entity-{entity_type.lower()}")),
            html.Td(confidence),
            html.Td(position)
        ]))
    
    table_body = [html.Tbody(rows)]
    
    entity_table_card = dbc.Card([
        dbc.CardHeader(f"Found {len(entities)} Entities"),
        dbc.CardBody([
            dbc.Table(
                table_header + table_body,
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
            ),
        ])
    ], className="mb-4 shadow-sm")
    
    text_card = dbc.Card([
        dbc.CardHeader("Text with Highlighted Entities"),
        dbc.CardBody([
            html.P(text_with_entities, className="mb-3"),
            html.Small([
                "Hover over highlighted entities to see confidence scores. ",
                "Colors represent entity types as shown in the legend below."
            ], className="text-muted")
        ])
    ], className="shadow-sm")
    
    # Entity type legend
    legend_card = dbc.Card([
        dbc.CardHeader("Entity Type Legend"),
        dbc.CardBody([
            html.Div([
                html.Mark("Person", className="entity-per me-2"),
                html.Span("Person (individuals, people names)", className="me-4"),
                
                html.Mark("Organization", className="entity-org me-2"),
                html.Span("Organization (companies, institutions)", className="me-4"),
            ], className="mb-2"),
            html.Div([
                html.Mark("Location", className="entity-loc me-2"),
                html.Span("Location (cities, countries, places)", className="me-4"),
                
                html.Mark("Miscellaneous", className="entity-misc me-2"),
                html.Span("Miscellaneous (other entities)", className="me-4"),
            ], className="mb-2"),
            html.Div([
                html.Mark("Date/Time", className="entity-date me-2"),
                html.Span("Date/Time (temporal expressions)", className="me-4"),
            ]),
        ]),
    ], className="mt-4 shadow-sm")

    # Create entity distribution chart
    entity_types_count = {}
    for entity in entities:
        entity_type = entity["label"]
        if entity_type not in entity_types_count:
            entity_types_count[entity_type] = 0
        entity_types_count[entity_type] += 1
    
    # Define colors for entity types
    colors = {
        "PER": "rgba(255, 99, 132, 0.7)",
        "ORG": "rgba(54, 162, 235, 0.7)",
        "LOC": "rgba(255, 206, 86, 0.7)",
        "MISC": "rgba(75, 192, 192, 0.7)",
        "DATE": "rgba(153, 102, 255, 0.7)",
        "TIME": "rgba(153, 102, 255, 0.7)",
        "PERSON": "rgba(255, 99, 132, 0.7)",
        "ORGANIZATION": "rgba(54, 162, 235, 0.7)",
        "LOCATION": "rgba(255, 206, 86, 0.7)",
        "MISCELLANEOUS": "rgba(75, 192, 192, 0.7)",
    }
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(entity_types_count.keys()),
        values=list(entity_types_count.values()),
        marker=dict(colors=[colors.get(t, "rgba(128, 128, 128, 0.7)") for t in entity_types_count.keys()]),
        textinfo='label+percent',
        insidetextorientation='radial',
    )])
    
    fig.update_layout(
        title="Entity Type Distribution",
        height=400,
    )
    
    chart_card = dbc.Card([
        dbc.CardHeader("Entity Distribution"),
        dbc.CardBody([
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': True}
            ),
        ])
    ], className="mt-4 shadow-sm")

    return html.Div([
        entity_table_card,
        text_card,
        legend_card,
        chart_card
    ])

def create_heatmap(tokens, attention_weights, colorscale='Viridis'):
    """
    Create an attention heatmap visualization.
    
    Args:
        tokens: List of tokens
        attention_weights: 2D numpy array of attention weights
        colorscale: Plotly colorscale name
        
    Returns:
        Plotly figure object
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens,
        y=tokens,
        colorscale=colorscale,
        colorbar=dict(title='Attention Weight'),
    ))
    
    # Update layout
    fig.update_layout(
        title="Attention Weights Visualization",
        xaxis=dict(
            title="Target Tokens",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Source Tokens",
        ),
        height=600,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    return fig

def create_entropy_heatmap(entropy_values, title="Attention Entropy"):
    """
    Create a heatmap of entropy values across layers and heads.
    
    Args:
        entropy_values: 2D numpy array of entropy values [layers, heads]
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Get dimensions
    num_layers, num_heads = entropy_values.shape
    
    # Create x and y labels
    y_labels = [f"Layer {i+1}" for i in range(num_layers)]
    x_labels = [f"Head {i+1}" for i in range(num_heads)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=entropy_values,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r',  # Red (high entropy) to Blue (low entropy)
        colorbar=dict(title="Entropy"),
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Attention Heads"),
        yaxis=dict(title="Layers"),
        height=400,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    return fig

def create_layer_head_selector(num_layers, num_heads):
    """
    Create layer and head selector components.
    
    Args:
        num_layers: Number of layers in the model
        num_heads: Number of attention heads per layer
        
    Returns:
        Dash component with layer and head selectors
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Layer:"),
                dcc.Slider(
                    id="layer-slider",
                    min=0,
                    max=num_layers-1,
                    value=0,
                    marks={i: f"{i+1}" for i in range(num_layers)},
                    step=1,
                ),
            ], md=6),
            dbc.Col([
                html.Label("Head:"),
                dcc.Slider(
                    id="head-slider",
                    min=0,
                    max=num_heads-1,
                    value=0,
                    marks={i: f"{i+1}" for i in range(num_heads)},
                    step=1,
                ),
            ], md=6),
        ], className="mb-3"),
    ])

def create_visualization_card(title, figure, description=None):
    """
    Create a card to display a visualization.
    
    Args:
        title: Card title
        figure: Plotly figure object
        description: Optional description text
        
    Returns:
        dbc.Card component
    """
    card_content = [
        dbc.CardHeader(html.H5(title, className="card-title")),
        dbc.CardBody([
            dcc.Graph(
                figure=figure,
                config={"responsive": True},
            ),
        ]),
    ]
    
    if description:
        card_content[1].children.append(
            html.P(description, className="text-muted mt-3 small")
        )
    
    return dbc.Card(card_content, className="mb-4 shadow-sm")

def create_entity_visualization(entities, text):
    """
    Create visualization of named entities.
    
    Args:
        entities: List of dictionaries, each representing an entity with 'text', 'label', 'start_idx', 'end_idx', 'score'.
        text: The original text string.
        
    Returns:
        HTML div component with highlighted entities and a table summary.
    """
    # Create a marked-up text display
    text_with_entities = []
    last_end = 0
    
    # Sort entities by start index
    sorted_entities = sorted(entities, key=lambda e: e.get("start_idx", 0))
    
    for entity in sorted_entities:
        # Add text before this entity
        if entity.get("start_idx", 0) > last_end:
            text_with_entities.append(text[last_end:entity.get("start_idx", last_end)])
        
        # Add the entity with appropriate styling
        entity_span = html.Mark(
            entity.get("text", ""),
            className=f"entity-{entity.get('label', '').lower()}",
            title=f"{entity.get('label', '')}: {entity.get('score', 0):.2f}"
        )
        text_with_entities.append(entity_span)
        
        # Update last end position
        last_end = entity.get("end_idx", last_end)
    
    # Add any remaining text
    if last_end < len(text):
        text_with_entities.append(text[last_end:])
    
    # Create a table to display the entities
    table_header = [
        html.Thead(html.Tr([
            html.Th("Entity Text"),
            html.Th("Type"),
            html.Th("Confidence"),
            html.Th("Position")
        ]))
    ]
    
    rows = []
    for entity in entities:
        entity_type = entity["label"]
        confidence = f"{entity['score']*100:.1f}%"
        position = f"{entity['start_idx']}-{entity['end_idx']}"
        
        rows.append(html.Tr([
            html.Td(entity["text"]),
            html.Td(html.Mark(entity_type, className=f"entity-{entity_type.lower()}")),
            html.Td(confidence),
            html.Td(position)
        ]))
    
    table_body = [html.Tbody(rows)]
    
    entity_table_card = dbc.Card([
        dbc.CardHeader(f"Found {len(entities)} Entities"),
        dbc.CardBody([
            dbc.Table(
                table_header + table_body,
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
            ),
        ])
    ], className="mb-4 shadow-sm")
    
    text_card = dbc.Card([
        dbc.CardHeader("Text with Highlighted Entities"),
        dbc.CardBody([
            html.P(text_with_entities, className="mb-3"),
            html.Small([
                "Hover over highlighted entities to see confidence scores. ",
                "Colors represent entity types as shown in the legend below."
            ], className="text-muted")
        ])
    ], className="shadow-sm")
    
    # Entity type legend
    legend_card = dbc.Card([
        dbc.CardHeader("Entity Type Legend"),
        dbc.CardBody([
            html.Div([
                html.Mark("Person", className="entity-per me-2"),
                html.Span("Person (individuals, people names)", className="me-4"),
                
                html.Mark("Organization", className="entity-org me-2"),
                html.Span("Organization (companies, institutions)", className="me-4"),
            ], className="mb-2"),
            html.Div([
                html.Mark("Location", className="entity-loc me-2"),
                html.Span("Location (cities, countries, places)", className="me-4"),
                
                html.Mark("Miscellaneous", className="entity-misc me-2"),
                html.Span("Miscellaneous (other entities)", className="me-4"),
            ], className="mb-2"),
            html.Div([
                html.Mark("Date/Time", className="entity-date me-2"),
                html.Span("Date/Time (temporal expressions)", className="me-4"),
            ]),
        ]),
    ], className="mt-4 shadow-sm")

    # Create entity distribution chart
    entity_types_count = {}
    for entity in entities:
        entity_type = entity["label"]
        if entity_type not in entity_types_count:
            entity_types_count[entity_type] = 0
        entity_types_count[entity_type] += 1
    
    # Define colors for entity types
    colors = {
        "PER": "rgba(255, 99, 132, 0.7)",
        "ORG": "rgba(54, 162, 235, 0.7)",
        "LOC": "rgba(255, 206, 86, 0.7)",
        "MISC": "rgba(75, 192, 192, 0.7)",
        "DATE": "rgba(153, 102, 255, 0.7)",
        "TIME": "rgba(153, 102, 255, 0.7)",
        "PERSON": "rgba(255, 99, 132, 0.7)",
        "ORGANIZATION": "rgba(54, 162, 235, 0.7)",
        "LOCATION": "rgba(255, 206, 86, 0.7)",
        "MISCELLANEOUS": "rgba(75, 192, 192, 0.7)",
    }
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(entity_types_count.keys()),
        values=list(entity_types_count.values()),
        marker=dict(colors=[colors.get(t, "rgba(128, 128, 128, 0.7)") for t in entity_types_count.keys()]),
        textinfo='label+percent',
        insidetextorientation='radial',
    )])
    
    fig.update_layout(
        title="Entity Type Distribution",
        height=400,
    )
    
    chart_card = dbc.Card([
        dbc.CardHeader("Entity Distribution"),
        dbc.CardBody([
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': True}
            ),
        ])
    ], className="mt-4 shadow-sm")

    return html.Div([
        entity_table_card,
        text_card,
        legend_card,
        chart_card
    ])

def create_entropy_heatmap(entropy_values, title="Attention Entropy"):
    """
    Create a heatmap of entropy values across layers and heads.
    
    Args:
        entropy_values: 2D numpy array of entropy values [layers, heads]
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Get dimensions
    num_layers, num_heads = entropy_values.shape
    
    # Create x and y labels
    y_labels = [f"Layer {i+1}" for i in range(num_layers)]
    x_labels = [f"Head {i+1}" for i in range(num_heads)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=entropy_values,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r',  # Red (high entropy) to Blue (low entropy)
        colorbar=dict(title="Entropy"),
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Attention Heads"),
        yaxis=dict(title="Layers"),
        height=400,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    return fig

def create_layer_head_selector(num_layers, num_heads):
    """
    Create layer and head selector components.
    
    Args:
        num_layers: Number of layers in the model
        num_heads: Number of attention heads per layer
        
    Returns:
        Dash component with layer and head selectors
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Layer:"),
                dcc.Slider(
                    id="layer-slider",
                    min=0,
                    max=num_layers-1,
                    value=0,
                    marks={i: f"{i+1}" for i in range(num_layers)},
                    step=1,
                ),
            ], md=6),
            dbc.Col([
                html.Label("Head:"),
                dcc.Slider(
                    id="head-slider",
                    min=0,
                    max=num_heads-1,
                    value=0,
                    marks={i: f"{i+1}" for i in range(num_heads)},
                    step=1,
                ),
            ], md=6),
        ], className="mb-3"),
    ])

def create_visualization_card(title, figure, description=None):
    """
    Create a card to display a visualization.
    
    Args:
        title: Card title
        figure: Plotly figure object
        description: Optional description text
        
    Returns:
        dbc.Card component
    """
    card_content = [
        dbc.CardHeader(html.H5(title, className="card-title")),
        dbc.CardBody([
            dcc.Graph(
                figure=figure,
                config={"responsive": True},
            ),
        ]),
    ]
    
    if description:
        card_content[1].children.append(
            html.P(description, className="text-muted mt-3 small")
        )
    
    return dbc.Card(card_content, className="mb-4 shadow-sm")

def create_logit_heatmap(logit_data, task_type="sentiment"):
    """
    Create a logit matrix heatmap visualization.
    
    Args:
        logit_data: Dictionary containing logit matrix data
        task_type: Type of task ("sentiment" or "ner")
        
    Returns:
        Plotly figure object
    """
    if not logit_data:
        return go.Figure()
    
    logits = np.array(logit_data["logits"])
    class_names = logit_data["class_names"]
    
    if task_type == "sentiment":
        # For sentiment: single prediction, show as bar chart and comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Logit Scores", "Probabilities", "Logit vs Probability", "Class Comparison"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Logit scores bar chart
        fig.add_trace(
            go.Bar(x=class_names, y=logits, name="Logits", 
                   marker_color=['#ff6b6b' if i == logit_data["predicted_class_idx"] else '#4ecdc4' 
                                for i in range(len(logits))]),
            row=1, col=1
        )
        
        # Probabilities bar chart
        probabilities = logit_data["probabilities"]
        fig.add_trace(
            go.Bar(x=class_names, y=probabilities, name="Probabilities",
                   marker_color=['#ff6b6b' if i == logit_data["predicted_class_idx"] else '#4ecdc4' 
                                for i in range(len(probabilities))]),
            row=1, col=2
        )
        
        # Logit vs Probability scatter
        fig.add_trace(
            go.Scatter(x=logits, y=probabilities, mode='markers+text',
                      text=class_names, textposition="top center",
                      marker=dict(size=12, color=['#ff6b6b' if i == logit_data["predicted_class_idx"] else '#4ecdc4' 
                                                 for i in range(len(logits))]),
                      name="Logit-Prob Relation"),
            row=2, col=1
        )
        
        # Class comparison (difference from max)
        max_logit = max(logits)
        logit_diff = [max_logit - logit for logit in logits]
        fig.add_trace(
            go.Bar(x=class_names, y=logit_diff, name="Distance from Max",
                   marker_color=['#45b7d1' if i != logit_data["predicted_class_idx"] else '#96ceb4' 
                                for i in range(len(logit_diff))]),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Logit Analysis for: '{logit_data['text'][:50]}...'",
            height=600,
            showlegend=False
        )
        
    else:  # NER task
        # For NER: token-level predictions, show as heatmap
        tokens = logit_data["tokens"]
        
        # Filter out special tokens for cleaner visualization
        special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
        valid_indices = [i for i, token in enumerate(tokens) if token not in special_tokens]
        
        if valid_indices:
            filtered_logits = logits[valid_indices]
            filtered_tokens = [tokens[i] for i in valid_indices]
        else:
            filtered_logits = logits
            filtered_tokens = tokens
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=filtered_logits.T,  # Transpose to have classes on Y-axis
            x=[token.replace("##", "") for token in filtered_tokens],
            y=class_names,
            colorscale="RdBu_r",
            colorbar=dict(title="Logit Score"),
            hoverongaps=False,
            hovertemplate="Token: %{x}<br>Class: %{y}<br>Logit: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Token-Level Logit Matrix for: '{logit_data['text'][:50]}...'",
            xaxis_title="Tokens",
            yaxis_title="Entity Classes",
            height=400 + len(class_names) * 20,  # Dynamic height based on classes
            xaxis=dict(tickangle=45)
        )
    
    return fig

def create_logit_comparison_chart(logit_data, task_type="sentiment"):
    """
    Create a comparison chart showing logits vs probabilities.
    
    Args:
        logit_data: Dictionary containing logit matrix data
        task_type: Type of task ("sentiment" or "ner")
        
    Returns:
        Plotly figure object
    """
    if not logit_data:
        return go.Figure()
    
    if task_type == "sentiment":
        logits = logit_data["logits"]
        probabilities = logit_data["probabilities"]
        class_names = logit_data["class_names"]
        
        fig = go.Figure()
        
        # Add logits trace
        fig.add_trace(go.Bar(
            x=class_names,
            y=logits,
            name="Logits (Raw Scores)",
            marker_color='rgba(55, 128, 191, 0.7)',
            yaxis='y'
        ))
        
        # Add probabilities trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=class_names,
            y=probabilities,
            mode='markers+lines',
            name="Probabilities (After Softmax)",
            marker=dict(size=10, color='rgba(219, 64, 82, 0.7)'),
            line=dict(color='rgba(219, 64, 82, 0.7)', width=3),
            yaxis='y2'
        ))
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title="Logits vs Probabilities Comparison",
            xaxis_title="Classes",
            yaxis=dict(title="Logit Scores", side="left"),
            yaxis2=dict(title="Probability", side="right", overlaying="y", range=[0, 1]),
            height=400,
            hovermode='x unified'
        )
        
    else:  # NER task - show average logits per class
        logits = np.array(logit_data["logits"])
        probabilities = np.array(logit_data["probabilities"])
        class_names = logit_data["class_names"]
        
        # Calculate average logits and probabilities per class
        avg_logits = np.mean(logits, axis=0)
        avg_probs = np.mean(probabilities, axis=0)
        
        fig = go.Figure()
        
        # Add average logits trace
        fig.add_trace(go.Bar(
            x=class_names,
            y=avg_logits,
            name="Average Logits",
            marker_color='rgba(55, 128, 191, 0.7)',
            yaxis='y'
        ))
        
        # Add average probabilities trace
        fig.add_trace(go.Scatter(
            x=class_names,
            y=avg_probs,
            mode='markers+lines',
            name="Average Probabilities",
            marker=dict(size=8, color='rgba(219, 64, 82, 0.7)'),
            line=dict(color='rgba(219, 64, 82, 0.7)', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Average Logits vs Probabilities per Entity Class",
            xaxis_title="Entity Classes",
            yaxis=dict(title="Average Logit Score", side="left"),
            yaxis2=dict(title="Average Probability", side="right", overlaying="y", range=[0, 1]),
            height=400,
            hovermode='x unified',
            xaxis=dict(tickangle=45)
        )
    
    return fig
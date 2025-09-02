"""
Helpers to generate BertViz HTML for QA model visualization.
"""
import logging
from typing import Tuple, List, Optional
import torch

logger = logging.getLogger(__name__)


def _truncate_pair(tokenizer, sentence_a: str, sentence_b: str, max_tokens: int = 128):
    # Tokenize pair with truncation keeping both sides
    return tokenizer.encode_plus(
        sentence_a,
        sentence_b,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        max_length=max_tokens,
        return_token_type_ids=True,
        return_attention_mask=True,
    )


def get_attention_for_pair(model, tokenizer, sentence_a: str, sentence_b: str, max_tokens: int = 128):
    """
    Run the model on (sentence_a, sentence_b) and return (attention, tokens, sentence_b_start)
    Attention shape expected by bertviz: list[tensor] per layer or tensor [layers, batch, heads, seq, seq]
    """
    try:
        # Ensure attentions are returned
        if getattr(model.config, 'output_attentions', False) is not True:
            model.config.output_attentions = True

        inputs = _truncate_pair(tokenizer, sentence_a, sentence_b, max_tokens)
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids')  # May be None for some models
        attention_mask = inputs.get('attention_mask')

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # HF models return attentions as last element or in outputs.attentions
        attentions = None
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions
        else:
            # Fallback: last element
            attentions = outputs[-1]

        # Convert input ids to tokens
        input_id_list = input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)

        # Determine sentence_b_start
        if token_type_ids is not None:
            # Typical for BERT: token_type_ids 0 for sentence A, 1 for sentence B
            tti = token_type_ids[0].tolist()
            sentence_b_start = tti.index(1) if 1 in tti else 0
        else:
            # For models without token_type_ids (e.g., DistilBERT), find first [SEP]
            try:
                sep_index = tokens.index('[SEP]')
                sentence_b_start = sep_index + 1
            except ValueError:
                sentence_b_start = 0

        return attentions, tokens, sentence_b_start
    except Exception as e:
        logger.error(f"Error generating attentions: {e}")
        raise


def build_model_view_html(attention, tokens: List[str], sentence_b_start: int) -> str:
    """
    Build HTML for BertViz model_view (overview).
    """
    try:
        from bertviz import model_view
        # model_view returns an HTML object for notebooks; get data string via _repr_html_ if present
        html_obj = model_view(attention, tokens, sentence_b_start=sentence_b_start)
        html_str = None
        # Try common attributes
        if hasattr(html_obj, 'data') and isinstance(html_obj.data, str):
            html_str = html_obj.data
        elif hasattr(html_obj, '_repr_html_'):
            html_str = html_obj._repr_html_()
        else:
            # Fallback to string
            html_str = str(html_obj)
        return html_str
    except Exception as e:
        logger.error(f"Error building model_view HTML: {e}")
        raise


def build_head_view_html(attention, tokens: List[str], sentence_b_start: int, layer: int = 4, head: int = 3) -> str:
    """
    Build HTML for BertViz head_view (detail zoom on a layer/head).
    """
    try:
        from bertviz import head_view
        html_obj = head_view(attention, tokens, sentence_b_start=sentence_b_start, layer=layer, heads=[head])
        if hasattr(html_obj, 'data') and isinstance(html_obj.data, str):
            return html_obj.data
        if hasattr(html_obj, '_repr_html_'):
            return html_obj._repr_html_()
        return str(html_obj)
    except Exception as e:
        logger.error(f"Error building head_view HTML: {e}")
        raise

import torch
from tqdm import tqdm
from utils import PromptUtils
import random 

# Global variables for hook
_head_scores_mrr = None
_head_scores_mass = None
_q_start = 0
_q_end = 0
_item_spans = None
_gold_tool_id = 0
_current_layer_idx = 0
_device = None

def _code3_attn_hook(module, args, kwargs, output):
    global _head_scores_mrr, _head_scores_mass, _q_start, _q_end, _item_spans, _gold_tool_id, _current_layer_idx, _device
    attn_weights = output[1]
    if attn_weights is not None:
        attn_layer = attn_weights.squeeze(0)  # (H, N, N)
        query_attn = attn_layer[:, _q_start:_q_end, :]  # (H, query_len, N)
        
        num_heads = attn_layer.size(0)
        doc_scores_all_heads = torch.zeros(num_heads, len(_item_spans), device=_device)
        for i, (d_start, d_end) in enumerate(_item_spans):
            doc_scores_all_heads[:, i] = query_attn[:, :, d_start:d_end].sum(dim=(-1, -2))
            
        ranked = torch.argsort(doc_scores_all_heads, dim=-1, descending=True)
        gold_mask = (ranked == _gold_tool_id)
        gold_ranks = gold_mask.float().argmax(dim=-1)
        
        # Accumulate MRR Approach
        _head_scores_mrr[_current_layer_idx] += 1.0 / (gold_ranks.float() + 1)
        
        # Accumulate RAW Mass Approach (raw scalar sum of query-to-gold attention)
        d_start, d_end = _item_spans[_gold_tool_id]
        gold_mass = query_attn[:, :, d_start:d_end].sum(dim=(-1, -2))
        _head_scores_mass[_current_layer_idx] += gold_mass
        
        _current_layer_idx += 1
        
        # Free memory!
        attn_weights.untyped_storage().resize_(0)
    return output


def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=30):
    """ Returns a dict with Top-K heads optimized by different strategies. """
    global _head_scores_mrr, _head_scores_mass, _q_start, _q_end, _item_spans, _gold_tool_id, _current_layer_idx, _device
    
    _device = device
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    _head_scores_mrr = torch.zeros(num_layers, num_heads, device=device)
    _head_scores_mass = torch.zeros(num_layers, num_heads, device=device)

    # Register hooks
    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(_code3_attn_hook, with_kwargs=True))

    for qix in tqdm(range(len(train_queries))):
        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils = PromptUtils(tokenizer=tokenizer, doc_ids=tool_ids, dict_all_docs=tools)
        
        _item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        _gold_tool_id = map_docname_id[gold_tool_name]
        query_prompt = f"Query: {question}" + "\nCorrect tool_id:"
        query_length = len(tokenizer(query_prompt, add_special_tokens=False).input_ids)
        total_length = inputs.input_ids.shape[1]
        
        _q_end = total_length - putils.prompt_suffix_length
        _q_start = _q_end - query_length
        _current_layer_idx = 0

        with torch.no_grad():
            _ = model(**inputs, output_attentions=True) 

        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    def get_top_k(scores_tensor, k=max_heads):
        flat_scores = scores_tensor.view(-1)
        top_indices = torch.argsort(flat_scores, descending=True)[:k]
        heads = []
        for idx in top_indices:
            layer = idx.item() // num_heads
            head = idx.item() % num_heads
            heads.append((layer, head))
        return heads

    return {
        "mrr": get_top_k(_head_scores_mrr, max_heads),
        "attn_mass": get_top_k(_head_scores_mass, max_heads)
    }
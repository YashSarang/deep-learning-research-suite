import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import time
import random
import numpy as np
import torch
import json
from tqdm import tqdm
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items
from code3 import select_retrieval_heads

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Globals for hooking
_q_start = 0
_q_end = 0
_item_spans = None
_selected_heads_by_layer = {}
_doc_scores = None

def _run3_attn_hook(module, args, kwargs, output):
    global _q_start, _q_end, _item_spans, _selected_heads_by_layer, _doc_scores
    attn_weights = output[1]
    if attn_weights is not None:
        layer_idx = module._layer_idx  # Must attach this manually!
        
        if layer_idx in _selected_heads_by_layer:
            for head_idx in _selected_heads_by_layer[layer_idx]:
                attn_head = attn_weights[0, head_idx]
                query_attn = attn_head[_q_start:_q_end, :]
                for i, (d_start, d_end) in enumerate(_item_spans):
                    _doc_scores[i] += query_attn[:, d_start:d_end].sum()
        
        # Free memory!
        attn_weights.untyped_storage().resize_(0)
    return output

def query_to_docs_attention_heads(attentions, query_span, doc_spans, selected_heads):
    # This is a dummy for signature preservation
    pass

def get_query_span(putils, query, total_length):
    query_prompt = f"Query: {query}" + "\nCorrect tool_id:"
    query_length = len(putils.tokenizer(query_prompt, add_special_tokens=False).input_ids)
    query_end = total_length - putils.prompt_suffix_length
    query_start = query_end - query_length
    return (query_start, query_end)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--max_heads', type=int, default=20)
parser.add_argument('--train_samples', type=int, default=200)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(args.seed)
    device = "cuda:0"    
    tokenizer, model = load_model_tokenizer(model_name=args.model, device=device, dtype=torch.float16)

    train_queries, test_queries, tools = get_queries_and_items()
    print("\n[Phase 1] Selecting retrieval heads...")

    selected_heads = select_retrieval_heads(
        train_queries=train_queries[:args.train_samples],
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        device=device,
        max_heads=args.max_heads
    )

    print(f"Selected {len(selected_heads)} heads")
    print(selected_heads)
    
    # Pre-process selected heads for O(1) loop lookup
    from collections import defaultdict
    _selected_heads_by_layer = defaultdict(list)
    for layer_id, head_id in selected_heads:
        _selected_heads_by_layer[layer_id].append(head_id)

    # Attach layer_idx manually and hook
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn._layer_idx = idx
        hooks.append(layer.self_attn.register_forward_hook(_run3_attn_hook, with_kwargs=True))

    print("\n[Phase 2] Evaluating on test set...")
    correct_at_1, correct_at_5, total = 0, 0, 0
    t0 = time.time()

    for qix in tqdm(range(len(test_queries))):
        sample = test_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)
        putils = PromptUtils(tokenizer=tokenizer, doc_ids=shuffled_keys, dict_all_docs=tools)

        _item_spans = putils.doc_spans
        gold_tool_id = putils.dict_doc_name_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        total_length = inputs.input_ids.shape[1]
        _q_start, _q_end = get_query_span(putils, question, total_length)
        
        _doc_scores = torch.zeros(len(_item_spans), device=device)

        with torch.no_grad():
            _ = model(**inputs, output_attentions=True)

        ranked_docs = torch.argsort(_doc_scores, descending=True)
        gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()

        if gold_rank == 0: correct_at_1 += 1
        if gold_rank < 5:  correct_at_5 += 1
        total += 1
        
        torch.cuda.empty_cache()
        
    for h in hooks: h.remove()

    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    print(f"\nRecall@1 (selected heads): {recall_at_1:.4f}")
    print(f"Recall@5 (selected heads): {recall_at_5:.4f}")
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/part3_heads_{args.max_heads}.json", "w") as f:
        json.dump({
            "selected_heads": selected_heads,
            "recall_1": recall_at_1,
            "recall_5": recall_at_5,
            "max_heads": args.max_heads
        }, f, indent=2)
    print(f"Results saved. Total time: {(time.time() - t0)/60:.1f} minutes")
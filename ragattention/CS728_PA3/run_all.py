import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import time
import random
import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items
from code3 import select_retrieval_heads
from collections import defaultdict

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
_selected_heads_by_layer = defaultdict(list)
_doc_scores_dict = {}
_avg_query_attn = None

def _run_all_attn_hook(module, args, kwargs, output):
    global _avg_query_attn, _q_start, _q_end, _item_spans, _selected_heads_by_layer, _doc_scores_dict
    attn_weights = output[1]
    if attn_weights is not None:
        layer_idx = module._layer_idx
        
        # --- Part 2 Logic ---
        layer_avg = attn_weights.squeeze(0).mean(dim=0)
        query_row = layer_avg[_q_start:_q_end, :].clone().cpu()
        if _avg_query_attn is None:
            _avg_query_attn = query_row
        else:
            _avg_query_attn += query_row

        # --- Part 3 Logic ---
        if layer_idx in _selected_heads_by_layer:
            for head_idx in _selected_heads_by_layer[layer_idx]:
                attn_head = attn_weights[0, head_idx]
                query_attn = attn_head[_q_start:_q_end, :]
                
                # We need to map (layer, head) to which group it belongs to in the dict.
                # Instead of looping through all spans here and sorting it out later,
                # we just add the sums to the respective configurations that include this head.
                
                # Precompute sums once per head
                sums = torch.zeros(len(_item_spans), device=attn_weights.device)
                for i, (d_start, d_end) in enumerate(_item_spans):
                    sums[i] = query_attn[:, d_start:d_end].sum()
                
                # Add it to the necessary configurations tracking running totals
                for group_name, group_heads in _doc_scores_dict.items():
                    if (layer_idx, head_idx) in group_heads['list']:
                        group_heads['scores'] += sums
        
        # Free memory!
        attn_weights.untyped_storage().resize_(0)
    return output

def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    positions = [r['gold_position'] for r in result]
    scores = [r['gold_score'] for r in result]
    
    df = pd.DataFrame({'position': positions, 'score': scores})
    grouped = df.groupby('position')['score'].mean().reset_index()
    grouped = grouped.sort_values('position')
    
    plt.figure(figsize=(14, 6))
    plt.plot(grouped['position'], grouped['score'], marker='o', markersize=3, linewidth=1, alpha=0.8)
    plt.xlabel('Position of Gold Tool in Prompt', fontsize=12)
    plt.ylabel('Average Attention Score', fontsize=12)
    plt.title('Attention to Gold Tool vs. Position in Prompt', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--train_samples', type=int, default=200)
parser.add_argument('--test_samples', type=int, default=5000)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(args.seed)
    device = "cuda:0"    
    tokenizer, model = load_model_tokenizer(model_name=args.model, device=device, dtype=torch.float16)

    train_queries, test_queries, tools = get_queries_and_items()
    print("\n[Phase 1] Selecting retrieval heads (Bonus Extended)...")

    # Request up to 30 heads. Returns dict with multiple strategies
    heads_dict = select_retrieval_heads(
        train_queries=train_queries[:args.train_samples],
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        device=device,
        max_heads=30
    )

    mrr_heads = heads_dict["mrr"]
    mass_heads = heads_dict["attn_mass"]
    
    print(f"Bonus: Extracted MRR Top-30 and Attn-Mass Top-30 heads.")
    
    # Register all unique heads we need to track
    all_unique_heads = set(mrr_heads) | set(mass_heads[:20])
    for layer_id, head_id in all_unique_heads:
        _selected_heads_by_layer[layer_id].append(head_id)

    hooks = []
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn._layer_idx = idx
        hooks.append(layer.self_attn.register_forward_hook(_run_all_attn_hook, with_kwargs=True))

    print(f"\n[Phase 2] Evaluating BOTH Part 2 and all Part 3 Configurations simultaneously on {args.test_samples} cases...")
    
    p2_results = []
    
    # Initialize trackers for the configurations
    group_trackers = {
        "mrr_10": {"list": set(mrr_heads[:10]), "c1": 0, "c5": 0, "scores": None},
        "mrr_20": {"list": set(mrr_heads[:20]), "c1": 0, "c5": 0, "scores": None},
        "mrr_30": {"list": set(mrr_heads[:30]), "c1": 0, "c5": 0, "scores": None},
        "attn_mass_20": {"list": set(mass_heads[:20]), "c1": 0, "c5": 0, "scores": None}
    }

    total = 0
    t0 = time.time()

    for qix in tqdm(range(min(len(test_queries), args.test_samples))):
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

        query_prompt = f"Query: {question}" + "\nCorrect tool_id:"
        query_length = len(putils.tokenizer(query_prompt, add_special_tokens=False).input_ids)
        total_length = inputs.input_ids.shape[1]
        
        _q_end = total_length - putils.prompt_suffix_length
        _q_start = _q_end - query_length
        
        # Reset globals appropriately
        _avg_query_attn = None
        _doc_scores_dict = group_trackers
        for k in _doc_scores_dict:
            _doc_scores_dict[k]["scores"] = torch.zeros(len(_item_spans), device=device)

        with torch.no_grad():
            _ = model(**inputs, output_attentions=True)
            
        # Part 2 Metrics
        avg_attn = _avg_query_attn / 16.0
        doc_scores_part2 = torch.zeros(len(_item_spans))
        for i, (d_start, d_end) in enumerate(_item_spans):
            doc_scores_part2[i] = avg_attn[:, d_start:d_end].sum()
            
        p2_ranked_docs = torch.argsort(doc_scores_part2, descending=True)
        p2_gold_rank = (p2_ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()
        p2_gold_score = doc_scores_part2[gold_tool_id].item()
        
        p2_results.append({
            "qid": sample["qid"],
            "gold_position": gold_tool_id,
            "gold_score": p2_gold_score,
            "gold_rank": p2_gold_rank
        })

        # Part 3 Metrics
        for k, v in group_trackers.items():
            p3_ranked_docs = torch.argsort(v["scores"], descending=True)
            p3_gold_rank = (p3_ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()

            if p3_gold_rank == 0: v["c1"] += 1
            if p3_gold_rank < 5:  v["c5"] += 1
        
        total += 1
        torch.cuda.empty_cache()
        
    for h in hooks: h.remove()

    print("\n--- Part 2 Results ---")
    p2_recall_1 = sum(1 for r in p2_results if r['gold_rank'] == 0) / len(p2_results)
    p2_recall_5 = sum(1 for r in p2_results if r['gold_rank'] < 5) / len(p2_results)
    print(f"Part 2 Full Attn | Recall@1: {p2_recall_1:.4f} | Recall@5: {p2_recall_5:.4f}")
    analyze_gold_attention(p2_results)

    print("\n--- Part 3 Results (Bonuses Included) ---")
    p3_export_metrics = {}
    for k, v in group_trackers.items():
        r1 = v["c1"] / total
        r5 = v["c5"] / total
        print(f"Group: {k:>12} | Recall@1: {r1:.4f}  |  Recall@5: {r5:.4f}")
        p3_export_metrics[k] = {"Recall@1": r1, "Recall@5": r5, "heads": list(v["list"])}
    
    os.makedirs("results", exist_ok=True)
    with open("results/part2_results.json", "w") as f:
        json.dump({"recall_1": p2_recall_1, "recall_5": p2_recall_5}, f, indent=2)
    with open("results/part3_bonus_results.json", "w") as f:
        json.dump(p3_export_metrics, f, indent=2)
        
    print(f"\nAll results saved. Total time: {(time.time() - t0)/60:.1f} minutes")

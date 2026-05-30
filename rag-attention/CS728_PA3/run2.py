import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Global variables for hook
_avg_query_attn = None
_q_start = 0
_q_end = 0

def attn_hook(module, args, kwargs, output):
    global _avg_query_attn, _q_start, _q_end
    attn_weights = output[1]
    if attn_weights is not None:
        layer_avg = attn_weights.squeeze(0).mean(dim=0)
        query_row = layer_avg[_q_start:_q_end, :].clone().cpu()
        if _avg_query_attn is None:
            _avg_query_attn = query_row
        else:
            _avg_query_attn += query_row
        # Clear tensor memory to avoid OOM
        attn_weights.untyped_storage().resize_(0)
    return output

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    Compute document scores.
    attentions: dummy variable here as we extract using hooks to avoid OOM.
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    global _avg_query_attn
    q_start, q_end = query_span
    
    avg_attn = _avg_query_attn / 16.0  # num_layers = 16
    
    doc_scores = torch.zeros(len(doc_spans))
    for i, (d_start, d_end) in enumerate(doc_spans):
        doc_scores[i] = avg_attn[:, d_start:d_end].sum()
    
    return doc_scores

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

def get_query_span(putils, query, total_length):
    query_prompt = f"Query: {query}" + "\nCorrect tool_id:"
    query_length = len(putils.tokenizer(query_prompt, add_special_tokens=False).input_ids)
    query_end = total_length - putils.prompt_suffix_length
    query_start = query_end - query_length
    return (query_start, query_end)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

if __name__ == '__main__':
    seed_all(seed=args.seed)
    device = "cuda:0"
    tokenizer, model = load_model_tokenizer(model_name=args.model, device=device, dtype=torch.float16)
    train_queries, test_queries, tools = get_queries_and_items()

    # Register hooks
    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(attn_hook, with_kwargs=True))

    results = []
    start_time = time.time()
    
    for qix in tqdm(range(len(test_queries))):
        sample = test_queries[qix]
        qid = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # Do Not change the shuffling here
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(tokenizer=tokenizer, doc_ids=shuffled_keys, dict_all_docs=tools)
        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id
        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        total_length = inputs.input_ids.shape[1]
        query_span = get_query_span(putils, question, total_length)
        
        _q_start, _q_end = query_span
        _avg_query_attn = None

        with torch.no_grad():
            _ = model(**inputs, output_attentions=True)
            
        doc_scores = query_to_docs_attention(None, query_span, item_spans)
        
        ranked_indices = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_indices == gold_tool_id).nonzero(as_tuple=True)[0].item()
        gold_score = doc_scores[gold_tool_id].item()
        
        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    recall_1 = sum(1 for r in results if r['gold_rank'] == 0) / len(results)
    recall_5 = sum(1 for r in results if r['gold_rank'] < 5) / len(results)
    print(f"\nRecall@1: {recall_1:.4f}")
    print(f"Recall@5: {recall_5:.4f}")

    analyze_gold_attention(results)

    os.makedirs("results", exist_ok=True)
    with open("results/part2_results.json", "w") as f:
        json.dump({"recall_1": recall_1, "recall_5": recall_5}, f, indent=2)
    print(f"Results saved. Total time: {(time.time() - start_time)/60:.1f} minutes")
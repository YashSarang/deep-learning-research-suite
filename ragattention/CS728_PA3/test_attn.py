"""Quick timing test for run2 with a full prompt"""
import os, torch, time, random, numpy as np, json
os.environ['TRANSFORMERS_OFFLINE'] = '1'

random.seed(64); np.random.seed(64); torch.manual_seed(64)
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

device = 'cuda:0'
tokenizer, model = load_model_tokenizer('meta-llama/Llama-3.2-1B-Instruct', device, torch.float16)
train_queries, test_queries, tools = get_queries_and_items()

# Test with 3 queries
times = []
for i in range(3):
    sample = test_queries[i]
    shuffled_keys = list(tools.keys()); random.shuffle(shuffled_keys)
    putils = PromptUtils(tokenizer=tokenizer, doc_ids=shuffled_keys, dict_all_docs=tools)
    prompt = putils.create_prompt(query=sample['text'])
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
    t1 = time.time()
    
    n_tok = inputs.input_ids.shape[1]
    has_attn = out.attentions is not None
    times.append(t1 - t0)
    print(f'Query {i}: {n_tok} tokens, {t1-t0:.2f}s, attentions={has_attn}', flush=True)
    
    del out
    torch.cuda.empty_cache()

avg_t = sum(times) / len(times)
print(f'\nAvg time per query: {avg_t:.2f}s', flush=True)
print(f'Estimated total for 5000 queries: {avg_t * 5000 / 3600:.1f} hours', flush=True)
print(f'GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB', flush=True)

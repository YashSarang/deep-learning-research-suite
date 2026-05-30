import os, torch, time, random, numpy as np
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

device = 'cuda:0'
tokenizer, model = load_model_tokenizer('meta-llama/Llama-3.2-1B-Instruct', device, torch.float16)

train_queries, test_queries, tools = get_queries_and_items()
shuffled_keys = list(tools.keys())
putils = PromptUtils(tokenizer=tokenizer, doc_ids=shuffled_keys, dict_all_docs=tools)

# Accumulator
avg_query_attn = None
q_start = 0
q_end = 0

def get_attn_hook(module, args, kwargs, output):
    global avg_query_attn, q_start, q_end
    attn_weights = output[1]
    if attn_weights is not None:
        layer_avg = attn_weights.squeeze(0).mean(dim=0)  # (N, N)
        query_row = layer_avg[q_start:q_end, :].clone().cpu()
        if avg_query_attn is None:
            avg_query_attn = query_row
        else:
            avg_query_attn += query_row
            
        # Free memory of the large attention matrix so it doesn't accumulate
        attn_weights.untyped_storage().resize_(0)
    return output

hooks = [layer.self_attn.register_forward_hook(get_attn_hook, with_kwargs=True) for layer in model.model.layers]

times = []
for i in range(3):
    sample = test_queries[i]
    prompt = putils.create_prompt(query=sample['text'])
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    
    q_start = inputs.input_ids.shape[1] - 100
    q_end = inputs.input_ids.shape[1] - 10
    avg_query_attn = None
    
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    t1 = time.time()
    times.append(t1-t0)
    print(f"Query {i+1} Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB, Time: {t1-t0:.2f}s")
    
for h in hooks: h.remove()
print(f"Average time per query: {np.mean(times):.3f}s")
print(f"Projected for 5000: {np.mean(times)*5000/60:.2f} minutes")

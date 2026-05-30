import os, torch, time, random, numpy as np
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

device = 'cuda:0'
tokenizer, model = load_model_tokenizer('meta-llama/Llama-3.2-1B-Instruct', device, torch.float16)

train_queries, test_queries, tools = get_queries_and_items()
sample = test_queries[0]
shuffled_keys = list(tools.keys()); random.shuffle(shuffled_keys)
putils = PromptUtils(tokenizer=tokenizer, doc_ids=shuffled_keys, dict_all_docs=tools)
prompt = putils.create_prompt(query=sample['text'])
inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)

q_start = inputs.input_ids.shape[1] - 100
q_end = inputs.input_ids.shape[1] - 10

# Accumulator
avg_query_attn = None
num_layers = model.config.num_hidden_layers

# Hook function
def get_attn_hook(module, args, kwargs, output):
    # output of LlamaAttention is (attn_output, attn_weights, past_key_value)
    # where attn_weights is (batch_size, num_heads, seq_len, seq_len)
    global avg_query_attn
    attn_weights = output[1]
    if attn_weights is not None:
        # Layer avg over heads:
        layer_avg = attn_weights.squeeze(0).mean(dim=0)  # (N, N)
        query_row = layer_avg[q_start:q_end, :].cpu()
        
        if avg_query_attn is None:
            avg_query_attn = query_row
        else:
            avg_query_attn += query_row
    return output

# Register hooks
hooks = []
for layer in model.model.layers:
    h = layer.self_attn.register_forward_hook(get_attn_hook, with_kwargs=True)
    hooks.append(h)

t0 = time.time()
with torch.no_grad():
    # Pass output_attentions=True so the attention module computes and returns weights
    _ = model(**inputs, output_attentions=True)
t1 = time.time()

for h in hooks:
    h.remove()

if avg_query_attn is not None:
    avg_query_attn /= num_layers
    print(f"Hook success! Query row shape: {avg_query_attn.shape}")
else:
    print("Hook failed to get attn_weights!")

print(f"Time: {t1-t0:.2f}s")
print(f"GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

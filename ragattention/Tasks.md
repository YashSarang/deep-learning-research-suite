# Implementation Tasks

## Preparation
- [x] Read through the entire Assignment 3 description.
- [x] Clone / Set up the repository from `https://github.com/deekshakoul/CS728_PA3`.
- [x] Ensure model and seed are kept as default.

## Part 1: Classical Retrieval
- [x] Implement independent query and tool encoding.
- [x] Compute similarity between the query and each tool.
- [x] Retrieve the top-k most relevant tools for each query.
- [x] Evaluate `Recall@1` and `Recall@5` for:
  - [x] BM25 (sparse retrieval baseline)
  - [x] `msmarco-MiniLM` (dense retrieval)
  - [x] `UAE-large-v1` (dense retrieval)

## Part 2: Lost-in-the-middle (Attention-based Retrieval)
- [x] Complete `get_query_span(...)` in `run2.py`: Identify the token span corresponding to the query in the prompt.
- [x] Complete `query_to_docs_attention(...)` in `run2.py`: Compute a score for each tool using attention from query tokens to tool tokens.
- [x] Rank tools using the computed attention scores.
- [x] Complete `analyze_gold_attention(...)` in `run2.py`: Aggregate results across queries and generate a plot.
- [x] Evaluate `Recall@1` and `Recall@5` for test queries using attention-based scores.
- [x] Generate and save the plot at `plot2/gold_attention_plot.png`.

## Part 3: Retrieval Heads
### Phase 1: Head Selection
- [x] Modify `select_retrieval_heads(...)` in `code3.py` using training data.
- [x] Analyze how each head distributes attention from the query to tools.
- [x] Aggregate information across multiple queries.
- [x] Return a list of K heads (e.g., K = 20), formatted as `(layer_id, head_id)`.

### Phase 2: Retrieval Using Selected Heads
- [x] Complete `query_to_docs_attention_heads(...)` in `run3.py`: Compute tool scores using only the selected heads.
- [x] Complete `get_query_span(...)` in `run3.py`: Identify the query tokens.
- [x] Rank tools for each query using the selected heads' attention scores.
- [x] Evaluate `Recall@1` and `Recall@5` for test queries.

## Report Generation
- [x] Create a report document containing all findings.
- [x] Add a table with `Recall@1` and `Recall@5` for the three baselines (Part 1).
- [x] Add a table with `Recall@1` and `Recall@5` for Attention-based Retrieval (Part 2).
- [x] Add the plot visualizing position effects on attention-based ranking (Part 2).
- [x] Add a brief description of the head selection strategy and list the chosen heads (Part 3 Phase 1).
- [x] Add a table with `Recall@1` and `Recall@5` for Selected Heads Retrieval (Part 3 Phase 2).
- [x] Add a comparison of recall performance for Parts 1, 2, and 3.
- [x] **[BONUS]** Test with `max_heads` = 10, 20, 30 and report the effect on test query recall.
- [x] **[BONUS]** Try different approaches for head selection, mention them briefly, and report their recall performance.

## Final Submission
- [ ] Ensure formatting guidelines are met (no Jupyter notebooks, don't change function names).
- [ ] Zip all code, asset files, and the report into a single file named `[Roll1_roll2].(zip/tar.gz)`.
- [ ] Submit via Moodle (or Google Drive link if it exceeds size limits).

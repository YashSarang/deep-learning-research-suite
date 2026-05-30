FROM ai-research-suite/nlp-base:latest

USER root

# Install RAG dependencies
RUN pip3 install \
    rank-bm25==0.2.2 \
    faiss-gpu==1.7.2 \
    langchain==0.1.0 \
    openai==1.6.1

WORKDIR /workspace
COPY rag-attention /workspace/

# Pre-download LLaMA model (optional, for faster startup)
# RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"

RUN if [ -f CS728_PA3/requirements.txt ]; then pip3 install -r CS728_PA3/requirements.txt; fi

RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

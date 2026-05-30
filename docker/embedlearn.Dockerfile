FROM ai-research-suite/nlp-base:latest

USER root

# Install additional NLP dependencies
RUN pip3 install \
    rank-bm25==0.2.2 \
    python-crfsuite==0.9.9 \
    conllu==4.5.3

WORKDIR /workspace
COPY embedlearn /workspace/

# Install project requirements
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

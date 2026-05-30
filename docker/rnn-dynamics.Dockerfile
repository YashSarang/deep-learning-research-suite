FROM ai-research-suite/nlp-base:latest

USER root

WORKDIR /workspace
COPY rnn-dynamics /workspace/

RUN if [ -f PA2_code/requirements.txt ]; then pip3 install -r PA2_code/requirements.txt; fi

RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

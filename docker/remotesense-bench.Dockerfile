FROM ai-research-suite/cv-base:latest

USER root

# Install experiment tracking
RUN pip3 install \
    hydra-core==1.3.2 \
    mlflow==2.9.2 \
    wandb==0.16.1

WORKDIR /workspace
COPY transfer-learning-benchmark /workspace/

RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

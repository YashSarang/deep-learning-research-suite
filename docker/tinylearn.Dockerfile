FROM ai-research-suite/cv-base:latest AS builder

USER root

# Install C++ build tools
RUN apt-get update && apt-get install -y \
    cmake \
    g++-11 \
    libopencv-dev \
    python3.10-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install pybind11 opencv-python

WORKDIR /workspace
COPY cpp-cnn-framework /workspace/

# Build C++ extension
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

FROM ai-research-suite/cv-base:latest

USER root
COPY --from=builder /workspace /workspace
RUN chown -R appuser:appuser /workspace

USER appuser
WORKDIR /workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

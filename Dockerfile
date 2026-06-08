# The Mind of Tashi — Docker Space.
# Docker (not the gradio SDK) so we can install a C/C++ toolchain and compile
# llama-cpp-python from source — the llama.cpp runtime that earns the
# Off-the-Grid + Llama-Champion badges. No cloud API at request time.
FROM python:3.12-slim

# Build toolchain for llama-cpp-python + libgomp for its OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as uid 1000; give it a writable home + HF cache.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    GENAI_OTEL_DISABLE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /home/user/app

# Install deps first for layer caching (llama-cpp-python compiles here).
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# App code (engine, llm, prompts, opponents, static frontend, assets, …).
COPY --chown=user . ./

EXPOSE 7860

# gradio.Server() + app.launch(); GRADIO_SERVER_* env binds it to 0.0.0.0:7860.
CMD ["python", "app.py"]

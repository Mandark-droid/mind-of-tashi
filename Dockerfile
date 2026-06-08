# The Mind of Tashi — Docker Space.
# Docker (not the gradio SDK) so we control the llama.cpp install: we pull the
# PREBUILT CPU wheel for llama-cpp-python rather than compiling from source
# (a source compile OOM-kills the free build runner — exit 137). No cloud API
# at request time — the Off-the-Grid + Llama-Champion contract.
FROM python:3.12-slim

# libgomp1 = the OpenMP runtime llama.cpp needs. No compiler toolchain: the
# wheel is prebuilt, so build-essential/cmake are unnecessary (smaller image,
# faster build, and no OOM).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
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

# The extra index serves prebuilt CPU wheels for llama-cpp-python;
# --only-binary on that package forces the wheel (fail fast if one is missing,
# never fall back to an OOM source compile). Everything else comes from PyPI.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        --only-binary=llama-cpp-python \
        -r requirements.txt

# App code (engine, llm, prompts, opponents, static frontend, assets, …).
COPY --chown=user . ./

EXPOSE 7860

# gradio.Server() + app.launch(); GRADIO_SERVER_* env binds it to 0.0.0.0:7860.
CMD ["python", "app.py"]

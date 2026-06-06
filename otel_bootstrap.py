"""
otel_bootstrap.py — one-call OpenTelemetry instrumentation for data-generation
scripts (selfplay, live self-play, live-traces capture).

We use `genai-otel-instrument` (https://pypi.org/project/genai-otel-instrument/)
which auto-traces popular LLM SDKs/clients (openai, ollama, google-genai,
anthropic, langchain, …) plus HTTP libraries (aiohttp/httpx/requests) that
the OpenAI-compatible teachers (OpenRouter / Mistral / Sarvam) ride on.

Telemetry is shipped via OTLP/HTTP to the collector running on our
traceverse VM. Override with OTEL_EXPORTER_OTLP_ENDPOINT if you point it
elsewhere.

Usage — call once, as early as possible (before importing openai/google-genai/
ollama clients):

    from otel_bootstrap import init_otel
    init_otel(service_name="mind-of-tashi-selfplay")

If the library or its deps are missing, init_otel() is a no-op so the
scripts keep working unmodified — instrumentation is opt-in by install.
"""

from __future__ import annotations
import os
from typing import Optional

# Module-level sentinel — instrument() is global and must run exactly once
# per process. selfplay/live can be imported via multiple entrypoints, so
# guard against double-init (would otherwise leak exporters).
_INITIALIZED = False

# Default collector — the traceverse VM running our OTLP collector.
# Port 4318 is OTLP/HTTP (4317 is gRPC); the library defaults to HTTP.
DEFAULT_OTLP_ENDPOINT = "http://192.168.206.129:4318"


def init_otel(service_name: str, *, force: bool = False) -> bool:
    """Initialize OTEL instrumentation. Idempotent.

    Returns True if instrumentation was activated this call, False if it
    was skipped (already initialized, disabled by env, or library missing).

    Reads from env (with sensible defaults):
      OTEL_EXPORTER_OTLP_ENDPOINT  default 192.168.206.129:4318 (traceverse VM)
      OTEL_SERVICE_NAME            falls back to the `service_name` arg
      GENAI_OTEL_DISABLE           set to "1" to skip init entirely
      GENAI_ENABLE_GPU_METRICS     default "true"  (GPU util, mem, power)
      GENAI_ENABLE_COST_TRACKING   default "true"  (per-call $ via model prices)
      GENAI_ENABLE_CO2_METRICS     default "true"  (energy -> kgCO2eq estimate)
      GENAI_ENABLE_EVAL            default "true"  (PII/toxicity/bias/etc.)
      GENAI_SAMPLING_RATE          default "1.0"   (1.0 = trace everything)
    """
    global _INITIALIZED
    if _INITIALIZED and not force:
        return False
    if os.environ.get("GENAI_OTEL_DISABLE", "0") == "1":
        return False

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)

    # Default-on knobs. setdefault preserves any explicit override the user
    # set in .env or the shell, so flipping a single metric off is one var.
    os.environ.setdefault("GENAI_ENABLE_GPU_METRICS", "true")
    os.environ.setdefault("GENAI_ENABLE_COST_TRACKING", "true")
    os.environ.setdefault("GENAI_ENABLE_CO2_METRICS", "true")
    os.environ.setdefault("GENAI_SAMPLING_RATE", "1.0")

    try:
        import genai_otel  # type: ignore
    except Exception as exc:
        print(f"[otel] genai-otel-instrument not installed ({exc}); "
              f"skipping instrumentation. `pip install genai-otel-instrument[gpu]`")
        return False

    enable_eval = os.environ.get("GENAI_ENABLE_EVAL", "true").lower() in ("1", "true", "yes")

    try:
        if enable_eval:
            genai_otel.instrument(
                enable_pii_detection=True,
                enable_toxicity_detection=True,
                enable_bias_detection=True,
                enable_prompt_injection_detection=True,
                enable_hallucination_detection=True,
                enable_restricted_topics=True,
            )
        else:
            genai_otel.instrument()
    except TypeError:
        # Older versions don't accept eval kwargs — fall back to plain init
        # so we don't hard-fail when the user has an older release pinned.
        genai_otel.instrument()
    except Exception as exc:
        print(f"[otel] genai_otel.instrument() failed: {exc}; continuing uninstrumented")
        return False

    # The openai + ollama Python SDKs both ride on httpx. genai-otel-instrument's
    # OpenAI/Ollama wrappers exist but in v1.3.1 don't reliably emit spans for
    # AsyncClient calls (observed: "enabled" log line, but no spans land).
    # Layer the httpx instrumentor on top so every SDK call still surfaces as
    # a span — generic POST shape, but with full URL + status + duration.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except ImportError:
        print("[otel] opentelemetry-instrumentation-httpx not installed; "
              "openai/ollama SDK calls may emit no spans. "
              "Install: pip install opentelemetry-instrumentation-httpx")
    except Exception as exc:
        print(f"[otel] httpx instrumentation failed: {exc}")

    _INITIALIZED = True
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    flags = []
    if os.environ.get("GENAI_ENABLE_GPU_METRICS", "").lower() in ("1", "true", "yes"):
        flags.append("gpu")
    if os.environ.get("GENAI_ENABLE_COST_TRACKING", "").lower() in ("1", "true", "yes"):
        flags.append("cost")
    if os.environ.get("GENAI_ENABLE_CO2_METRICS", "").lower() in ("1", "true", "yes"):
        flags.append("co2")
    if enable_eval:
        flags.append("eval")
    print(f"[otel] instrumented service={service_name} -> {endpoint} "
          f"[{','.join(flags) if flags else 'core-only'}]")
    return True

import os
import time
from inspect import signature

import mlflow
import mlflow.langchain
from flask import Flask, jsonify, request
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "open-chat-llmops")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
PORT = int(os.getenv("PORT", "8000"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def configure_langchain_autolog() -> None:
    kwargs = {}
    autolog_params = signature(mlflow.langchain.autolog).parameters

    if "log_models" in autolog_params:
        kwargs["log_models"] = False
    if "log_inputs_outputs" in autolog_params:
        kwargs["log_inputs_outputs"] = True
    if "log_traces" in autolog_params:
        kwargs["log_traces"] = True

    try:
        mlflow.langchain.autolog(**kwargs)
    except Exception:
        # Traces are still captured by explicit spans below.
        pass


configure_langchain_autolog()

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=OLLAMA_TEMPERATURE,
    top_p=OLLAMA_TOP_P,
)


@app.get("/health")
def health() -> tuple[dict, int]:
    return {"status": "ok"}, 200


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Campo 'message' e obrigatorio."}), 400

    start = time.perf_counter()

    with mlflow.start_run(run_name="chat_request") as run:
        mlflow.log_params(
            {
                "model": OLLAMA_MODEL,
                "temperature": OLLAMA_TEMPERATURE,
                "top_p": OLLAMA_TOP_P,
            }
        )
        mlflow.log_text(message, "prompt.txt")

        try:
            with mlflow.start_span(name="chat_completion", span_type="CHAT_MODEL") as span:
                span.set_inputs({"message": message})
                span.set_attributes(
                    {
                        "llm.model": OLLAMA_MODEL,
                        "llm.temperature": OLLAMA_TEMPERATURE,
                        "llm.top_p": OLLAMA_TOP_P,
                    }
                )
                response = llm.invoke(message)
                answer = response.content if hasattr(response, "content") else str(response)
                span.set_outputs({"response": answer})
                span.set_status("OK")
                trace_id = span.request_id

            latency_ms = (time.perf_counter() - start) * 1000

            mlflow.log_text(answer, "response.txt")
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("trace_id", trace_id)

            return (
                jsonify(
                    {
                        "response": answer,
                        "latency_ms": round(latency_ms, 2),
                        "run_id": run.info.run_id,
                        "trace_id": trace_id,
                    }
                ),
                200,
            )
        except Exception as exc:
            with mlflow.start_span(name="chat_completion", span_type="CHAT_MODEL") as span:
                span.set_inputs({"message": message})
                span.set_outputs({"error": str(exc)})
                span.set_status("ERROR")
            mlflow.set_tag("status", "error")
            mlflow.set_tag("error_type", type(exc).__name__)
            mlflow.set_tag("error", str(exc)[:2000])
            return jsonify({"error": f"Falha ao consultar o modelo: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)

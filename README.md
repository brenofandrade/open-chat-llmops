# open-chat-llmops
Local LLM chat & MLflow Observability.

A robust implementation of a local chatbot ecosystem designed with **LLMOps** principles. This project bridges the gap between running local open-source models and professional production monitoring.

By combining **Ollama** for seamless model serving and **MLflow** for deep observability, you can track every interaction, compare model performances, and optimize prompts in a fully private environment.

---

## 🛠️ Technology Stack

* **[Ollama](https://ollama.com/):** The engine for running high-performance open-source models (Llama 3, Mistral, Phi-3) locally. It handles model management and provides a clean API for inference.
* **[MLflow](https://mlflow.org/):** An open-source platform for the machine learning lifecycle. In this project, it is used for **LLM Tracking & Tracing**:
    * Logging inputs, outputs, and system prompts.
    * Monitoring latency and execution time.
    * Versioning different model configurations (Temperature, Top-P).
* **[Python](https://www.python.org/):** The core logic orchestrating the communication between the LLM and the tracking server.

---

## 🚀 Getting Started

### 1) Prerequisites

* Docker + Docker Compose
* Ollama running on the host machine (`http://localhost:11434`)
* A pulled model in Ollama, for example:

```bash
ollama pull llama3.2
```

### 2) Start the stack

```bash
docker compose up --build
```

Services:

* Flask + LangChain API: `http://localhost:8000`
* Streamlit Chat UI: `http://localhost:8501`
* MLflow UI: `http://localhost:5000`

### 3) Send a chat message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explique o que e LLMOps em uma frase."}'
```

Expected JSON response:

```json
{
  "response": "...",
  "latency_ms": 1234.56,
  "run_id": "..."
}
```

### 4) Test via Streamlit UI

Open `http://localhost:8501`, type a message, and chat through the backend.
The UI also includes a `/health` check button.

### 5) View observability in MLflow

Open `http://localhost:5000`, then check experiment `open-chat-llmops`.
Each request logs:

* Params: model/temperature/top_p
* Metric: latency
* Artifacts: `prompt.txt` and `response.txt`

### 6) Run tests

The test in `tests/test_health_check_server.py` is an integration test and expects the backend running on `http://127.0.0.1:8000`.

```bash
docker compose up -d --build
docker compose exec app pip install -r requirements-dev.txt
docker compose exec app python -m pytest -q tests/test_health_check_server.py
docker compose down
```




---

📜 License
Distributed under the MIT License. See LICENSE for more information.


--- Built by Breno F. Andrade

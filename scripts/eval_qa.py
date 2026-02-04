import os
import json
import requests
import pandas as pd
import mlflow

from dotenv import load_dotenv
load_dotenv(override=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "open-chat-llmops")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATASET_PATH = os.getenv("DATASET_PATH", "data/eval_qa.csv")

def normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())

def exact_match(pred: str, expected: str) -> int:
    return int(normalize(pred) == normalize(expected))

def contains_expected(pred: str, expected: str) -> int:
    return int(normalize(expected) in normalize(pred))

def main():
    df = pd.read_csv(DATASET_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    rows = []
    with mlflow.start_run(run_name="eval_qa"):
        for _, r in df.iterrows():
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={"message": r["question"]},
                timeout=120,
            )
            payload = resp.json()
            pred = payload.get("response", "")

            row = {
                "id": r["id"],
                "question": r["question"],
                "expected_answer": r["expected_answer"],
                "predicted_answer": pred,
                "run_id_chat": payload.get("run_id"),
                "trace_id_chat": payload.get("trace_id"),
                "exact_match": exact_match(pred, r["expected_answer"]),
                "contains_expected": contains_expected(pred, r["expected_answer"]),
            }
            rows.append(row)

        out = pd.DataFrame(rows)
        mlflow.log_metric("exact_match_rate", float(out["exact_match"].mean()))
        mlflow.log_metric("contains_expected_rate", float(out["contains_expected"].mean()))
        mlflow.log_table(out, "eval/results.json")
        mlflow.log_artifact(DATASET_PATH, "eval")

        summary = {
            "n_samples": len(out),
            "exact_match_rate": float(out["exact_match"].mean()),
            "contains_expected_rate": float(out["contains_expected"].mean()),
        }
        mlflow.log_text(json.dumps(summary, ensure_ascii=False, indent=2), "eval/summary.json")

if __name__ == "__main__":
    main()
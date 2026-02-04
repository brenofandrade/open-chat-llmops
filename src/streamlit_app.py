import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://app:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

st.set_page_config(page_title="Open Chat LLMOps", page_icon="💬", layout="centered")
st.title("Open Chat LLMOps")
st.caption("Interface Streamlit para testar o backend Flask/LangChain.")

with st.sidebar:
    st.subheader("Backend")
    st.code(BACKEND_URL)
    if st.button("Verificar /health", use_container_width=True):
        try:
            health_response = requests.get(HEALTH_ENDPOINT, timeout=5)
            health_response.raise_for_status()
            st.success(f"Backend online: {health_response.json()}")
        except requests.RequestException as exc:
            st.error(f"Falha ao conectar no backend: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

prompt = st.chat_input("Digite sua mensagem")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo..."):
            try:
                response = requests.post(
                    CHAT_ENDPOINT,
                    json={"message": prompt},
                    timeout=120,
                )
                payload = response.json()

                if response.status_code >= 400:
                    error_msg = payload.get("error", "Erro desconhecido.")
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Erro: {error_msg}"}
                    )
                else:
                    answer = payload.get("response", "")
                    latency_ms = payload.get("latency_ms", "-")
                    run_id = payload.get("run_id", "-")
                    st.markdown(answer)
                    st.caption(f"latency_ms: {latency_ms} | run_id: {run_id}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "meta": f"latency_ms: {latency_ms} | run_id: {run_id}",
                        }
                    )
            except requests.RequestException as exc:
                error_msg = f"Falha na chamada ao backend: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Erro: {error_msg}"}
                )

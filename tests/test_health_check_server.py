import requests


def test_health_check():
    response = requests.get("http://127.0.0.1:8000/health", timeout=5)

    assert response.status_code == 200, "Teste de backend falhou. Verifique o servidor e tente novamente!"
    assert response.json() == {"status": "ok"}

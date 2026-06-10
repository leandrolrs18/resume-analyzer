import io

import pytest


@pytest.mark.asyncio
async def test_healthcheck(client) -> None:
    # Confirma que a rota de saúde responde com status OK.
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_analyze_without_query_returns_summaries(client) -> None:
    # Confirma que a análise sem query retorna resumos por candidato.
    files = {"files": ("maria.pdf", io.BytesIO(b"fake"), "application/pdf")}
    data = {"request_id": "abc", "user_id": "user-1"}

    response = await client.post("/analyze", files=files, data=data)

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "abc"
    assert body["results"][0]["candidate"] == "Maria"


@pytest.mark.asyncio
async def test_analyze_with_query_returns_ranking(client) -> None:
    # Confirma que a análise com query retorna ranking com score e justificativa.
    files = {"files": ("maria.pdf", io.BytesIO(b"fake"), "application/pdf")}
    data = {
        "request_id": "rank-1",
        "user_id": "user-1",
        "query": "Python AWS",
        "language": "pt",
        "llm_provider": "local",
    }

    response = await client.post("/analyze", files=files, data=data)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Python AWS"
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["score"] == 0.92


@pytest.mark.asyncio
async def test_logs_endpoint_returns_payload(client) -> None:
    # Confirma que a auditoria por request_id responde com o payload esperado.
    response = await client.get("/logs/abc")

    assert response.status_code == 200
    assert response.json() == {"request_id": "abc", "logs": []}

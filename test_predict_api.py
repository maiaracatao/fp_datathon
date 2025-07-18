import pytest
from fastapi.testclient import TestClient
from predict import app

client = TestClient(app)

# Payload de exemplo válido
valid_payload = {
    "tipo_contratacao": "CLT",
    "nivel_profissional_vaga": "Pleno",
    "nivel_academico_vaga": "Superior completo",
    "nivel_ingles_vaga": "Intermediário",
    "nivel_academico_candidato": "Superior completo",
    "nivel_ingles_candidato": "Intermediário",
    "area_atuacao_candidato": "Administração",
    "area_atuacao_vaga": "Administração",
    "texto_cv": "Profissional com experiência em rotinas administrativas, atendimento ao cliente e controle de planilhas financeiras."
}

def test_predict_status_code():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

def test_predict_response_keys():
    response = client.post("/predict", json=valid_payload)
    data = response.json()
    assert "probabilidade" in data
    assert "contratado" in data
    assert isinstance(data["probabilidade"], float)
    assert data["contratado"] in [0, 1]

def test_predict_invalid_missing_field():
    # Removendo um campo obrigatório
    invalid_payload = valid_payload.copy()
    invalid_payload.pop("texto_cv")

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity (erro de validação)

def test_predict_invalid_type():
    payload = valid_payload.copy()
    payload["probabilidade"] = 0.5  # campo inválido propositalmente

    response = client.post("/predict", json=payload)
    assert response.status_code == 200  # Ignorado se campo extra, ou 422 se validado

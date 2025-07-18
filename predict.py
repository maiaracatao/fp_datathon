from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re
import string
import os
import shutil
import subprocess
import warnings
import sys
import logging

# ─── Configuração de logging ──────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/predict.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

from pipeline.preprocessador import preprocessar_dados

# ─── Carrega os modelos salvos ────────────────────────────────────────────────
modelo = joblib.load("modelo.pkl")
encoder = joblib.load("encoder.pkl")
tfidf = joblib.load("tfidf.pkl")

# ─── Inicializa a aplicação FastAPI ───────────────────────────────────────────
app = FastAPI(title="API de Previsão de Contratação")

# ─── Stopwords e limpeza de texto ─────────────────────────────────────────────
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
stopwords_pt = set(stopwords.words("portuguese"))

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(rf"[{string.punctuation}]", '', texto)
    texto = " ".join(p for p in texto.split() if p not in stopwords_pt)
    return texto.strip()

# ─── Schema de entrada com exemplo visível no Swagger UI ──────────────────────
class CandidatoInput(BaseModel):
    tipo_contratacao: str
    nivel_profissional_vaga: str
    nivel_academico_vaga: str
    nivel_ingles_vaga: str
    nivel_academico_candidato: str
    nivel_ingles_candidato: str
    area_atuacao_candidato: str
    area_atuacao_vaga: str
    texto_cv: str

    class Config:
        schema_extra = {
            "example": {
                "tipo_contratacao": "CLT",
                "nivel_profissional_vaga": "Pleno",
                "nivel_academico_vaga": "Superior completo",
                "nivel_ingles_vaga": "Intermediário",
                "nivel_academico_candidato": "Superior completo",
                "nivel_ingles_candidato": "Intermediário",
                "area_atuacao_candidato": "Administração",
                "area_atuacao_vaga": "Administração",
                "texto_cv": "Profissional com experiência em rotinas administrativas, atendimento ao cliente e controle de planilhas financeiras. Conhecimento em pacote Office e inglês intermediário."
            }
        }

class ResultadoOutput(BaseModel):
    probabilidade: float
    contratado: int

# ─── Rota principal: previsão de contratação ──────────────────────────────────
@app.post("/predict", response_model=ResultadoOutput)
def prever_contratacao(entrada: CandidatoInput):
    log.info("Recebida solicitação de predição")
    log.info(f"Área do candidato: {entrada.area_atuacao_candidato}, Nível de inglês: {entrada.nivel_ingles_candidato}")

    try:
        dados = pd.DataFrame([entrada.model_dump()])
        dados = dados.apply(lambda x: x.str.lower().str.strip())

        log.info("Aplicando codificação categórica")
        X_cat = encoder.transform(dados[
            [
                "tipo_contratacao",
                "nivel_profissional_vaga",
                "nivel_academico_vaga",
                "nivel_ingles_vaga",
                "nivel_academico_candidato",
                "nivel_ingles_candidato",
                "area_atuacao_candidato",
                "area_atuacao_vaga"
            ]
        ])

        texto_cv = limpar_texto(entrada.texto_cv)
        log.info(f"Texto do CV processado com {len(texto_cv)} caracteres")

        X_cv = tfidf.transform([texto_cv]).toarray()
        X_final = np.hstack([X_cat, X_cv])

        prob = modelo.predict_proba(X_final)[0][1]
        pred = int(prob >= 0.5)

        log.info(f"Predição concluída - Probabilidade: {prob:.4f}, Contratado: {pred}")

        return ResultadoOutput(probabilidade=round(prob, 4), contratado=pred)

    except Exception as e:
        log.exception(f"Erro durante a predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno na predição")


# ─── Rota para upload de dados e re-treinamento automático ────────────────────
@app.post("/atualizar_e_treinar")
async def atualizar_e_treinar(
    vagas: UploadFile = File(...),
    applicants: UploadFile = File(...),
    prospects: UploadFile = File(...)
):
    log.info("Recebido pedido de atualização de dados e re-treinamento do modelo")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    arquivos = {
        "vagas.json": vagas,
        "applicants.json": applicants,
        "prospects.json": prospects
    }

    for nome_arquivo, file in arquivos.items():
        caminho = os.path.join(data_dir, nome_arquivo)
        with open(caminho, "wb") as f:
            shutil.copyfileobj(file.file, f)
        log.info(f"Arquivo salvo: {nome_arquivo}")

    python_exec = sys.executable

    try:
        log.info("Executando pipeline de reprocessamento e treino")
        subprocess.run([python_exec, "pipeline/data_loader.py"], check=True)
        subprocess.run([python_exec, "pipeline/preprocessador.py"], check=True)
        result = subprocess.run(
            [python_exec, "pipeline/train_model.py"],
            check=True, capture_output=True, text=True
        )
        log.info("Pipeline executada com sucesso")

    except subprocess.CalledProcessError as e:
        log.exception("Erro durante execução da pipeline")
        return {"status": "erro", "detalhes": str(e)}

    saida = result.stdout
    linha_final = [l for l in saida.splitlines() if "Melhor modelo" in l]
    melhor_modelo = linha_final[0] if linha_final else "Desconhecido"
    log.info(f"Modelo vencedor: {melhor_modelo}")

    return {
        "status": "ok",
        "mensagem": "Pipeline executada com sucesso!",
        "modelo_vencedor": melhor_modelo
    }

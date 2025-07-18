import pandas as pd
import re
import string
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import nltk
import joblib

# Baixa stopwords se necessário
nltk.download("stopwords")
from nltk.corpus import stopwords

stopwords_pt = set(stopwords.words("portuguese"))

# Função para limpar texto
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'\d+', '', texto)  # remove números
    texto = re.sub(rf"[{string.punctuation}]", '', texto)
    texto = " ".join(p for p in texto.split() if p not in stopwords_pt)
    return texto.strip()


def preprocessar_dados(csv_path="dados_unificados.csv"):
    # Carrega o dataset
    df = pd.read_csv(csv_path)

    # Preenche apenas colunas de texto (object) com 'desconhecido'
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna("desconhecido")

    # Limpa o currículo
    df["texto_cv_limpo"] = df["texto_cv"].apply(limpar_texto)

    # Define colunas categóricas
    colunas_cat = [
        "tipo_contratacao",
        "nivel_profissional_vaga",
        "nivel_academico_vaga",
        "nivel_ingles_vaga",
        "nivel_academico_candidato",
        "nivel_ingles_candidato",
        "area_atuacao_candidato",
        "area_atuacao_vaga"
    ]

    # Normaliza texto
    df_cat = df[colunas_cat].astype(str).apply(lambda x: x.str.lower().str.strip())

    # Codifica categorias com OneHotEncoder (compatível com sklearn 1.7.0)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df_cat)

    # Vetoriza o texto do currículo
    tfidf = TfidfVectorizer(max_features=500)
    X_cv = tfidf.fit_transform(df["texto_cv_limpo"]).toarray()

    # Une as features numéricas
    X = np.hstack([X_cat, X_cv])
    y = df["contratado"]

    # Divide em treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Salva pré-processadores
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(tfidf, "tfidf.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocessar_dados()
    print(" Pré-processamento concluído.")
    print(" Shape X_train:", X_train.shape)
    print(" Distribuição y_train:", np.bincount(y_train))

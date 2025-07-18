import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Título do painel
st.title("Painel de Monitoramento do Modelo")

# Caminho do log
log_path = "logs/predict.log"
if not os.path.exists(log_path):
    st.warning("Arquivo de log ainda não foi gerado.")
    st.stop()

# Função para extrair dados do log
def carregar_predicoes_do_log(path):
    dados = []
    with open(path, "r", encoding="utf-8") as f:
        linhas = f.readlines()

    atual = {}
    for linha in linhas:
        if "Recebida solicitação de predição" in linha:
            try:
                timestamp = linha.split(" [")[0]
                atual["data"] = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
            except:
                continue
        elif "Área do candidato:" in linha:
            partes = linha.split("Área do candidato:")[1].strip().split(", Nível de inglês:")
            atual["area_atuacao"] = partes[0].strip()
            atual["nivel_ingles"] = partes[1].strip() if len(partes) > 1 else "desconhecido"
        elif "Predição concluída" in linha:
            try:
                partes = linha.split("Probabilidade:")[1].strip().split(", Contratado:")
                atual["probabilidade"] = float(partes[0])
                atual["contratado"] = int(partes[1])
                dados.append(atual.copy())
            except:
                continue
    return pd.DataFrame(dados)

# Carregar dados do log
df = carregar_predicoes_do_log(log_path)

if df.empty:
    st.info("Nenhuma predição registrada nos logs ainda.")
    st.stop()

# Métricas gerais
st.subheader("Métricas Gerais")
col1, col2, col3 = st.columns(3)

col1.metric("Total de Predições", len(df))
col2.metric("Taxa de Contratação (%)", f"{df['contratado'].mean() * 100:.2f}")
col3.metric("Probabilidade Média", f"{df['probabilidade'].mean():.2f}")

# Gráfico de distribuição
st.subheader("Distribuição das Predições")
contagem = df["contratado"].value_counts().sort_index()

labels = []
colors = []
if 0 in contagem.index:
    labels.append("Não Contratado")
    colors.append("red")
if 1 in contagem.index:
    labels.append("Contratado")
    colors.append("green")

fig1, ax1 = plt.subplots()
ax1.bar(range(len(contagem)), contagem.values, color=colors)
ax1.set_title("Distribuição das Predições")
ax1.set_ylabel("Quantidade")
ax1.set_xticks(range(len(contagem)))
ax1.set_xticklabels([label for _, label in zip(contagem.index, labels)], rotation=0)
st.pyplot(fig1)

# Gráfico por área
st.subheader("Probabilidade Média por Área de Atuação")
area_prob = df.groupby("area_atuacao")["probabilidade"].mean().sort_values()
fig2, ax2 = plt.subplots()
area_prob.plot(kind="barh", ax=ax2, color="skyblue")
ax2.set_xlabel("Probabilidade Média de Contratação")
ax2.set_ylabel("Área de Atuação")
st.pyplot(fig2)

# Verificação de drift
st.subheader("Monitoramento de Drift (últimos 7 dias)")
df["semana"] = df["data"].dt.to_period("W").astype(str)
semanas = df["semana"].unique()
if len(semanas) >= 2:
    semana_atual = semanas[-1]
    semana_anterior = semanas[-2]

    media_atual = df[df["semana"] == semana_atual]["probabilidade"].mean()
    media_anterior = df[df["semana"] == semana_anterior]["probabilidade"].mean()
    diff = media_atual - media_anterior

    st.markdown(f"- Semana atual: `{semana_atual}`")
    st.markdown(f"- Semana anterior: `{semana_anterior}`")
    st.markdown(f"- Diferença de probabilidade: `{diff:.4f}`")

    if abs(diff) > 0.05:
        st.error("Alerta de possível drift detectado.")
    else:
        st.success("Sem indícios de drift.")
else:
    st.warning("Ainda não há dados suficientes para comparar semanas.")

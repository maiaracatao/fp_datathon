# Dockerfile para o painel Streamlit
FROM python:3.12-slim

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos
COPY dashboard.py .
COPY requirements.txt ./requirements.txt

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta usada pelo Streamlit
EXPOSE 8501

# Comando para iniciar o painel
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

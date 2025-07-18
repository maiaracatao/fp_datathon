
# FIAP Datathon - Previsão de Contratação com Machine Learning

Este projeto foi desenvolvido como solução para o desafio do Datathon da FIAP, com foco em engenharia de machine learning. O sistema realiza a previsão de contratação com base em dados de candidatos e vagas utilizando modelos de machine learning, com deploy em API via FastAPI e monitoramento com Streamlit.

---

## 🔧 Funcionalidades

- 🔍 Previsão de contratação com base em atributos do candidato e da vaga
- 🔁 Atualização automática dos dados e re-treinamento do modelo
- 📦 Empacotamento com Docker
- 📊 Dashboard para análise de desempenho e monitoramento de *drift*
- ✅ Testes unitários com Pytest
- 📈 Logs detalhados de predição e performance

---

## 📁 Estrutura do Projeto

```
.
├── pipeline/               # Scripts da pipeline
│   ├── data_loader.py      # Unificação dos dados
│   ├── preprocessador.py   # Pré-processamento de texto e dados categóricos
│   └── train_model.py      # Treinamento dos modelos e salvamento
├── dashboard.py            # Scripts de monitoramento
│   ├── dashboard.py        # Painel de monitoramento com Streamlit
│   ├── Dockerfile          # Empacotamento do app
│   ├── requirements.txt    # Requisitos
├── logs/                   # Logs de predição
├── data/                   # Arquivos de entrada (.json)
├── Dockerfile              # Empacotamento do app
├── docker-compose.yml      # Orquestração da API + Dashboard
├── requirements.txt        # Requisitos
├── test_predict_api.py     # Testes da API
├── predict.py              # API com FastAPI
└── README.md               # Documentação
```

---

## 🚀 Como Executar

### 1. Clonar o Repositório

```bash
git clone https://github.com/maiaracatao/fp_datathon.git
cd fp_datathon
```

### 2. Executar com Docker Compose

```bash
docker-compose up --build
```

Acesse:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## 🧪 Executar Testes

```bash
pytest test_predict_api.py
pytest test_preprocessador.py
```

---

## 📌 Endpoints da API

- `POST /predict` → Recebe os dados do candidato e retorna a previsão
- `POST /atualizar_e_treinar` → Atualiza arquivos `.json` e re-treina o modelo

---

## 📊 Monitoramento

O painel em Streamlit exibe:
- Total de predições e taxa de contratação
- Distribuição das predições
- Probabilidade média por área
- Alertas de *drift* baseados no log de predições

---

## 🐳 Requisitos

- Docker
- Docker Compose

---

## 🙌 Contribuição

Sinta-se à vontade para contribuir com melhorias, correções ou sugestões!

---

## 📄 Licença

Este projeto é apenas para fins educacionais e de avaliação.

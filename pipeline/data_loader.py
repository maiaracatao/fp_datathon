import json
import pandas as pd
import os

# Diretório base (code/)
base_path = os.path.abspath(os.path.dirname(__file__))

applicants_path = os.path.join(base_path, "../data", "applicants.json")
prospects_path = os.path.join(base_path, "../data", "prospects.json")
vagas_path     = os.path.join(base_path, "../data", "vagas.json")

# Carrega os arquivos JSON
with open(applicants_path, encoding='utf-8') as f:
    applicants = json.load(f)

with open(prospects_path, encoding='utf-8') as f:
    prospects = json.load(f)

with open(vagas_path, encoding='utf-8') as f:
    vagas = json.load(f)

# Lista para armazenar os registros combinados
rows = []

# Percorre as vagas e seus prospects
for vaga_id, vaga_data in prospects.items():
    for prospect in vaga_data.get('prospects', []):
        candidato_id = prospect.get('codigo')
        situacao = prospect.get('situacao_candidado', '')

        # Ignora se o candidato não está na base de applicants
        if candidato_id not in applicants:
            continue

        # Dados da vaga (se existir)
        vaga_info = vagas.get(vaga_id, {})
        info_basicas = vaga_info.get("informacoes_basicas", {})
        perfil_vaga = vaga_info.get("perfil_vaga", {})

        # Dados do candidato
        app = applicants[candidato_id]
        nome = app.get("infos_basicas", {}).get("nome", "")
        nivel_ingles_cand = app.get("formacao_e_idiomas", {}).get("nivel_ingles", "")
        nivel_academico_cand = app.get("formacao_e_idiomas", {}).get("nivel_academico", "")
        area_atuacao_cand = app.get("informacoes_profissionais", {}).get("area_atuacao", "")
        texto_cv = app.get("cv_pt", "")

        row = {
            # Dados da vaga
            "vaga_id": vaga_id,
            "titulo_vaga": info_basicas.get("titulo_vaga", ""),
            "tipo_contratacao": info_basicas.get("tipo_contratacao", ""),
            "nivel_profissional_vaga": perfil_vaga.get("nivel profissional", ""),
            "nivel_ingles_vaga": perfil_vaga.get("nivel_ingles", ""),
            "nivel_academico_vaga": perfil_vaga.get("nivel_academico", ""),
            "area_atuacao_vaga": perfil_vaga.get("areas_atuacao", ""),

            # Dados do candidato
            "candidato_id": candidato_id,
            "nome": nome,
            "situacao": situacao,
            "contratado": 1 if "contratado" in situacao.lower() else 0,
            "nivel_ingles_candidato": nivel_ingles_cand,
            "nivel_academico_candidato": nivel_academico_cand,
            "area_atuacao_candidato": area_atuacao_cand,
            "texto_cv": texto_cv
        }

        rows.append(row)

# Converte para DataFrame
df = pd.DataFrame(rows)

# Exibe as primeiras linhas
print(df.head())

output_csv_path = os.path.join(base_path, "..", "dados_unificados.csv")
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')


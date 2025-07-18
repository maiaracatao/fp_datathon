import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from preprocessador import preprocessar_dados

def avaliar_modelo(nome, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n Modelo: {nome}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    return {
        "nome": nome,
        "modelo": modelo,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

if __name__ == "__main__":
    # Carrega os dados
    X_train, X_test, y_train, y_test = preprocessar_dados()

    # Lista de modelos
    modelos = [
        ("RandomForest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]

    resultados = []

    # Treina e avalia todos
    for nome, modelo in modelos:
        modelo.fit(X_train, y_train)
        resultado = avaliar_modelo(nome, modelo, X_test, y_test)
        resultados.append(resultado)

    # Encontra o melhor modelo com base no F1-score
    melhor = max(resultados, key=lambda r: r["f1"])
    print(f"\n Melhor modelo: {melhor['nome']} (F1: {melhor['f1']:.4f})")

    # Salva o melhor modelo
    joblib.dump(melhor["modelo"], "modelo.pkl")
    print(" Modelo salvo como 'modelo.pkl'")

"""
Entrena modelos para predecir gastos médicos.
Ejecutar UNA VEZ con: python entrenamiento.py
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 1. Cargar datos
df = pd.read_csv(BASE_DIR / "data" / "gastos_medicos.csv")

# 2. Definir features y target
features_num = ["edad", "imc", "hijos"]
features_cat = ["sexo", "fumador", "region"]
target = "gastos"

X = df[features_num + features_cat]
y = df[target]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def crear_preprocesador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), features_num),
        ("cat", OneHotEncoder(drop="first"), features_cat),
    ])


def entrenar_y_evaluar(nombre: str, estimador) -> tuple[Pipeline, dict]:
    pipeline = Pipeline([
        ("preprocesador", crear_preprocesador()),
        ("modelo", estimador),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metricas = {
        "r2_test": r2_score(y_test, y_pred),
        "rmse_test": rmse(y_test, y_pred),
        "mae_test": mean_absolute_error(y_test, y_pred),
        "r2_train": r2_score(y_train, pipeline.predict(X_train)),
    }
    return pipeline, metricas


modelos_config = [
    ("Regresión Lineal", LinearRegression(), MODELS_DIR / "modelo_gastos.pkl", "lr"),
    (
        "Árbol de Regresión",
        DecisionTreeRegressor(max_depth=5, random_state=42),
        MODELS_DIR / "modelo_arbol.pkl",
        "tree",
    ),
    (
        "Random Forest",
        RandomForestRegressor(n_estimators=100, random_state=42),
        MODELS_DIR / "modelo_rf.pkl",
        "rf",
    ),
]

os.makedirs(MODELS_DIR, exist_ok=True)
resultados = {}

print("=" * 50)
print("EVALUACIÓN DE MODELOS (conjunto de test)")
print("=" * 50)

for nombre, estimador, ruta, clave in modelos_config:
    pipeline, metricas = entrenar_y_evaluar(nombre, estimador)
    joblib.dump(pipeline, ruta)
    resultados[clave] = {
        "nombre": nombre,
        "archivo": ruta,
        **metricas,
    }
    print(f"\n--- {nombre} ---")
    print(f"R² (test):    {metricas['r2_test']:.4f}")
    print(f"R² (train):   {metricas['r2_train']:.4f}")
    print(f"MAE (test):   ${metricas['mae_test']:,.2f}")
    print(f"RMSE (test):  ${metricas['rmse_test']:,.2f}")
    print(f"Guardado en:  {ruta}")

mejor = max(resultados, key=lambda k: resultados[k]["r2_test"])
print("\n" + "=" * 50)
print(f"Mejor R² en test: {resultados[mejor]['nombre']} ({resultados[mejor]['r2_test']:.4f})")
print("=" * 50)

joblib.dump(
    {
        "features_num": features_num,
        "features_cat": features_cat,
        "modelos": resultados,
        "mejor_modelo": mejor,
    },
    MODELS_DIR / "metadata.pkl",
)
print(f"\nOK: Modelos y metadata guardados en {MODELS_DIR}/")

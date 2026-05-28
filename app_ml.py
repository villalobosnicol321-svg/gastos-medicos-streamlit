"""
App de Machine Learning: Predicción de gastos médicos.
Curso de Ciencia de Datos - Ingeniería Industrial.
"""
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Predicción de gastos médicos",
    page_icon="💰",
    layout="wide",
)

OPCIONES_MODELO = {
    "Regresión Lineal": "lr",
    "Árbol de Regresión": "tree",
    "Random Forest": "rf",
}


@st.cache_resource
def cargar_modelos():
    """Carga los tres pipelines y metadata una sola vez."""
    pipelines = {
        "lr": joblib.load(MODELS_DIR / "modelo_gastos.pkl"),
        "tree": joblib.load(MODELS_DIR / "modelo_arbol.pkl"),
        "rf": joblib.load(MODELS_DIR / "modelo_rf.pkl"),
    }
    metadata = joblib.load(MODELS_DIR / "metadata.pkl")
    return pipelines, metadata


if not (MODELS_DIR / "modelo_gastos.pkl").exists():
    st.error(
        "No se encontraron los modelos. Ejecuta primero: "
        "`python entrenamiento.py` desde la carpeta del proyecto."
    )
    st.stop()

pipelines, metadata = cargar_modelos()
info_modelos = metadata["modelos"]


def construir_entrada(edad, imc, hijos, sexo, fumador, region) -> pd.DataFrame:
    return pd.DataFrame([{
        "edad": edad,
        "imc": imc,
        "hijos": hijos,
        "sexo": sexo,
        "fumador": fumador,
        "region": region,
    }])


def nombres_features(pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocesador"].get_feature_names_out())


def grafico_interpretabilidad(pipeline) -> go.Figure:
    modelo = pipeline.named_steps["modelo"]
    nombres = nombres_features(pipeline)
    tipo = type(modelo).__name__

    if tipo == "LinearRegression":
        valores = modelo.coef_
        titulo = "Coeficientes (naranja: aumenta gastos · azul: los reduce)"
        etiqueta = "Coeficiente"
    else:
        valores = modelo.feature_importances_
        titulo = "Importancia de variables (árbol / bosque aleatorio)"
        etiqueta = "Importancia"

    tabla = pd.DataFrame({"variable": nombres, etiqueta: valores})
    tabla = tabla.sort_values(etiqueta, key=abs, ascending=False)

    fig = go.Figure(go.Bar(
        x=tabla[etiqueta],
        y=tabla["variable"],
        orientation="h",
        marker_color=["#ea580c" if v > 0 else "#1e3a8a" for v in tabla[etiqueta]],
    ))
    fig.update_layout(title=titulo, height=400, xaxis_title=etiqueta)
    return fig


def alerta_gasto(prediccion: float):
    if prediccion > 30000:
        st.error("⚠️ Gasto proyectado MUY ALTO")
    elif prediccion > 15000:
        st.warning("🟡 Gasto proyectado ALTO")
    else:
        st.success("✅ Gasto proyectado en rango normal")


def velocimetro(prediccion: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediccion,
        number={"prefix": "$", "valueformat": ",.0f"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Gasto estimado (USD)"},
        gauge={
            "axis": {"range": [0, 50000]},
            "bar": {"color": "#1e3a8a"},
            "steps": [
                {"range": [0, 10000], "color": "#dcfce7"},
                {"range": [10000, 25000], "color": "#fef3c7"},
                {"range": [25000, 50000], "color": "#fee2e2"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ---------- Encabezado ----------
st.title("💰 Predicción de gastos médicos")
st.markdown("""
Esta app estima los **gastos médicos anuales** de un paciente con regresión lineal,
árbol de regresión o random forest. Ingresa los datos y pulsa _Predecir_.
""")

with st.sidebar:
    st.header("🤖 Modelo principal")
    modelo_sel_label = st.selectbox(
        "Usar para velocímetro e interpretabilidad",
        list(OPCIONES_MODELO.keys()),
    )
    clave_sel = OPCIONES_MODELO[modelo_sel_label]
    info_sel = info_modelos[clave_sel]

    st.divider()
    st.header("📐 Calidad del modelo seleccionado")
    st.metric("R² en test", f"{info_sel['r2_test']:.3f}")
    st.metric("RMSE en test", f"${info_sel['rmse_test']:,.0f}")
    st.caption("Métricas sobre 20% de datos no vistos en entrenamiento.")
    with st.expander("Ver métricas de los 3 modelos"):
        tabla = pd.DataFrame([
            {
                "Modelo": info["nombre"],
                "R² test": f"{info['r2_test']:.3f}",
                "RMSE test": f"${info['rmse_test']:,.0f}",
            }
            for info in info_modelos.values()
        ])
        st.dataframe(tabla, hide_index=True, use_container_width=True)

# ---------- Formulario ----------
with st.form("formulario_paciente"):
    st.subheader("📋 Datos del paciente")

    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad", 18, 64, 30)
        imc = st.number_input(
            "IMC (Índice de Masa Corporal)",
            min_value=15.0, max_value=55.0, value=25.0, step=0.1,
        )
        hijos = st.number_input("Número de hijos", 0, 5, 0)
    with col2:
        sexo = st.radio("Sexo", ["mujer", "hombre"], horizontal=True)
        fumador = st.radio("¿Es fumador?", ["no", "si"], horizontal=True)
        region = st.selectbox(
            "Región",
            ["suroccidente", "suroriente", "noroccidente", "nororiente"],
        )

    enviado = st.form_submit_button(
        "🔮 Predecir gastos",
        type="primary",
        use_container_width=True,
    )

# ---------- Predicción ----------
if enviado:
    X_nuevo = construir_entrada(edad, imc, hijos, sexo, fumador, region)
    predicciones = {
        clave: pipelines[clave].predict(X_nuevo)[0]
        for clave in OPCIONES_MODELO.values()
    }
    pred_sel = predicciones[clave_sel]
    rmse_sel = info_sel["rmse_test"]

    st.divider()
    st.subheader("📊 Comparación entre modelos")

    cols = st.columns(3)
    for col, (label, clave) in zip(cols, OPCIONES_MODELO.items()):
        pred = predicciones[clave]
        info = info_modelos[clave]
        with col:
            st.markdown(f"**{label}**")
            st.metric("Gasto estimado", f"${pred:,.0f}")
            st.caption(f"R² test: {info['r2_test']:.3f} · RMSE: ${info['rmse_test']:,.0f}")
            if clave == clave_sel:
                st.caption("← modelo seleccionado en sidebar")

    st.divider()
    st.subheader(f"Detalle: {modelo_sel_label}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Gasto médico estimado",
            f"${pred_sel:,.0f}",
            help=f"Estimación con {modelo_sel_label}.",
        )
        limite_inf = max(0, pred_sel - rmse_sel)
        limite_sup = pred_sel + rmse_sel
        st.caption(
            f"Intervalo aproximado (±1 RMSE): "
            f"${limite_inf:,.0f} – ${limite_sup:,.0f}"
        )
        alerta_gasto(pred_sel)

    with col2:
        st.plotly_chart(velocimetro(pred_sel), use_container_width=True)

    pipeline_sel = pipelines[clave_sel]
    with st.expander("🔍 ¿Cómo interpreta el modelo cada variable?"):
        st.plotly_chart(
            grafico_interpretabilidad(pipeline_sel),
            use_container_width=True,
        )
        tipo = type(pipeline_sel.named_steps["modelo"]).__name__
        if tipo == "LinearRegression":
            st.caption(
                "Los coeficientes indican el cambio en gastos al aumentar una unidad "
                "(variables numéricas estandarizadas) o al pertenecer a esa categoría."
            )
        else:
            st.caption(
                "La importancia mide cuánto contribuye cada variable a las divisiones "
                "del árbol o del bosque; no indica dirección (sube/baja el gasto)."
            )

    mejor_clave = metadata["mejor_modelo"]
    mejor = info_modelos[mejor_clave]
    rf = info_modelos["rf"]
    tree = info_modelos["tree"]
    lr = info_modelos["lr"]

    with st.expander("¿Cuál modelo recomiendo?"):
        st.markdown(f"""
**Mejor R² en test:** {mejor['nombre']} (R² = {mejor['r2_test']:.3f},
RMSE = ${mejor['rmse_test']:,.0f}).

**Regresión lineal** (R² = {lr['r2_test']:.3f}): explica bien la tendencia general y es
fácil de interpretar con coeficientes, pero no captura bien interacciones fuertes
(por ejemplo, fumador con IMC alto).

**Árbol de regresión** (R² = {tree['r2_test']:.3f}, train = {tree['r2_train']:.3f}):
puede modelar no linealidades con reglas simples; si el R² en train es mucho mayor que en test,
hay riesgo de sobreajuste.

**Random Forest** (R² = {rf['r2_test']:.3f}, train = {rf['r2_train']:.3f}):
suele generalizar mejor que un solo árbol al promediar muchos árboles. Si train >> test,
conviene no usarlo en producción sin más validación.

**Recomendación para producción:** usaría **{mejor['nombre']}** porque obtuvo el mayor R²
en el conjunto de test (datos que el modelo no vio al entrenar). Para comunicar resultados
a usuarios no técnicos, la regresión lineal sigue siendo útil por su interpretabilidad;
para maximizar precisión predictiva en este dataset, priorizaría el modelo con mejor R² en test.
        """)

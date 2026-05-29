# Predicción y Análisis de Gastos Médicos

Proyecto de Ciencia de Datos — Ingeniería Industrial

Dos aplicaciones web construidas con **Streamlit** que permiten explorar, visualizar y predecir los gastos médicos anuales de pacientes asegurados.

---

## Estructura del proyecto

```
proyecto/
├── data/
│   └── gastos_medicos.csv       # Dataset de 1338 pacientes
├── models/                      # Generado automáticamente al entrenar
│   ├── modelo_gastos.pkl        # Pipeline: Regresión Lineal
│   ├── modelo_arbol.pkl         # Pipeline: Árbol de Regresión
│   ├── modelo_rf.pkl            # Pipeline: Random Forest
│   └── metadata.pkl             # Métricas y configuración
├── entrenamiento.py             # Script de entrenamiento (ejecutar una vez)
├── app_visualizacion.py         # App 1: Dashboard exploratorio
└── app_ml.py                    # App 2: Predictor de gastos
```

---

## Requisitos

```bash
pip install streamlit pandas scikit-learn joblib plotly
```

---

## Primeros pasos

### 1. Entrenar los modelos (solo la primera vez)

```bash
python entrenamiento.py
```

Este script lee `data/gastos_medicos.csv`, entrena tres modelos de regresión, los evalúa y guarda los pipelines en la carpeta `models/`. Al final imprime las métricas de cada modelo y cuál obtuvo el mejor R² en el conjunto de test.

### 2. Lanzar las aplicaciones

Cada app se ejecuta de forma independiente:

```bash
# Dashboard de visualización
streamlit run app_visualizacion.py

# Predictor con Machine Learning
streamlit run app_ml.py
```

---

## App 1 — Dashboard de gastos médicos (`app_visualizacion.py`)

**Objetivo:** explorar y entender los datos antes de modelar.

### Qué hace
- Carga el dataset de 1338 pacientes asegurados.
- Muestra 5 KPIs en tiempo real: número de pacientes, gasto promedio, gasto mediano, porcentaje de fumadores y edad promedio.
- Actualiza todos los gráficos dinámicamente según los filtros aplicados.

### Filtros disponibles (barra lateral)
| Filtro | Descripción |
|--------|-------------|
| Sexo | mujer / hombre |
| Fumador | sí / no |
| Región | suroccidente, suroriente, noroccidente, nororiente |
| Rango de edad | slider entre 18 y 64 años |
| Número de hijos (mín.) | slider de 0 a 5 |

### Pestañas
- **Distribuciones** — histograma de gastos, boxplot por hábito de fumar, barras por región y boxplot de gastos por región.
- **🔗 Relaciones** — scatter Edad vs. Gastos (tamaño = IMC), scatter IMC vs. Gastos con línea de tendencia, violín por sexo y hábito de fumar.
- **Datos** — tabla filtrable con opción de descarga en CSV.

---

## App 2 — Predictor de gastos médicos (`app_ml.py`)

**Objetivo:** estimar el gasto médico anual de un paciente nuevo a partir de sus características.

### Qué hace
- Carga los tres modelos entrenados y sus métricas de evaluación.
- Recibe los datos del paciente mediante un formulario.
- Muestra la predicción de los tres modelos en paralelo y un análisis detallado del modelo seleccionado.

### Cómo usar
1. En la **barra lateral**, elige el modelo principal (Regresión Lineal, Árbol de Regresión o Random Forest) y consulta sus métricas (R² y RMSE en test).
2. En el **formulario**, ingresa los datos del paciente:
   - Edad (18–64), IMC (15.0–55.0), número de hijos (0–5)
   - Sexo, condición de fumador y región
3. Pulsa **🔮 Predecir gastos**.

### Resultados
- **Comparación entre modelos**: gasto estimado por cada uno con su R² y RMSE.
- **Velocímetro**: indicador visual del gasto estimado en escala de riesgo (verde / amarillo / rojo).
- **Intervalo aproximado**: rango ±1 RMSE alrededor de la predicción.
- **Interpretabilidad**: gráfico de coeficientes (Regresión Lineal) o importancia de variables (Árbol / Random Forest).
- **Recomendación**: análisis automático de cuál modelo conviene usar según las métricas de test.

### Modelos disponibles
| Modelo | Descripción |
|--------|-------------|
| Regresión Lineal | Relación lineal entre variables; fácil de interpretar con coeficientes |
| Árbol de Regresión | Captura no linealidades con reglas de decisión; `max_depth=5` |
| Random Forest | Conjunto de 100 árboles; generalmente mayor precisión predictiva |

---

## Variables del dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `edad` | Numérica | Edad del paciente (años) |
| `imc` | Numérica | Índice de Masa Corporal |
| `hijos` | Numérica | Número de hijos cubiertos por el seguro |
| `sexo` | Categórica | `mujer` / `hombre` |
| `fumador` | Categórica | `si` / `no` |
| `region` | Categórica | Zona geográfica del asegurado |
| `gastos` | Numérica | **Variable objetivo** — gasto médico anual en USD |

---

## Preprocesamiento aplicado

El mismo pipeline se aplica en entrenamiento y predicción:

- **Variables numéricas** (`edad`, `imc`, `hijos`): estandarización con `StandardScaler`.
- **Variables categóricas** (`sexo`, `fumador`, `region`): codificación con `OneHotEncoder(drop="first")`.
- División train/test: 80% / 20% con `random_state=42`.

---

*Proyecto académico — Curso de Ciencia de Datos, Ingeniería Industrial.*

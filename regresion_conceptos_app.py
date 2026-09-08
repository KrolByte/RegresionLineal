"""
Módulo 6 — Regresión
App educativa: visualización interactiva de los conceptos principales

Cubre:
  1. Regresión lineal simple (ajuste manual vs. óptimo)
  2. Regresión lineal múltiple (coeficientes e importancia de variables)
  3. Función de costo (MSE) y el gradiente
  4. Descenso de gradiente (convergencia)
  5. Métricas de evaluación: R², MAE, RMSE

Ejecutar con:  streamlit run regresion_conceptos_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Regresión — Conceptos clave", layout="wide")

# Paleta corporativa (colores coherentes con la app)
CORP = {
    "primary": "#0B3954",   # azul oscuro (encabezados)
    "secondary": "#55A868", # verde (coeficientes)
    "accent": "#E15759",    # rojo/rosa (líneas de predicción / resaltado)
    "muted": "#8172B2",     # morado suave (puntos)
    "bg_start": "#f5fbff",
    "bg_mid": "#eef9f3",
    "bg_end": "#fffaf0",
    "table_header": "#0B3954",
}

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
FEATURE_LABELS = {
    "MedInc": "Ingreso medio",
    "HouseAge": "Edad de la vivienda",
    "AveRooms": "Habitaciones promedio",
    "AveBedrms": "Dormitorios promedio",
    "Population": "Población",
    "AveOccup": "Ocupantes promedio",
}

# ------------------------------------------------------------------
# Datos (reales: censo de vivienda de California)
# ------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.sample(n=1200, random_state=42).reset_index(drop=True)
    return df

# Añadimos una columna de fecha sintética para poder filtrar por rango si no hay fecha real
def añadir_fechas(df, inicio="1990-01-01", fin="1992-12-31"):
    df = df.copy()
    inicio = pd.to_datetime(inicio)
    fin = pd.to_datetime(fin)
    fechas = pd.date_range(inicio, fin, periods=len(df))
    df["date"] = fechas
    return df

# Metadatos estáticos (no forman parte de la configuración de búsqueda)
STATION_NAME = "Estación Central — CA Housing"
STATION_CODE = "EC-CA-1990"
DATA_QUALITY = "Alta (completa, sin valores faltantes)"

# Cargar datos base (sin fecha aún)
df = cargar_datos()

# -- Apariencia CSS (fondo, tipografías, tarjetas) --------------------------------
st.markdown(
    f"""
    <style>
    /* Fondo con degradado suave */
    .stApp {{
        background: linear-gradient(120deg, {CORP['bg_start']} 0%, {CORP['bg_mid']} 50%, {CORP['bg_end']} 100%);
    }}
    /* Encabezado grande */
    .big-title {{font-size:28px; font-weight:700; color:{CORP['primary']};}}
    /* Tarjetas de métricas personalizadas */
    .metric-box {{background: linear-gradient(90deg,#ffffffcc,#f0f8ffcc); padding:10px; border-radius:8px;}}
    /* Tabla bonita */
    .dataframe thead tr th {{background-color: {CORP['table_header']}; color: white;}}
    /* Ajustes para el expander */
    .stExpander {{ background-color: rgba(255,255,255,0.6); border-radius:8px; padding:8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ BARRA LATERAL: configuración mínima (solo rango de fechas) ----
with st.sidebar:
    # Opción para subir logo o proveer URL (se muestra en encabezado)
    st.markdown("### Branding")
    logo_upload = st.file_uploader("(Opcional) Subir logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    logo_url = st.text_input("(Opcional) URL de logo", value="")

    st.markdown(f"<div class='metric-box'><strong>{STATION_NAME}</strong><br>Codigo: {STATION_CODE}<br>Calidad: {DATA_QUALITY}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("🔎 Filtros")
    st.markdown("Los metadatos de estación son estáticos arriba; puedes subir un CSV con columna 'date' para usar fechas reales, o usar las fechas sintéticas por defecto.")

    uploaded = st.file_uploader("(Opcional) Subir CSV con columna 'date' para usar fechas reales", type=["csv"]) 
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            # Verificar columnas mínimas (MedHouseVal y features)
            required = set(FEATURES + ["MedHouseVal"]) 
            missing = required - set(df_up.columns)
            if missing:
                st.warning(f"El CSV subido no contiene todas las columnas requeridas: {sorted(list(missing))}. Se ignorará y se usará el dataset interno.")
            else:
                if "date" in df_up.columns:
                    df_up["date"] = pd.to_datetime(df_up["date"], errors="coerce")
                    if df_up["date"].isna().all():
                        st.warning("La columna 'date' existe pero no se pudo parsear. Se usará fecha sintética.")
                    else:
                        df = df_up.copy()
                        st.success("CSV cargado y columna 'date' detectada — usando fechas reales del archivo.")
                else:
                    st.warning("El CSV no contiene columna 'date' — se usará la fecha sintética por defecto.")
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    # Si no existe columna 'date' en df, añadimos fechas sintéticas
    if "date" not in df.columns:
        df = añadir_fechas(df)

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    date_range = st.date_input("Rango de fechas (desde - hasta)", value=[date_min, date_max], min_value=date_min, max_value=date_max)

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        df = df.loc[mask].reset_index(drop=True)

    st.markdown("---")
    st.markdown("## Resumen rápido")
    st.metric("Registros", f"{len(df)}")
    st.metric("Precio medio (MedHouseVal)", f"{df['MedHouseVal'].mean():.3f}")

# Encabezado principal con logo (si se proporcionó)
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if logo_upload is not None:
        st.image(logo_upload, width=120)
    elif logo_url:
        try:
            st.image(logo_url, width=120)
        except Exception:
            st.text("")
    else:
        # espacio reservado
        st.write("")

with col_title:
    st.markdown(f"<h1 class='big-title'>📈 Regresión — Conceptos clave</h1>", unsafe_allow_html=True)
    st.markdown(
        "La regresión permite predecir valores numéricos a partir de datos históricos. "
        "Esta app recorre, de forma interactiva, las piezas que componen un modelo de regresión: "
        "**el modelo, la función de costo, el gradiente, el algoritmo de aprendizaje y las métricas "
        "para evaluar qué tan bien predice.** Todo con datos reales de vivienda en California.")

# Mostrar tabla de datos con estilo bonito (limitada para no sobrecargar la UI)
with st.expander("📋 Ver tabla de datos (muestra)", expanded=False):
    muestra = df.sort_values("date", ascending=False).head(200)
    # Selección de columnas para mostrar
    mostrar_cols = ["date", "MedHouseVal"] + FEATURES
    mostrar = muestra[mostrar_cols].copy()
    mostrar["date"] = mostrar["date"].dt.date
    # Usamos Styler para formato y gradiente
    sty = mostrar.style.format({"MedHouseVal": "{:.3f}"}).background_gradient(subset=FEATURES + ["MedHouseVal"], cmap="Blues")
    st.write("Tabla (muestra):")
    st.write(sty.to_html(), unsafe_allow_html=True)

    # Botón para descargar la tabla filtrada
    csv_bytes = mostrar.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar tabla filtrada (CSV)", data=csv_bytes, file_name="tabla_filtrada.csv", mime="text/csv")

# Pestañas principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Regresión simple",
    "2️⃣ Regresión múltiple",
    "3️⃣ Costo y gradiente",
    "4️⃣ Descenso de gradiente",
    "5️⃣ Métricas de evaluación",
])

# (Usamos un estilo gráfico coherente)
plt.style.use("seaborn-v0_8")

# ====================================================================
# TAB 1 — REGRESIÓN SIMPLE
# ====================================================================
with tab1:
    st.header("Regresión lineal simple")
    st.markdown(
        r"""
Un modelo de regresión lineal simple predice un valor con una sola variable de entrada:

$$\hat{y} = w \cdot x + b$$

Mueve `w` (pendiente) y `b` (intercepto) manualmente y observa cómo cambia el error,
 o deja que el algoritmo encuentre el ajuste óptimo.
"""
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        var_simple = st.selectbox(
            "Variable predictora", FEATURES,
            format_func=lambda v: f"{v} — {FEATURE_LABELS[v]}",
            key="var_simple",
        )
        x = df[var_simple].values
        y = df["MedHouseVal"].values

        modo = st.radio("Modo", ["Ajuste manual", "Ajuste óptimo (automático)"], key="modo_simple")

        if modo == "Ajuste manual":
            w_manual = st.slider("w (pendiente)", -3.0, 3.0, 0.0, 0.01)
            b_manual = st.slider("b (intercepto)", -3.0, 3.0, float(y.mean()), 0.01)
            w_used, b_used = w_manual, b_manual
        else:
            modelo = LinearRegression()
            modelo.fit(x.reshape(-1, 1), y)
            w_used, b_used = modelo.coef_[0], modelo.intercept_
            st.info(f"w óptimo = {w_used:.4f}  ·  b óptimo = {b_used:.4f}", icon="✅")

        y_pred = w_used * x + b_used
        mse_actual = np.mean((y_pred - y) ** 2)
        st.metric("Error actual (MSE)", f"{mse_actual:.4f}")

    with col_b:
        # Gráfico interactivo con plotly usando colores corporativos
        fig_px = px.scatter(df, x=var_simple, y="MedHouseVal", opacity=0.25, color_discrete_sequence=[CORP['muted']])
        # añadir línea de predicción
        x_line = np.linspace(df[var_simple].min(), df[var_simple].max(), 100)
        y_line = w_used * x_line + b_used
        fig_px.add_traces(px.line(x=x_line, y=y_line, color_discrete_sequence=[CORP['accent']]).data)
        fig_px.update_layout(height=420, title=f"ŷ = {w_used:.3f}·x + {b_used:.3f}", title_font_color=CORP['primary'])
        st.plotly_chart(fig_px, use_container_width=True)

    st.caption(
        "💡 En modo manual, intenta minimizar el MSE moviendo los sliders. "
        "Luego compara contra el ajuste automático: ese es exactamente el problema que resuelve el entrenamiento."
    )

# ====================================================================
# TAB 2 — REGRESIÓN MÚLTIPLE
# ====================================================================
with tab2:
    st.header("Regresión lineal múltiple")
    st.markdown(
        r"""
En la práctica, varias variables influyen a la vez en la predicción:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_k x_k + b$$

Selecciona qué variables incluir y observa cómo cambian los coeficientes y el desempeño del modelo.
"""
    )

    vars_multiples = st.multiselect(
        "Variables a incluir en el modelo",
        FEATURES,
        default=["MedInc", "HouseAge", "AveRooms"],
        format_func=lambda v: f"{v} — {FEATURE_LABELS[v]}",
    )

    if len(vars_multiples) == 0:
        st.warning("Selecciona al menos una variable.")
    else:
        X = df[vars_multiples]
        y = df["MedHouseVal"]

        modelo_m = LinearRegression()
        modelo_m.fit(X, y)
        y_pred_m = modelo_m.predict(X)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("**Coeficientes aprendidos**")
            coefs = pd.Series(modelo_m.coef_, index=vars_multiples).sort_values()
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            coefs.plot(kind="barh", ax=ax2, color=CORP['secondary'])
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("Coeficiente (w)")
            st.pyplot(fig2)
            st.caption(f"Intercepto (b) = {modelo_m.intercept_:.4f}")

        with col_d:
            st.markdown("**Real vs. predicho**")
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            ax3.scatter(y, y_pred_m, alpha=0.15, s=12, color=CORP['accent'])
            lims = [min(y.min(), y_pred_m.min()), max(y.max(), y_pred_m.max())]
            ax3.plot(lims, lims, "--", color="black", linewidth=1)
            ax3.set_xlabel("Valor real")
            ax3.set_ylabel("Valor predicho")
            r2_m = r2_score(y, y_pred_m)
            ax3.set_title(f"R² = {r2_m:.3f}")
            st.pyplot(fig3)

        st.caption(
            "💡 Agrega o quita variables y observa: más variables relevantes normalmente mejora el R², "
            "pero cada coeficiente representa el efecto de esa variable manteniendo las demás fijas."
        )

# ====================================================================
# TAB 3 — FUNCIÓN DE COSTO Y GRADIENTE
# ====================================================================
with tab3:
    st.header("Función de costo (MSE) y el gradiente")
    st.markdown(
        r"""
El error cuadrático medio mide qué tan mal predice el modelo:

$$J(w) = \frac{1}{n}\sum_{i=1}^{n}(w x_i + b - y_i)^2$$

Es una función con forma de "tazón": tiene un único mínimo. El **gradiente** es la pendiente
 de esa curva en un punto — indica hacia dónde y qué tanto hay que mover `w` para reducir el error.
"""
    )

    var_costo = st.selectbox(
        "Variable", FEATURES, format_func=lambda v: f"{v} — {FEATURE_LABELS[v]}", key="var_costo"
    )
    x_c = df[var_costo].values
    y_c = df["MedHouseVal"].values
    x_norm = (x_c - x_c.mean()) / x_c.std()
    y_norm = (y_c - y_c.mean()) / y_c.std()

    modelo_c = LinearRegression()
    modelo_c.fit(x_norm.reshape(-1, 1), y_norm)
    w_opt, b_opt = modelo_c.coef_[0], modelo_c.intercept_

    w_probe = st.slider(
        "Explora distintos valores de w (b fijo en su valor óptimo)",
        float(w_opt - 2), float(w_opt + 2), float(w_opt), 0.01,
    )

    def mse(w, b, x, y):
        return np.mean((w * x + b - y) ** 2)

    def grad_w(w, b, x, y):
        return (2 / len(x)) * np.sum((w * x + b - y) * x)

    w_range = np.linspace(w_opt - 2, w_opt + 2, 200)
    costos = [mse(w, b_opt, x_norm, y_norm) for w in w_range]

    costo_actual = mse(w_probe, b_opt, x_norm, y_norm)
    gradiente_actual = grad_w(w_probe, b_opt, x_norm, y_norm)

    col_e, col_f = st.columns([2, 1])

    with col_e:
        fig4, ax4 = plt.subplots(figsize=(6, 4.2))
        ax4.plot(w_range, costos, color=CORP['primary'], linewidth=2)
        ax4.axvline(w_opt, color="green", linestyle="--", linewidth=1, label=f"w óptimo ≈ {w_opt:.3f}")
        ax4.scatter([w_probe], [costo_actual], color=CORP['accent'], s=80, zorder=5, label="Tu punto actual")

        # Recta tangente (visualiza el gradiente)
        tang_x = np.linspace(w_probe - 0.6, w_probe + 0.6, 20)
        tang_y = costo_actual + gradiente_actual * (tang_x - w_probe)
        ax4.plot(tang_x, tang_y, color="orange", linewidth=2, label="Tangente (gradiente)")

        ax4.set_xlabel("w")
        ax4.set_ylabel("J(w)  —  MSE")
        ax4.set_title("Función de costo — el 'tazón'")
        ax4.legend(fontsize=8)
        st.pyplot(fig4)

    with col_f:
        st.metric("Costo J(w)", f"{costo_actual:.4f}")
        st.metric("Gradiente ∂J/∂w", f"{gradiente_actual:.4f}")
        if abs(gradiente_actual) < 0.02:
            st.success("Gradiente ≈ 0 → estás cerca del mínimo ✅")
        elif gradiente_actual > 0:
            st.warning("Gradiente positivo → hay que **disminuir** w")
        else:
            st.warning("Gradiente negativo → hay que **aumentar** w")

    st.caption(
        "💡 El gradiente es la pendiente de la tangente (línea naranja). Cuando es 0, estás en el mínimo — "
        "el punto de menor error. Esa es la señal que usa el descenso de gradiente para actualizar w."
    )

# ====================================================================
# TAB 4 — DESCENSO DE GRADIENTE
# ====================================================================
with tab4:
    st.header("Descenso de gradiente")
    st.markdown(
        r"""
El algoritmo que "aprende" los parámetros óptimos, iteración por iteración:

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w} \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

Configura el *learning rate* (`α`) y el número de iteraciones, y observa cómo converge (o diverge).
"""
    )

    col_g, col_h = st.columns([1, 2])

    with col_g:
        var_gd = st.selectbox(
            "Variable", FEATURES, format_func=lambda v: f"{v} — {FEATURE_LABELS[v]}", key="var_gd"
        )
        alpha_gd = st.slider("Learning rate (α)", 0.001, 1.0, 0.1, 0.001, key="alpha_gd")
        epochs_gd = st.slider("Iteraciones", 5, 200, 50, key="epochs_gd")

    x_gd = df[var_gd].values
    y_gd = df["MedHouseVal"].values
    x_gd_n = (x_gd - x_gd.mean()) / x_gd.std()
    y_gd_n = (y_gd - y_gd.mean()) / y_gd.std()

    def descenso_gradiente(x, y, alpha, epochs):
        n = len(x)
        w, b = 0.0, 0.0
        historial = []
        for _ in range(epochs):
            pred = w * x + b
            error = pred - y
            gw = (2 / n) * np.sum(error * x)
            gb = (2 / n) * np.sum(error)
            w -= alpha * gw
            b -= alpha * gb
            historial.append(np.mean(error ** 2))
        return w, b, historial

    w_final, b_final, historial_costo = descenso_gradiente(x_gd_n, y_gd_n, alpha_gd, epochs_gd)

    with col_g:
        st.metric("w final", f"{w_final:.4f}")
        st.metric("b final", f"{b_final:.4f}")
        st.metric("Costo final", f"{historial_costo[-1]:.4f}")
        if historial_costo[-1] > historial_costo[0]:
            st.error("⚠️ El costo aumentó — el α es demasiado grande (diverge)")

    with col_h:
        fig5, ax5 = plt.subplots(figsize=(6, 4.2))
        ax5.plot(historial_costo, color="#DD8452", linewidth=2)
        ax5.set_xlabel("Iteración")
        ax5.set_ylabel("Costo (MSE)")
        ax5.set_title(f"Convergencia — α = {alpha_gd}")
        st.pyplot(fig5)

    st.caption(
        "💡 Prueba un α muy pequeño (converge lento) y uno muy grande (puede oscilar o divergir). "
        "El valor adecuado depende de cada problema — por eso se explora experimentalmente."
    )

# ====================================================================
# TAB 5 — MÉTRICAS DE EVALUACIÓN
# ====================================================================
with tab5:
    st.header("Métricas de evaluación: R², MAE, RMSE")
    st.markdown(
        r"""
Con el modelo entrenado, necesitamos medir qué tan bien predice sobre datos **nunca vistos**:

- **MAE** — error absoluto promedio, en las unidades originales.
- **RMSE** — penaliza más los errores grandes; es $\sqrt{MSE}$.
- **R²** — proporción de la variabilidad de `y` que el modelo logra explicar (0 a 1).
"""
    )

    vars_metricas = st.multiselect(
        "Variables del modelo",
        FEATURES,
        default=["MedInc", "HouseAge", "AveRooms", "Population"],
        format_func=lambda v: f"{v} — {FEATURE_LABELS[v]}",
        key="vars_metricas",
    )
    test_size = st.slider("Proporción de datos de prueba", 0.1, 0.5, 0.2, 0.05)

    if len(vars_metricas) == 0:
        st.warning("Selecciona al menos una variable.")
    else:
        X = df[vars_metricas]
        y = df["MedHouseVal"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        modelo_met = LinearRegression()
        modelo_met.fit(X_train, y_train)
        y_pred_test = modelo_met.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)

        col_i, col_j, col_k = st.columns(3)
        col_i.metric("MAE", f"{mae:.3f}")
        col_j.metric("RMSE", f"{rmse:.3f}")
        col_k.metric("R²", f"{r2:.3f}", help="1.0 = predicción perfecta, 0.0 = igual que predecir siempre el promedio")

        # Gráfico interactivo de real vs predicho usando paleta corporativa
        fig6 = px.scatter(x=y_test, y=y_pred_test, labels={"x":"Valor real","y":"Valor predicho"}, opacity=0.6, color_discrete_sequence=[CORP['muted']])
        fig6.add_traces(px.line(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color_discrete_sequence=[CORP['primary']]).data)
        fig6.update_layout(title="Desempeño sobre datos de PRUEBA (nunca vistos)", height=420, title_font_color=CORP['primary'])
        st.plotly_chart(fig6, use_container_width=True)

        st.caption(
            "💡 Estas métricas se calculan sobre el conjunto de **prueba**, no el de entrenamiento — "
            "así sabemos si el modelo generaliza a datos nuevos, no solo si memorizó los que ya vio."
        )

st.markdown("---")
st.caption(
    "Datos: censo de vivienda de California (1990), vía scikit-learn. "
    "App complementaria al Módulo 6 — Regresión."
)

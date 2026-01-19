"""
Dashboard Financiero con Machine Learning
-----------------------------------------
Autor: Evee_
Tech Stack: Streamlit, Yahoo Finance, Prophet, Plotly
"""

import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# 1. Configuración de la página
st.set_page_config(
    page_title="Predictor Financiero AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('📈 Dashboard Financiero con Predicción AI')

# 2. Sidebar: Parámetros del usuario
st.sidebar.header("Configuración")
selected_stock = st.sidebar.text_input("Símbolo (Ticker)", "AAPL")

# --- CORRECCIÓN 1: Aumentado el límite a 10 años ---
n_years = st.sidebar.slider('Años de historia para entrenar:', 1, 10, 5)

# Slider 2: Cuánto FUTURO predecir
prediction_months = st.sidebar.slider('Meses a predecir:', 1, 24, 12)

# 3. Calcular fecha de inicio dinámica
start_date = date.today() - timedelta(days=n_years * 365)
start_date_str = start_date.strftime("%Y-%m-%d")

# 4. Función de Carga de datos
@st.cache_data
def load_data(ticker, start):
    """Descarga y limpia los datos de Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start, end=date.today().strftime("%Y-%m-%d"))
        
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# Ejecución de carga
data_load_state = st.text('Cargando datos...')
data = load_data(selected_stock, start_date_str)
data_load_state.text('¡Datos cargados!')

# 5. Lógica Principal y Visualización
if data.empty:
    st.error(f"⚠️ No se encontraron datos para '{selected_stock}'.")
else:
    # --- CORRECCIÓN 2: Tabla interactiva en vez de estática ---
    st.subheader(f'Datos Históricos ({n_years} años)')
    st.caption("Usa el scroll en la tabla para ver los datos antiguos.")
    # st.dataframe permite hacer scroll y ver todos los años, no solo el final
    st.dataframe(data, height=300, use_container_width=True)

    # --- PREDICCIÓN CON MACHINE LEARNING ---
    st.subheader(f'🔮 Predicción de Precio a {prediction_months} meses')

    df_train = data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    if len(df_train) < 20:
        st.warning("⚠️ Necesitas más datos históricos.")
    else:
        with st.spinner('Entrenando modelo de IA...'):
            m = Prophet()
            m.fit(df_train)

            future = m.make_future_dataframe(periods=prediction_months * 30)
            forecast = m.predict(future)

            # Mostrar tabla de predicciones (Interactivo también)
            st.write("Datos de la proyección futura:")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_months*30), height=200)

            # --- CORRECCIÓN 3: Gráfico SOLO PREDICCIÓN ---
            fig_custom = go.Figure()

            # Calculamos dónde empieza el futuro para pintar solo desde ahí
            last_real_date = data['Date'].max()
            future_only = forecast[forecast['ds'] > last_real_date]

            # Línea de Predicción (Roja)
            fig_custom.add_trace(go.Scatter(
                x=future_only['ds'],
                y=future_only['yhat'],
                name="Tendencia Futura",
                line=dict(color='#ff2b2b', width=4) # Rojo sólido intenso
            ))

            # Intervalo de Confianza Superior
            fig_custom.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            # Intervalo de Confianza Inferior (Crea la sombra)
            fig_custom.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_lower'],
                fill='tonexty', mode='lines', line=dict(width=0),
                fillcolor='rgba(255, 43, 43, 0.2)', # Sombra roja
                showlegend=False, hoverinfo='skip'
            ))

            fig_custom.update_layout(
                title=f"Proyección Futura Exclusiva: {selected_stock}",
                xaxis_title="Fecha Futura",
                yaxis_title="Precio Estimado (USD)",
                hovermode="x unified",
                showlegend=True
            )

            st.plotly_chart(fig_custom, use_container_width=True)

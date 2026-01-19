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

# Slider 1: Historia (hasta 10 años)
n_years = st.sidebar.slider('Años de historia para entrenar:', 1, 10, 5)

# Slider 2: Futuro
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
    # --- SECCIÓN 1: DATOS HISTÓRICOS  ---
    st.subheader(f'Datos Históricos ({n_years} años)')
    
    # Tabla con scroll
    st.dataframe(data, height=200, use_container_width=True)

    # GRÁFICA HISTÓRICA 
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'], 
            y=data['Close'], 
            name="Precio Cierre",
            line=dict(color='blue')
        ))
        fig.layout.update(
            title_text=f'Evolución Histórica: {selected_stock}', 
            xaxis_rangeslider_visible=True,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    plot_raw_data()

    # --- SECCIÓN 2: PREDICCIÓN  ---
    st.markdown("---") 
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

            # Tabla de predicción
            st.write("Datos de la proyección futura:")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_months*30), height=200)

            # GRÁFICA DE PREDICCIÓN (Solo Futuro)
            fig_custom = go.Figure()

            last_real_date = data['Date'].max()
            future_only = forecast[forecast['ds'] > last_real_date]

            fig_custom.add_trace(go.Scatter(
                x=future_only['ds'],
                y=future_only['yhat'],
                name="Tendencia Futura",
                line=dict(color='#ff2b2b', width=4) 
            ))

            fig_custom.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_custom.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_lower'],
                fill='tonexty', mode='lines', line=dict(width=0),
                fillcolor='rgba(255, 43, 43, 0.2)', 
                showlegend=False, hoverinfo='skip'
            ))

            fig_custom.update_layout(
                title=f"Proyección Futura Exclusiva: {selected_stock}",
                xaxis_title="Fecha Futura",
                yaxis_title="Precio Estimado (USD)",
                hovermode="x unified"
            )

            st.plotly_chart(fig_custom, use_container_width=True)

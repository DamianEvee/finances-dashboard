"""
Dashboard Financiero Profesional
-----------------------------------------
Autor: Evee_
Tech Stack: Streamlit, Yahoo Finance, Prophet, Plotly
"""

import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
import pandas as pd

# 1. Configuración de la página 
st.set_page_config(
    page_title="AI Stock Vision",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal 
st.title('AI Stock Vision: Análisis y Predicción')

# 2. Sidebar: Panel de Control
st.sidebar.header("⚙️ Panel de Control")
selected_stock = st.sidebar.text_input("Símbolo (Ticker)", "AAPL").upper() 

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de IA")
n_years = st.sidebar.slider('Años de aprendizaje:', 1, 10, 5)
prediction_months = st.sidebar.slider('Meses a predecir:', 1, 24, 12)

# 3. Función de Carga 
start_date = date.today() - timedelta(days=n_years * 365)
start_date_str = start_date.strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker, start):
    try:

        end_date = date.today() + timedelta(days=1)
        df = yf.download(ticker, start=start, end=end_date.strftime("%Y-%m-%d"))
        
        if df.empty: return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        df = df.sort_values('Date')
        return df
    except Exception as e:
        return pd.DataFrame()

# Cargar datos
data = load_data(selected_stock, start_date_str)

# 4. Lógica Principal
if data.empty:
    st.error(f" No se encontraron datos para '{selected_stock}'. Intenta con otro ticker (ej: TSLA, MSFT, BTC-USD).")
else:
    # --- A. METRICAS ENCABEZADO ---
    # Calculamos precio actual y variación
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    variation = last_close - prev_close
    pct_variation = (variation / prev_close) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label=f"Precio Actual ({selected_stock})", 
                  value=f"${last_close:.2f}", 
                  delta=f"{variation:.2f} ({pct_variation:.2f}%)")
    
    with col2:
        st.metric(label="Volumen", 
                  value=f"{data['Volume'].iloc[-1]:,}")

    with col3:
        st.metric(label="Máximo (52 semanas)", 
                  value=f"${data['High'].tail(365).max():.2f}")
        
    with col4:
        st.metric(label="Mínimo (52 semanas)", 
                  value=f"${data['Low'].tail(365).min():.2f}")

    st.markdown("---")

    # --- B. SISTEMA DE PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs([" Análisis Histórico", " Predicción IA", " Datos"])

    # PESTAÑA 1: Histórico
    with tab1:
        st.subheader("Evolución del Precio")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'], 
            name="Precio Cierre",
            line=dict(color='#0068C9', width=2),
            fill='tozeroy', 
            fillcolor='rgba(0, 104, 201, 0.1)'
        ))
        fig_hist.layout.update(
            xaxis_rangeslider_visible=True,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # PESTAÑA 2: Predicción IA (Prophet)
    with tab2:
        st.subheader(f"Proyección a {prediction_months} meses")
        
        # Preparamos datos para Prophet
        df_train = data[['Date', 'Close']].copy()
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        if len(df_train) < 20:
            st.warning(" Datos insuficientes para predecir.")
        else:
            with st.spinner(' La IA está analizando tendencias...'):
                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=prediction_months * 30)
                forecast = m.predict(future)

                # Gráfico Rojo = Futuro
                fig_pred = go.Figure()
                
                last_real_date = data['Date'].max()
                future_only = forecast[forecast['ds'] > last_real_date]

                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat'],
                    name="Predicción IA",
                    line=dict(color='#FF4B4B', width=3)
                ))
                
                # Intervalo de confianza
                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat_upper'],
                    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat_lower'],
                    fill='tonexty', mode='lines', line=dict(width=0),
                    fillcolor='rgba(255, 75, 75, 0.2)',
                    showlegend=False, hoverinfo='skip'
                ))

                fig_pred.update_layout(
                    title="Tendencia Esperada",
                    yaxis_title="Precio Estimado (USD)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Análisis de tendencia simple
                start_pred = future_only['yhat'].iloc[0]
                end_pred = future_only['yhat'].iloc[-1]
                trend = "ALCISTA " if end_pred > start_pred else "BAJISTA "
                
                st.info(f"Tendencia proyectada: **{trend}** (Basado en análisis de estacionalidad)")

                # Botón de descarga CSV
                csv = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=" Descargar Predicción (CSV)",
                    data=csv,
                    file_name=f'prediccion_{selected_stock}.csv',
                    mime='text/csv',
                )

    # PESTAÑA 3: Datos Crudos
    with tab3:
        st.subheader("Datos Recientes")
        st.dataframe(data.sort_values('Date', ascending=False), use_container_width=True)

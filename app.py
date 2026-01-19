"""
Dashboard Financiero Profesional v3.0 (Clean Version)
-----------------------------------------
Autor: Evee_
Tech Stack: Streamlit, Yahoo Finance, Prophet, Plotly
Features: Catálogo de acciones clasificado por sector. Sin emojis.
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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('AI Stock Vision: Análisis y Predicción')

# --- BASE DE DATOS DE TICKERS (CATÁLOGO) ---
STOCK_DB = {
    "Favoritos": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
    "Tecnología & Software": ["AMD", "INTC", "CRM", "ADBE", "ORCL", "IBM", "SAP", "SPOT"],
    "Finanzas & Bancos": ["JPM", "BAC", "V", "MA", "GS", "MS", "PYPL", "AXP"],
    "Automotriz": ["F", "GM", "TM", "HMC", "RACE", "STLA"],
    "Salud & Farmacéutica": ["JNJ", "PFE", "MRNA", "LLY", "UNH", "ABBV"],
    "Consumo & Retail": ["WMT", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS"],
    "Energía & Petróleo": ["XOM", "CVX", "SHELL", "BP", "TTE"],
    "Minerales & Commodities": ["GOLD", "SLV", "FCX", "RIO", "VALE", "CL=F", "GC=F"],
    "Criptomonedas": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"],
    "Índices & ETFs": ["SPY", "QQQ", "DIA", "IWM", "VOO", "VTI"],
    "Búsqueda Manual": [] 
}

# 2. Sidebar: Panel de Control
st.sidebar.header("Selección de Activo")

# Selector de Categoría
sector = st.sidebar.selectbox("Selecciona un Sector:", list(STOCK_DB.keys()))

# Lógica para seleccionar el Ticker
if sector == "Búsqueda Manual":
    selected_stock = st.sidebar.text_input("Escribe el Ticker (ej: BABA):", "AAPL").upper()
else:
    selected_stock = st.sidebar.selectbox("Selecciona la Empresa:", STOCK_DB[sector])

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
data_load_state = st.empty() 
data_load_state.text('Descargando datos del mercado...')
data = load_data(selected_stock, start_date_str)
data_load_state.empty() 

# 4. Lógica Principal
if data.empty:
    st.error(f"No pudimos encontrar datos para {selected_stock}. Si usaste búsqueda manual, verifica el símbolo en Yahoo Finance.")
else:
    # --- A. METRICAS ---
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    variation = last_close - prev_close
    pct_variation = (variation / prev_close) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Precio ({selected_stock})", f"${last_close:.2f}", f"{variation:.2f} ({pct_variation:.2f}%)")
    with col2:
        st.metric("Volumen", f"{data['Volume'].iloc[-1]:,}")
    with col3:
        st.metric("Máximo Anual", f"${data['High'].tail(365).max():.2f}")
    with col4:
        st.metric("Mínimo Anual", f"${data['Low'].tail(365).min():.2f}")

    st.markdown("---")

    # --- B. PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["Histórico", "Predicción IA", "Datos"])

    # PESTAÑA 1: Histórico
    with tab1:
        st.subheader(f"Evolución: {selected_stock}")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'], 
            name="Cierre",
            line=dict(color='#0068C9', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 104, 201, 0.1)'
        ))
        fig_hist.layout.update(xaxis_rangeslider_visible=True, hovermode="x unified", margin=dict(t=30))
        st.plotly_chart(fig_hist, use_container_width=True)

    # PESTAÑA 2: Predicción
    with tab2:
        st.subheader(f"Proyección IA a {prediction_months} meses")
        
        df_train = data[['Date', 'Close']].copy()
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        if len(df_train) < 20:
            st.warning("Datos insuficientes para predecir.")
        else:
            with st.spinner('Analizando tendencias de mercado...'):
                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=prediction_months * 30)
                forecast = m.predict(future)

                fig_pred = go.Figure()
                last_real_date = data['Date'].max()
                future_only = forecast[forecast['ds'] > last_real_date]

                # Línea predicción
                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat'],
                    name="Tendencia IA",
                    line=dict(color='#FF4B4B', width=3)
                ))
                # Sombras (Intervalo de confianza)
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

                fig_pred.update_layout(title="Proyección de Valor", hovermode="x unified")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Análisis de texto
                change = future_only['yhat'].iloc[-1] - future_only['yhat'].iloc[0]
                trend_text = "ALCISTA" if change > 0 else "BAJISTA"
                st.info(f"Tendencia proyectada: {trend_text}")

                # Descarga CSV
                csv = future_only.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar Proyección (CSV)", csv, f'pred_{selected_stock}.csv', 'text/csv')

    # PESTAÑA 3: Datos
    with tab3:
        st.subheader("Datos en Crudo")
        st.dataframe(data.sort_values('Date', ascending=False), use_container_width=True)

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
    page_title="AI Stock Vision (EUR)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('🇪🇺 AI Stock Vision: Análisis en Euros')

# --- BASE DE DATOS DE TICKERS ---
STOCK_DB = {
    "Favoritos": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
    "Tecnología": ["AMD", "INTC", "CRM", "ADBE", "ORCL", "IBM", "SAP", "SPOT"],
    "Finanzas": ["JPM", "BAC", "V", "MA", "GS", "MS", "PYPL", "AXP"],
    "Automotriz": ["F", "GM", "TM", "HMC", "RACE", "STLA"],
    "Salud": ["JNJ", "PFE", "MRNA", "LLY", "UNH", "ABBV"],
    "Consumo": ["WMT", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS"],
    "Energía": ["XOM", "CVX", "SHELL", "BP", "TTE"],
    "Minerales": ["GOLD", "SLV", "FCX", "RIO", "VALE", "CL=F", "GC=F"],
    "Criptomonedas": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"],
    "Índices": ["SPY", "QQQ", "DIA", "IWM", "VOO", "VTI"],
    "Búsqueda Manual": [] 
}

# 2. Sidebar
st.sidebar.header("Selección de Activo")
sector = st.sidebar.selectbox("Selecciona un Sector:", list(STOCK_DB.keys()))

if sector == "Búsqueda Manual":
    selected_stock = st.sidebar.text_input("Escribe el Ticker:", "AAPL").upper()
else:
    selected_stock = st.sidebar.selectbox("Selecciona la Empresa:", STOCK_DB[sector])

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de IA")
n_years = st.sidebar.slider('Años de aprendizaje:', 1, 10, 5)
prediction_months = st.sidebar.slider('Meses a predecir:', 1, 24, 12)

# 3. Carga de Datos y Conversión
start_date = date.today() - timedelta(days=n_years * 365)
start_date_str = start_date.strftime("%Y-%m-%d")

@st.cache_data
def get_exchange_rate():
    """Descarga el tipo de cambio actual USD -> EUR"""
    try:
        # 'EUR=X' es el ticker de Yahoo para USD a EUR
        fx = yf.download("EUR=X", period="1d")
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        return fx['Close'].iloc[-1]
    except:
        return 1.0 # Si falla, usamos 1 a 1 para no romper la app

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
        
        # --- CONVERSIÓN A EUROS ---
        rate = get_exchange_rate()
        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        # Multiplicamos las columnas de precio por el tipo de cambio
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = df[col] * rate
                
        return df, rate
    except Exception:
        return pd.DataFrame(), 1.0

data_load_state = st.empty()
data_load_state.text('Obteniendo datos y convirtiendo divisa...')
# Ahora la función devuelve DOS cosas: los datos y la tasa de cambio usada
data, rate_used = load_data(selected_stock, start_date_str)
data_load_state.empty()

# 4. Visualización
if data.empty:
    st.error(f"No hay datos para {selected_stock}.")
else:
    # Mostramos la tasa de conversión usada (Transparencia para el usuario)
    st.caption(f"ℹ️ Datos convertidos a Euros (Tasa aplicada: 1 USD = {rate_used:.4f} EUR)")

    # Métricas
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    variation = last_close - prev_close
    pct_variation = (variation / prev_close) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    # Fíjate que ahora usamos el símbolo €
    col1.metric(f"Precio ({selected_stock})", f"€{last_close:.2f}", f"{variation:.2f}€ ({pct_variation:.2f}%)")
    col2.metric("Volumen", f"{data['Volume'].iloc[-1]:,}")
    col3.metric("Máximo Anual", f"€{data['High'].tail(365).max():.2f}")
    col4.metric("Mínimo Anual", f"€{data['Low'].tail(365).min():.2f}")

    st.markdown("---")

    # Pestañas
    tab1, tab2, tab3 = st.tabs(["Histórico (EUR)", "Predicción IA (EUR)", "Datos"])

    with tab1:
        st.subheader(f"Evolución: {selected_stock}")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'], name="Cierre",
            line=dict(color='#0068C9', width=2), fill='tozeroy', fillcolor='rgba(0, 104, 201, 0.1)'
        ))
        fig_hist.update_layout(
            hovermode="x unified",
            yaxis_title="Precio (Euros €)" # Etiqueta correcta
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader(f"Proyección a {prediction_months} meses")
        df_train = data[['Date', 'Close']].copy().rename(columns={"Date": "ds", "Close": "y"})
        
        if len(df_train) < 20:
            st.warning("Datos insuficientes.")
        else:
            with st.spinner('La IA está calculando en Euros...'):
                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=prediction_months * 30)
                forecast = m.predict(future)
                
                fig_pred = go.Figure()
                last_real_date = data['Date'].max()
                future_only = forecast[forecast['ds'] > last_real_date]

                # 1. Línea Principal
                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat'], 
                    name="Predicción (Promedio)",
                    line=dict(color='#FF4B4B', width=4)
                ))

                # 2. Límite Superior
                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat_upper'],
                    name="Máximo Estimado",
                    mode='lines', 
                    line=dict(width=1, color='rgba(255, 75, 75, 0.5)', dash='dot'), 
                    showlegend=True
                ))

                # 3. Límite Inferior
                fig_pred.add_trace(go.Scatter(
                    x=future_only['ds'], y=future_only['yhat_lower'],
                    name="Mínimo Estimado",
                    mode='lines', 
                    line=dict(width=1, color='rgba(255, 75, 75, 0.5)', dash='dot'), 
                    fill='tonexty', 
                    fillcolor='rgba(255, 75, 75, 0.2)', 
                    showlegend=True
                ))

                fig_pred.update_layout(
                    title="Tendencia Esperada",
                    hovermode="x unified",
                    yaxis_title="Precio Estimado (Euros €)"
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                csv = future_only.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar CSV", csv, f'pred_{selected_stock}_EUR.csv', 'text/csv')

    with tab3:
        st.dataframe(data.sort_values('Date', ascending=False), use_container_width=True)

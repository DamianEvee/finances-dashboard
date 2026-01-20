"""
Dashboard Financiero v8.3 (Extreme Memory Edition)
-----------------------------------------
Autor: Evee_
Motor: TensorFlow / Keras (LSTM)
Mejoras:
1. Slider de Memoria ampliado a 10 AÑOS (3650 días).
2. Slider de Neuronas EXPLICITO (50-200).
3. Entrenamiento profundo desbloqueado.
4. Sin Emojis (Clean Interface).
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from plotly import graph_objs as go
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout

# 1. Configuración
st.set_page_config(
    page_title="AI Stock Vision (Deep Learning)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('AI Stock Vision: Deep Learning (LSTM)')

# --- CONFIGURACIÓN BASE DE DATOS ---
STOCK_DB = {
    "Favoritos": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
    "Tecnología": ["AMD", "INTC", "CRM", "ADBE", "ORCL", "IBM", "SAP", "SPOT"],
    "Finanzas": ["JPM", "BAC", "V", "MA", "GS", "MS", "PYPL", "AXP"],
    "Automotriz": ["F", "GM", "TM", "HMC", "RACE", "STLA"],
    "Salud": ["JNJ", "PFE", "MRNA", "LLY", "UNH", "ABBV"],
    "Consumo": ["WMT", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS"],
    "Criptomonedas": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "Búsqueda Manual": [] 
}

# 2. Sidebar
st.sidebar.header("Selección de Activo")
sector = st.sidebar.selectbox("Sector:", list(STOCK_DB.keys()))

if sector == "Búsqueda Manual":
    selected_stock = st.sidebar.text_input("Ticker:", "AAPL").upper()
else:
    selected_stock = st.sidebar.selectbox("Empresa:", STOCK_DB[sector])

st.sidebar.markdown("---")
st.sidebar.subheader("Control Total IA")

st.sidebar.info("Entrenando con historial completo disponible.")

# MODIFICADO: Memoria hasta 10 AÑOS (3650 días)
look_back = st.sidebar.slider('Memoria (Dias previos):', 30, 3650, 365, help="Cuantos dias del pasado mira la IA para predecir el futuro.")

# NUEVO: Control de Neuronas (Potencia)
neurons = st.sidebar.slider('Potencia (Neuronas LSTM):', 50, 200, 100, help="Mas neuronas = cerebro mas complejo.")

# MODIFICADO: Predicción
prediction_days = st.sidebar.slider('Dias a predecir:', 5, 45, 30)

# MODIFICADO: Mas Epochs
epochs = st.sidebar.slider('Epochs (Entrenamiento):', 1, 50, 15, help="Repeticiones de aprendizaje.")

# Verificar aceleración de hardware
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    st.sidebar.success(f"Aceleracion GPU Activada: {len(gpus)} dispositivo(s)")
else:
    st.sidebar.warning("Usando CPU (Modo Compatibilidad)")

# 3. Funciones de Carga
def get_exchange_rate():
    try:
        fx = yf.download("EUR=X", period="1d", progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        if fx.empty: return 1.0
        val = fx['Close'].iloc[-1]
        if hasattr(val, 'item'): return float(val.item())
        return float(val)
    except: return 1.0

def load_data(ticker):
    try:
        # Descargamos 15 años para asegurar tener suficiente data para look_back grandes
        start = (date.today() - timedelta(days=15*365)).strftime("%Y-%m-%d")
        end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return df, 1.0
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        df = df.sort_values('Date')
        
        # Conversión EUR
        rate = get_exchange_rate()
        cols = ['Open', 'High', 'Low', 'Close']
        for c in cols:
            if c in df.columns: df[c] = df[c] * rate
            
        return df, rate
    except Exception as e:
        return pd.DataFrame(), 1.0

# --- CEREBRO TENSORFLOW (LSTM DINAMICO) ---
def predict_lstm_tf(data, days_to_predict, look_back_window, num_epochs, num_neurons):
    # Preprocesamiento
    df_close = data.filter(['Close'])
    dataset = df_close.values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    train_len = len(scaled_data)
    
    # Validacion: Si la memoria es mayor que los datos disponibles, ajustamos
    if look_back_window >= train_len:
         look_back_window = int(train_len * 0.5)
    
    start_idx = look_back_window
    
    for i in range(start_idx, train_len):
        x_train.append(scaled_data[i-look_back_window:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # --- ARQUITECTURA DINAMICA ---
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    
    # 1. Capa LSTM (Potencia variable segun slider)
    model.add(LSTM(num_neurons, return_sequences=False))
    
    # 2. Capa Dropout (Realismo)
    model.add(Dropout(0.2))
    
    # 3. Capas Densas
    model.add(Dense(25)) 
    model.add(Dense(1))
    
    # Compilar
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=num_epochs, verbose=0)
    
    # Predicción Recursiva
    last_window_raw = scaled_data[-look_back_window:]
    current_batch = last_window_raw.reshape((1, look_back_window, 1))
    
    predicted_prices = []
    
    for i in range(days_to_predict):
        current_pred = model.predict(current_batch, verbose=0)[0]
        predicted_prices.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
        
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Generar fechas futuras
    last_date = data['Date'].max()
    future_dates = []
    final_prices = []
    
    current_date = last_date
    for price in predicted_prices:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            future_dates.append(current_date)
            final_prices.append(price[0])
            
    return pd.DataFrame({'Date': future_dates, 'Predicted_Close': final_prices})

# --- EJECUCIÓN ---
status = st.empty()
status.text('Descargando historial...')

data, rate = load_data(selected_stock)

forecast = pd.DataFrame()

if not data.empty:
    status.text(f'Entrenando LSTM ({neurons} Neuronas)... Epochs: {epochs}')
    bar = st.progress(0)
    
    try:
        # Pasamos la variable 'neurons' a la funcion
        forecast = predict_lstm_tf(data, prediction_days, look_back, epochs, neurons)
        bar.progress(100)
        status.empty()
    except Exception as e:
        st.error(f"Error TensorFlow: {e}")
        status.empty()

# --- VISUALIZACIÓN ---
if not data.empty:
    st.caption(f"Divisa: EUR (Tasa: {rate:.4f}) | Motor: TensorFlow LSTM")
    
    last_price = data['Close'].iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio Actual", f"EUR {last_price:.2f}")
    col2.metric("Volumen", f"{data['Volume'].iloc[-1]:,}")
    
    if not forecast.empty and len(forecast) > 0:
        pred_end = forecast['Predicted_Close'].iloc[-1]
        
        # Color dinámico
        trend_color = "normal"
        if pred_end > last_price:
            trend = "ALCISTA"
        else:
            trend = "BAJISTA"
            trend_color = "inverse"
            
        pct_change = ((pred_end - last_price) / last_price) * 100
        col3.metric("Tendencia IA", trend, f"{pct_change:.2f}%", delta_color=trend_color)

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Grafico Neuronal", "Datos"])
    
    with tab1:
        fig = go.Figure()
        
        # Historia Real
        subset_data = data.tail(365)
        fig.add_trace(go.Scatter(
            x=subset_data['Date'], y=subset_data['Close'],
            name="Historia Real",
            line=dict(color='#0068C9', width=2)
        ))
        
        # Predicción
        if not forecast.empty:
            last_real_point = pd.DataFrame({'Date': [data['Date'].max()], 'Predicted_Close': [last_price]})
            forecast_plot = pd.concat([last_real_point, forecast])
            
            fig.add_trace(go.Scatter(
                x=forecast_plot['Date'], y=forecast_plot['Predicted_Close'],
                name="Proyeccion LSTM",
                line=dict(color='#FF4B4B', width=3)
            ))

        fig.update_layout(
            title=f"Analisis Deep Learning: {selected_stock}",
            yaxis_title="Precio (EUR)",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if not forecast.empty:
            st.subheader("Datos Generados (LSTM)")
            st.dataframe(forecast, use_container_width=True)
            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV", csv, "prediccion_lstm.csv", "text/csv")

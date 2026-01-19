import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# 1. Configuración
st.set_page_config(page_title="Predictor Financiero AI", layout="wide")
st.title('📈 Dashboard Financiero con Predicción AI')

# 2. Sidebar
st.sidebar.header("Configuración")
selected_stock = st.sidebar.text_input("Símbolo (Ticker)", "AAPL") 

# Slider 1: Cuánto PASADO estudiar
n_years = st.sidebar.slider('Años de historia para entrenar:', 1, 5, 2)

# Slider 2: Cuánto FUTURO predecir 
prediction_months = st.sidebar.slider('Meses a predecir:', 1, 24, 12) # De 1 mes a 2 años

# 3. Calcular fecha de inicio dinámica
start_date = date.today() - timedelta(days=n_years*365)
start_date_str = start_date.strftime("%Y-%m-%d")

# 4. Carga de datos
@st.cache_data
def load_data(ticker, start):
    df = yf.download(ticker, start=start, end=date.today().strftime("%Y-%m-%d"))
    
    if df.empty:
        return df
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
    return df

data_load_state = st.text('Cargando datos...')
data = load_data(selected_stock, start_date_str)
data_load_state.text('¡Datos cargados!')

# 5. Lógica Principal
if data.empty:
    st.error(f"⚠️ No se encontraron datos para '{selected_stock}'.")
else:
    st.subheader(f'Datos Históricos de {selected_stock}')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Apertura"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Cierre"))
        fig.layout.update(title_text=f'Evolución: {selected_stock}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Predicción Dinámica
    st.subheader(f'🔮 Predicción de Precio a {prediction_months} meses') # Título dinámico
    
    df_train = data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    if len(df_train) < 20:
        st.warning("⚠️ Necesitas más datos históricos para predecir.")
    else:
        with st.spinner('Calculando futuro...'):
            m = Prophet()
            m.fit(df_train)
            
            future = m.make_future_dataframe(periods=prediction_months * 30) 
            forecast = m.predict(future)

            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

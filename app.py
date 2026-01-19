import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Financiero AI", layout="wide")
st.title('📈 Dashboard Financiero con Predicción AI')

# 2. Sidebar
st.sidebar.header("Configuración")
selected_stock = st.sidebar.text_input("Símbolo (Ticker)", "AAPL") 
n_years = st.sidebar.slider('Años de datos históricos:', 1, 5, 2)
period = n_years * 365

# 3. Carga de datos (CORREGIDA)
@st.cache_data
def load_data(ticker):
    # Descargamos los datos
    df = yf.download(ticker, start="2018-01-01", end=date.today().strftime("%Y-%m-%d"))
    
    # Si los datos vienen vacíos, retornamos inmediato
    if df.empty:
        return df
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
    return df

data_load_state = st.text('Cargando datos...')
data = load_data(selected_stock)
data_load_state.text('¡Datos cargados!')

# 4. Lógica Principal
if data.empty:
    st.error(f"⚠️ No se encontraron datos para '{selected_stock}'. Revisa que el ticker sea correcto.")
else:
    # Mostrar tabla de datos reciente
    st.subheader(f'Datos Históricos de {selected_stock}')
    st.write(data.tail())

    # Gráfico interactivo
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Apertura"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Cierre"))
        fig.layout.update(title_text=f'Evolución del precio: {selected_stock}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # 5. Predicción con Prophet
    st.subheader(f'🔮 Predicción de Precio a 1 año')
    
    # Preparar datos para Prophet
    df_train = data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Verificar cantidad de datos
    if len(df_train) < 20:
        st.warning("⚠️ No hay suficientes datos para predecir.")
    else:
        with st.spinner('Entrenando la IA...'):
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)

            # Mostrar datos de predicción
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            
            # Gráfico de predicción
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            st.success("✅ Predicción completada")

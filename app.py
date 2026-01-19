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
n_years = st.sidebar.slider('Años de datos históricos:', 1, 5, 2)

# 3. Calcular fecha de inicio dinámica
# Restamos a la fecha de hoy los años seleccionados por el usuario
start_date = date.today() - timedelta(days=n_years*365)
start_date_str = start_date.strftime("%Y-%m-%d")

# 4. Carga de datos
@st.cache_data
def load_data(ticker, start):
    df = yf.download(ticker, start=start, end=date.today().strftime("%Y-%m-%d"))
    
    if df.empty:
        return df
        
    # Aplanamos MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    
    # Quitamos zona horaria
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
    return df

data_load_state = st.text('Cargando datos...')
# Pasamos el ticker Y la fecha calculada
data = load_data(selected_stock, start_date_str)
data_load_state.text('¡Datos cargados!')

# 5. Lógica Principal
if data.empty:
    st.error(f"⚠️ No se encontraron datos para '{selected_stock}'.")
else:
    # Mostrar datos
    st.subheader(f'Datos Históricos de {selected_stock}')
    st.write(f"Mostrando datos desde: **{start_date_str}**") # Confirmación visual
    st.write(data.tail())

    # Gráfico
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Apertura"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Cierre"))
        fig.layout.update(title_text=f'Evolución: {selected_stock}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Predicción
    st.subheader(f'🔮 Predicción de Precio a 1 año')
    
    df_train = data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    if len(df_train) < 20:
        st.warning("⚠️ Necesitas más de 1 año de datos para que la IA funcione bien. Aumenta los años en la barra lateral.")
    else:
        with st.spinner('Entrenando la IA...'):
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)

            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

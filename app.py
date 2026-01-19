import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Financiero AI", layout="wide")
st.title('📈 Dashboard Financiero con Predicción AI')

# 2. Sidebar para inputs del usuario
st.sidebar.header("Configuración")
selected_stock = st.sidebar.text_input("Símbolo (Ticker)", "AAPL") 
n_years = st.sidebar.slider('Años de datos históricos:', 1, 5, 2)
period = n_years * 365

# 3. Función para cargar datos (con Cache para que sea rápido)
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2018-01-01", end=date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Cargando datos...')
data = load_data(selected_stock)
data_load_state.text('¡Datos cargados con éxito!')

# 4. Mostrar datos crudos y gráfico simple
st.subheader(f'Datos Históricos de {selected_stock}')
st.write(data.tail()) # Muestra las últimas 5 filas

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="precio_apertura"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="precio_cierre"))
    fig.layout.update(title_text=f'Línea de Tiempo: {selected_stock}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# 5. EL FACTOR INNOVADOR: Predicción con Prophet
st.subheader(f'🔮 Predicción de Precio a 1 año')
st.write("Entrenando modelo de IA (esto puede tardar unos segundos)...")

# Preparar datos para Prophet (requiere columnas 'ds' y 'y')
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Entrenar modelo
m = Prophet()
m.fit(df_train)

# Crear futuro
future = m.make_future_dataframe(periods=365) # Predicción a 1 año
forecast = m.predict(future)

# 6. Visualizar Predicción
st.write(forecast.tail())

# Gráfico interactivo de la predicción
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Nota: Las áreas sombreadas representan el intervalo de confianza (incertidumbre).")

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Financiero AI", layout="wide")
st.title('📈 Dashboard Financiero con Predicción AI')

# 2. Sidebar
st.sidebar.header("Configuración")
selected_stock = st.sidebar.text_input("Símbolo (Ticker)", "AAPL") 
n_years = st.sidebar.slider('Años de datos históricos:', 1, 5, 2)
period = n_years * 365

# 3. Carga de datos
@st.cache_data
def load_data(ticker):
    # Bajamos datos desde 2015 para tener suficiente historia
    data = yf.download(ticker, start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Cargando datos...')
data = load_data(selected_stock)
data_load_state.text('¡Datos cargados!')

# --- NUEVO: PROTECCIÓN CONTRA ERRORES ---
if data.empty:
    st.error(f"No se encontraron datos para el símbolo '{selected_stock}'. Por favor verifica que sea correcto (ej: AAPL, TSLA, BTC-USD).")
else:
    
    # 4. Mostrar datos crudos
    st.subheader(f'Datos Históricos de {selected_stock}')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Apertura"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Cierre"))
        fig.layout.update(title_text=f'Línea de Tiempo: {selected_stock}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # 5. Predicción con Prophet
    st.subheader(f'🔮 Predicción de Precio a 1 año')
    
    # Preparamos datos: Prophet necesita columnas exactas 'ds' y 'y'
    df_train = data[['Date', 'Close']].copy() # Usamos .copy() para evitar advertencias
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Verificación de tener suficientes datos para entrenar el modelo
    if len(df_train) < 20:
        st.warning("⚠️ No hay suficientes datos históricos para hacer una predicción fiable.")
    else:
        st.write("Entrenando modelo de IA...")
        m = Prophet()
        m.fit(df_train)
        
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        st.write(forecast.tail())
        
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.success("✅ Predicción completada. El área sombreada es el margen de error.")

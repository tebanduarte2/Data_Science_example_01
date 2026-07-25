import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Ambiental con IA Llama 3.3",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dashboard de Monitoreo Ambiental & Asistente IA")
st.markdown("Análisis interactivo de calidad del aire, ruido y meteorología integrado con **Llama 3.3 70B (Groq)**.")

# ---------------------------------------------------------
# Panel Lateral: Carga de API Key y Filtros
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración & API Key")

api_key = st.sidebar.text_input(
    "Ingresa tu Groq API Key:",
    type="password",
    help="Necesaria para la interpretación con IA y el Chatbot."
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtros de Datos")

# ---------------------------------------------------------
# 1. Carga de Datos y Conversión a DataFrame
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('monitoreo_ambiental.csv')
    df['Hora_DT'] = pd.to_datetime(df['Hora_Lectura'], format='%H:%M')
    df['Hora_Num'] = df['Hora_DT'].dt.hour
    return df

try:
    df = load_data()
    st.sidebar.success(" Archivo 'monitoreo_ambiental.csv' cargado")
except Exception as e:
    st.error(f"Error al cargar 'monitoreo_ambiental.csv': {e}")
    st.stop()

# Filtros Interactivos
ciudades_sel = st.sidebar.multiselect(
    "Seleccionar Ciudad(es):",
    options=df['Ciudad'].unique(),
    default=df['Ciudad'].unique()
)

zonas_sel = st.sidebar.multiselect(
    "Seleccionar Tipo de Zona:",
    options=df['Tipo_Zona'].unique(),
    default=df['Tipo_Zona'].unique()
)

df_filtered = df[(df['Ciudad'].isin(ciudades_sel)) & (df['Tipo_Zona'].isin(zonas_sel))]

# ---------------------------------------------------------
# Construcción del Contexto para la IA
# ---------------------------------------------------------
def build_data_context(data: pd.DataFrame) -> str:
    if data.empty:
        return "No hay datos seleccionados en los filtros actuales."
    
    total_registros = len(data)
    pm_mean = data['PM2_5_Ug_m3'].mean()
    pm_max = data['PM2_5_Ug_m3'].max()
    ruido_mean = data['Nivel_Ruido_dB'].mean()
    ruido_max = data['Nivel_Ruido_dB'].max()
    temp_mean = data['Temperatura_C'].mean()
    humedad_mean = data['Humedad_Relativa_Pct'].mean()
    lluvia_pct = data['Presencia_Lluvia'].mean() * 100
    
    # Resumen por zona
    zona_summary = data.groupby('Tipo_Zona')['PM2_5_Ug_m3'].agg(['mean', 'max']).to_dict(orient='index')
    
    # Resumen por horario en zonas residenciales
    res_data = data[data['Tipo_Zona'] == 'Residencial']
    res_horarios = ""
    if not res_data.empty:
        top_res_hours = res_data.groupby('Hora_Num')['PM2_5_Ug_m3'].mean().nlargest(3).to_dict()
        res_horarios = f"Horas residenciales con mayor PM2.5 promedio (Hora: PM2.5): {top_res_hours}"

    context = f"""
    --- RESUMEN DE DATOS AMBIENTALES ACTUALES ---
    - Registros Analizados: {total_registros}
    - Ciudades Filtro: {list(data['Ciudad'].unique())}
    - Tipos de Zona Filtro: {list(data['Tipo_Zona'].unique())}
    - PM2.5 Promedio: {pm_mean:.2f} µg/m³ (Máximo: {pm_max:.2f} µg/m³)
    - Ruido Promedio: {ruido_mean:.2f} dB (Máximo: {ruido_max:.2f} dB)
    - Temperatura Promedio: {temp_mean:.2f} °C
    - Humedad Promedio: {humedad_mean:.2f} %
    - Porcentaje de Días/Tomas con Lluvia: {lluvia_pct:.1f}%
    - Detalle de PM2.5 Promedio y Máximo por Zona: {zona_summary}
    - {res_horarios}
    -------------------------------------------
    """
    return context

# ---------------------------------------------------------
# Tabs Principales
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA",
    "📈 Visualizaciones Key",
    "🎨 Gráficas Seaborn",
    "📝 Reportes",
    "🤖 Interpretación IA",
    "💬 Chat con IA (Contexto)"
])

# ---------------------------------------------------------
# Tab 1: EDA
# ---------------------------------------------------------
with tab1:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Registro de Lecturas", len(df_filtered))
    col_m2.metric("Promedio PM2.5 (µg/m³)", f"{df_filtered['PM2_5_Ug_m3'].mean():.2f}")
    col_m3.metric("Promedio Ruido (dB)", f"{df_filtered['Nivel_Ruido_dB'].mean():.2f}")
    col_m4.metric("Tomas con Lluvia (%)", f"{(df_filtered['Presencia_Lluvia'].mean()*100):.1f}%")
    
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df_filtered.head(10), use_container_width=True)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(df_filtered.describe(), use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Visualización Clave (Scatterplot)
# ---------------------------------------------------------
with tab2:
    st.header("Visualización Clave: Scatterplot PM2.5 vs Hora")
    st.write("**Pregunta de Negocio:** ¿Existen zonas horarias críticas para la salud en zonas residenciales?")
    
    fig_scatter = px.scatter(
        df_filtered,
        x='Hora_Lectura',
        y='PM2_5_Ug_m3',
        color='Tipo_Zona',
        hover_data=['Ciudad', 'Nivel_Ruido_dB', 'Indice_Calidad_Aire_ICA'],
        title="Partículas PM2.5 vs Hora de Lectura por Tipo de Zona",
        labels={'PM2_5_Ug_m3': 'PM2.5 (µg/m³)', 'Hora_Lectura': 'Hora del Día'},
        category_orders={"Hora_Lectura": sorted(df_filtered['Hora_Lectura'].unique())}
    )
    fig_scatter.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Límite Moderado (50 µg/m³)")
    fig_scatter.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Límite Dañino (100 µg/m³)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Seaborn & Pyplot
# ---------------------------------------------------------
with tab3:
    st.header("Análisis Detallado con Seaborn y Matplotlib")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Distribución de Ruido (dB) por Tipo de Zona")
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_filtered, x='Tipo_Zona', y='Nivel_Ruido_dB', ax=ax_box, palette='Set2')
        plt.xticks(rotation=45)
        plt.title("Nivel de Ruido por Zona")
        st.pyplot(fig_box)
        
    with col_g2:
        st.subheader("Matriz de Correlación de Variables")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        corr_cols = ['PM2_5_Ug_m3', 'Temperatura_C', 'Humedad_Relativa_Pct', 'Nivel_Ruido_dB']
        sns.heatmap(df_filtered[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        plt.title("Correlación de Variables Ambientales")
        st.pyplot(fig_corr)

# ---------------------------------------------------------
# Tab 4: Reportes Tradicionales
# ---------------------------------------------------------
with tab4:
    st.header("📝 Reporte Estadístico de Hallazgos")
    res_data = df_filtered[df_filtered['Tipo_Zona'] == 'Residencial']
    pm_critico = res_data[res_data['PM2_5_Ug_m3'] > 100]
    
    st.subheader("1. Conclusión de Zonas Horarias Críticas")
    st.markdown(f"""
    - **Total de lecturas críticas (>100 µg/m³) en Residencial:** {len(pm_critico)}
    - **Horarios con picos de contaminación:** Se observan concentraciones elevadas entre **11:00 AM - 1:00 PM** y **5:00 PM - 8:00 PM**.
    - **Factor Ruido:** El nivel medio de ruido registra promedios cercanos a **70-75 dB**.
    """)

# ---------------------------------------------------------
# Tab 5: Interpretación y Resumen Generado por Llama 3.3
# ---------------------------------------------------------
with tab5:
    st.header("🤖 Resumen & Interpretación Ejecutiva por IA")
    st.markdown("Genera un análisis sintético del estado ambiental usando el modelo **Llama 3.3 70B**.")
    
    if st.button("✨ Generar Diagnóstico Ambiental con IA"):
        if not api_key:
            st.error("⚠️ Por favor ingresa tu Groq API Key en la barra lateral.")
        else:
            with st.spinner("Llama 3.3 70B analizando los datos..."):
                try:
                    client = Groq(api_key=api_key)
                    data_summary_text = build_data_context(df_filtered)
                    
                    prompt_summary = f"""
                    Eres un consultor experto en gestión ambiental y salud pública.
                    Analiza la siguiente información resumida del dataset de monitoreo ambiental y responde:
                    
                    1. ¿Cuál es el diagnóstico general de la calidad del aire y los niveles de ruido?
                    2. ¿Qué zonas o franjas horarias representan un riesgo crítico para la salud?
                    3. Tres recomendaciones estratégicas para mitigar la contaminación.
                    
                    {data_summary_text}
                    """
                    
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt_summary}],
                        temperature=0.4,
                        max_tokens=1500
                    )
                    
                    st.success("Diagnóstico Generado Exitosamente:")
                    st.markdown(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error al generar interpretación con la API: {e}")

# ---------------------------------------------------------
# Tab 6: Chatbot Conversacional con Contexto
# ---------------------------------------------------------
with tab6:
    st.header("💬 Chatbot Ambiental con Contexto de Datos")
    st.caption("Pregúntale al modelo Llama 3.3 sobre los datos de contaminación, ruido, horas pico o recomendaciones.")
    
    if "env_messages" not in st.session_state:
        st.session_state.env_messages = []
        
    for msg in st.session_state.env_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_query = st.chat_input("Escribe tu pregunta sobre los datos de monitoreo ambiental...")
    
    if user_query:
        if not api_key:
            st.error("⚠️ Ingrese su Groq API Key en el menú lateral para chatear.")
            st.stop()
            
        st.session_state.env_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                client = Groq(api_key=api_key)
                
                # Construir Prompt de Sistema inyectando el contexto de los datos actuales
                current_context = build_data_context(df_filtered)
                system_instruction = {
                    "role": "system",
                    "content": (
                        "Eres un asistente virtual especializado en análisis ambiental y ciencia de datos. "
                        "Tienes acceso al siguiente resumen consolidado de las mediciones de monitoreo ambiental cargadas por el usuario:\n"
                        f"{current_context}\n"
                        "Responde de forma clara, profesional, fundamentando tus respuestas en estos datos."
                    )
                }
                
                chat_history = [system_instruction] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.env_messages
                ]
                
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=chat_history,
                    temperature=0.5,
                    max_tokens=1500,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                        
                message_placeholder.markdown(full_response)
                st.session_state.env_messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error de conexión con la API de Groq: {e}")

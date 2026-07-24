import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Monitoreo Ambiental",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dashboard de Monitoreo Ambiental")
st.markdown("Análisis de calidad del aire, ruido y variables meteorológicas.")

# ---------------------------------------------------------
# 1. Carga de Datos y Conversión a DataFrame
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('monitoreo_ambiental.csv')
    # Convertir hora a formato DateTime para ordenación correcta
    df['Hora_DT'] = pd.to_datetime(df['Hora_Lectura'], format='%H:%M')
    df['Hora_Num'] = df['Hora_DT'].dt.hour
    return df

try:
    df = load_data()
    st.sidebar.success(" Archivo de datos cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el archivo 'monitoreo_ambiental.csv': {e}")
    st.stop()

# Filtros Interactivos en Sidebar
st.sidebar.header("Filtros Globales")
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
# Tabs Principales
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA (Análisis Exploratorio)",
    "📈 Visualizaciones Key",
    "🎨 Gráficas Seaborn & Matplotlib",
    "📝 Reportes y Conclusiones"
])

# ---------------------------------------------------------
# 2. EDA (Análisis Exploratorio de Datos)
# ---------------------------------------------------------
with tab1:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Registro de Lecturas", len(df_filtered))
    col_m2.metric("Promedio PM2.5 (µg/m³)", f"{df_filtered['PM2_5_Ug_m3'].mean():.2f}")
    col_m3.metric("Promedio Ruido (dB)", f"{df_filtered['Nivel_Ruido_dB'].mean():.2f}")
    col_m4.metric("Días con Lluvia (%)", f"{(df_filtered['Presencia_Lluvia'].mean()*100):.1f}%")
    
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df_filtered.head(10), use_container_width=True)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(df_filtered.describe(), use_container_width=True)

# ---------------------------------------------------------
# 3. Visualizaciones Interactivas (Plotly)
# ---------------------------------------------------------
with tab2:
    st.header("Visualización Clave: Scatterplot PM2.5 vs Hora")
    st.write("**Pregunta de Negocio:** ¿Existen zonas horarias críticas para la salud en zonas residenciales?")
    
    # Scatterplot Plotly (Visualización Clave requerida)
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
# 3b. Gráficas Seaborn y Pyplot
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
# 4. Generar Reportes
# ---------------------------------------------------------
with tab4:
    st.header("📝 Reporte de Hallazgos y Recomendaciones")
    
    res_data = df_filtered[df_filtered['Tipo_Zona'] == 'Residencial']
    pm_critico = res_data[res_data['PM2_5_Ug_m3'] > 100]
    
    st.subheader("1. Conclusión de Zonas Horarias Críticas")
    st.markdown(f"""
    - **Total de lecturas críticas (>100 µg/m³) en Residencial:** {len(pm_critico)}
    - **Horarios con picos de contaminación:** Se observan concentraciones elevadas entre **11:00 AM - 1:00 PM** y **5:00 PM - 8:00 PM**.
    - **Factor Ruido:** El nivel medio de ruido en zonas residenciales registra promedios cercanos a **70-75 dB**, superando los límites recomendados de confort acústico urbano.
    """)
    
    st.subheader("2. Recomendaciones de Salud Pública")
    st.warning("""
    * **Restricción Ventilación:** Se sugiere evitar ventilar los hogares en zonas residenciales durante las franjas de 11:00 AM a 1:00 PM y de 5:00 PM a 8:00 PM.
    * **Actividad Física:** Desaconsejar el ejercicio al aire libre en horarios de mayor tráfico e intensidad ambiental en áreas residenciales.
    """)

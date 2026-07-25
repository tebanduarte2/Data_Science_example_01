import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Text-to-Data & EDA con Llama 3.3",
    page_icon="📑",
    layout="wide"
)

st.title("📑 Extractor de Datos no Estructurados & EDA con IA")
st.markdown("Convierte párrafos con cifras en tablas estructuradas y realiza un Análisis Exploratorio automáticamente usando **Llama 3.3 70B**.")

# ---------------------------------------------------------
# Sidebar: API Key
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración")
api_key = st.sidebar.text_input(
    "Ingresa tu Groq API Key:",
    type="password",
    help="Necesaria para procesar el texto con Llama 3.3 70B."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Puedes pegar reportes financieros, métricas de producción, resúmenes de ventas o cualquier texto narrativo que contenga datos numéricos.")

# ---------------------------------------------------------
# Área de Entrada de Texto
# ---------------------------------------------------------
st.subheader("1. Ingrese el texto con información numérica")

default_text = (
    "En el primer trimestre de 2024, la división de Tecnología registró ingresos por 45.5 millones de dólares "
    "con un costo operativo de 28.2 millones y un total de 120 empleados. Por su parte, la división de Salud reportó "
    "ingresos de 62.1 millones, costos de 35.0 millones y 210 empleados. La división de Educación alcanzó 18.3 millones en ingresos, "
    "con costos de 12.1 millones y 65 empleados. Finalmente, la división de Logística generó 31.8 millones en ingresos, "
    "costos por 22.4 millones y cuenta con 95 empleados."
)

user_text = st.text_area(
    "Párrafo de entrada:",
    value=default_text,
    height=150
)

# ---------------------------------------------------------
# Función para Extracción de Datos con LLM
# ---------------------------------------------------------
def extract_data_from_text(text: str, client: Groq) -> pd.DataFrame:
    system_prompt = (
        "Eres un experto extractor de datos no estructurados. Tu tarea es analizar el párrafo ingresado por el usuario "
        "y extraer TODAS las entidades, métricas y cifras numéricas en una lista de objetos JSON.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Devuelve ÚNICAMENTE un arreglo JSON válido (ej: [{\"columna1\": valor1, \"columna2\": valor2}]).\n"
        "2. No agregues explicaciones, ni etiquetas de markdown tipo ```json, solo el texto en JSON puro.\n"
        "3. Convierte todos los números a valores numéricos (float o int), eliminando símbolos como $, %, comas de miles, etc.\n"
        "4. Asigna nombres claros y consistentes a las columnas extraídas."
    )
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    
    raw_content = response.choices[0].message.content.strip()
    
    # Limpieza por si el modelo incluye comillas triples o formato markdown
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
    raw_content = raw_content.strip()
    
    data_json = json.loads(raw_content)
    df = pd.DataFrame(data_json)
    return df

# ---------------------------------------------------------
# Procesamiento
# ---------------------------------------------------------
if st.button("🚀 Extraer Datos y Procesar EDA", type="primary"):
    if not api_key:
        st.error("⚠️ Por favor ingresa tu Groq API Key en el menú lateral.")
    elif not user_text.strip():
        st.warning("⚠️ Escribe o pega un párrafo de texto para analizar.")
    else:
        with st.spinner("Procesando texto con Llama 3.3 70B y convirtiendo a tabla..."):
            try:
                client = Groq(api_key=api_key)
                df_extracted = extract_data_from_text(user_text, client)
                st.session_state["df_extracted"] = df_extracted
                st.success("✅ Extracción completada con éxito.")
            except json.JSONDecodeError as e:
                st.error(f"Error al parsear el JSON generado por el modelo: {e}")
            except Exception as e:
                st.error(f"Ocurrió un error en la API: {e}")

# ---------------------------------------------------------
# Visualización y EDA (Si hay datos extraídos)
# ---------------------------------------------------------
if "df_extracted" in st.session_state and st.session_state["df_extracted"] is not None:
    df = st.session_state["df_extracted"]
    
    st.markdown("---")
    st.header("2. Tabla de Datos Extraída")
    
    # Mostrar DataFrame e permitir edición interactiva si el usuario lo desea
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
    
    # Identificar columnas numéricas y categóricas
    num_cols = edited_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = edited_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.markdown("---")
    st.header("3. Análisis Exploratorio de Datos (EDA)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Métricas & Estadísticas", "📈 Gráficos Interactivos", "🤖 Conclusiones por IA"])
    
    # Tab 1: Resumen Estadístico
    with tab1:
        st.subheader("Resumen Estadístico Numérico")
        if num_cols:
            st.dataframe(edited_df[num_cols].describe().T, use_container_width=True)
            
            # Métricas rápidas (KPIs)
            st.subheader("KPIs Principales")
            cols = st.columns(min(len(num_cols), 4))
            for idx, col_name in enumerate(num_cols[:4]):
                with cols[idx]:
                    st.metric(
                        label=f"Total {col_name}",
                        value=f"{edited_df[col_name].sum():,.2f}",
                        delta=f"Prom: {edited_df[col_name].mean():,.2f}"
                    )
        else:
            st.info("No se detectaron columnas numéricas para el resumen estadístico.")
            
    # Tab 2: Gráficos
    with tab2:
        st.subheader("Visualización de Datos Extraídos")
        if num_cols and cat_cols:
            col_left, col_right = st.columns(2)
            
            with col_left:
                selected_cat = st.selectbox("Selecciona Eje X (Categoría):", cat_cols, key="cat_x")
                selected_num = st.selectbox("Selecciona Eje Y (Métrica):", num_cols, key="num_y")
                
                fig_bar = px.bar(
                    edited_df,
                    x=selected_cat,
                    y=selected_num,
                    color=selected_cat,
                    title=f"{selected_num} por {selected_cat}",
                    text_auto='.2s'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_right:
                if len(num_cols) >= 2:
                    selected_num2 = st.selectbox(
                        "Selecciona Métrica Secundaria para Scatterplot:",
                        [c for c in num_cols if c != selected_num],
                        key="num_y2"
                    )
                    fig_scatter = px.scatter(
                        edited_df,
                        x=selected_num,
                        y=selected_num2,
                        color=cat_cols[0],
                        size=selected_num,
                        title=f"{selected_num} vs {selected_num2}"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    fig_pie = px.pie(
                        edited_df,
                        names=selected_cat,
                        values=selected_num,
                        title=f"Distribución Porcentual de {selected_num}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        elif len(num_cols) >= 2:
            fig_scatter = px.scatter(
                edited_df,
                x=num_cols[0],
                y=num_cols[1],
                title=f"Relación entre {num_cols[0]} y {num_cols[1]}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Se requieren al menos 1 columna categórica y 1 numérica para generar gráficos interactivos automáticamente.")

    # Tab 3: Interpretación Final por la IA
    with tab3:
        st.subheader("💡 Interpretación de los Datos Extraídos")
        if st.button("Generar Diagnóstico Narrativo con Llama 3.3"):
            if not api_key:
                st.error("⚠️ Ingresa tu API Key de Groq en la barra lateral.")
            else:
                with st.spinner("Analizando la tabla con Llama 3.3 70B..."):
                    try:
                        client = Groq(api_key=api_key)
                        prompt_analysis = f"""
                        Eres un analista de datos sénior.
                        Basándote en la siguiente tabla extraída de un texto:
                        
                        {edited_df.to_markdown(index=False)}
                        
                        Proporciona:
                        1. Un resumen de los hallazgos numéricos clave.
                        2. Identificación de la categoría/elemento con mayor y menor desempeño o valor.
                        3. Tres conclusiones o recomendaciones operativas basadas en los datos.
                        """
                        
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt_analysis}],
                            temperature=0.3,
                            max_tokens=1000
                        )
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error al consultar la API: {e}")

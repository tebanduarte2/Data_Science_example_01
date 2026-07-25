import streamlit as st
from groq import Groq

# ---------------------------------------------------------
# Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bot de Historia Mundial y Cultura General",
    page_icon="🏛️",
    layout="centered"
)

st.title("🏛️ Bot de Historia Mundial & Cultura General")
st.caption("Especialista en eventos históricos, civilizaciones, arte y conocimiento general impulsado por Llama 3.3 70B.")

# ---------------------------------------------------------
# Panel Lateral (Sidebar) - Configuración y API Key
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

# Campo para ingresar la API Key en el Dashboard
api_key = st.sidebar.text_input(
    "Ingresa tu Groq API Key:",
    type="password",
    help="Puedes obtener tu API Key en la consola de Groq (https://console.groq.com/)."
)

# Control de temperatura para la creatividad de las respuestas
temperatura = st.sidebar.slider(
    "Creatividad (Temperatura):",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Valores más bajos dan respuestas más precisas y objetivas; valores más altos, más creativas."
)

if st.sidebar.button("🧹 Limpiar conversación"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------
# Inicialización de la Memoria de Chat
# ---------------------------------------------------------
# Prompt de sistema especializado en Historia y Cultura General
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Eres un historiador experto, erudito y divulgador cultural apasionado. "
        "Tu objetivo es responder preguntas sobre historia mundial, civilizaciones antiguas, "
        "acontecimientos geopolíticos, arte, literatura y cultura general de forma rigurosa, "
        "clara, objetiva y entretenida. Si la pregunta no está relacionada con historia o cultura general, "
        "responde amablemente y redirige la conversación al ámbito histórico o cultural."
    )
}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de la conversación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# Interacción con el Modelo Llama 3.3 70B
# ---------------------------------------------------------
prompt = st.chat_input("Haz una pregunta sobre historia o cultura general...")

if prompt:
    # 1. Validar si el usuario ingresó la API Key
    if not api_key:
        st.error("⚠️ Por favor ingresa tu API Key en la barra lateral para continuar.")
        st.stop()

    # 2. Mostrar la pregunta del usuario en la interfaz
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generar la respuesta llamando a la API con Llama 3.3 70B
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Inicializar cliente de Groq con la clave proporcionada
            client = Groq(api_key=api_key)
            
            # Construir el contexto completo (System Prompt + Historial)
            full_messages = [SYSTEM_PROMPT] + [
                {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]

            # Llamada en modo Streaming
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=temperatura,
                max_tokens=2048,
                stream=True
            )

            # Transmitir la respuesta texto a texto
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Guardar la respuesta del bot en la memoria
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Ocurrió un error al conectar con la API: {e}")

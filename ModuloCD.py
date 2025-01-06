import streamlit as st 
def generar_portada():
    st.empty()
    st.image("Data/logocg.jpg", width =100)
    st.markdown("<h1 style='text-align: center;'>PROYECTO FINAL</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:red;'>Maestria en Inteligencia Artificial y Ciencia de Datos</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue;'>Autor: Patricio Navarrete</h2>", unsafe_allow_html=True)
    st.caption('Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos C2024 P1')
    with st.expander("Ver detalles del Proyecto"):
        texto_largo = """
        La aplicación creada en un ambiente visual atractivo para el usuario, permite a los usuarios cargar datasets, explorar, 
        realizar EDA y visualizar resultados.
        También permite aplicar a un dataset predefinido, métodos de regresión para la predicción de datos, donde se visualiza los 
        principales hallazgos a través de las métricas de evaluación y también de manera gráfica para su mejor interpretación.  
        Al final se puede exportar a formato PDF esos resultados.
        """
        st.write(texto_largo)

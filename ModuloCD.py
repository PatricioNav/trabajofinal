import streamlit as st 
def generar_portada():
    st.empty()
    st.image("Data/logocg.jpg", width =100)
    st.markdown("<h1 style='text-align: center;'>PROYECTO FINAL</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:red;'>Maestria en Inteligencia Artificial y Ciencia de Datos</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:blue;'>Autor: Patricio Navarrete</h2>", unsafe_allow_html=True)
    st.caption('Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos C2024 P1')
    with st.expander("Ver detalles del Proyecto"):
        st.write("Este proyecto se divide en dos partes: Aquí puedes añadir información técnica, métricas adicionales o gráficos secundarios.")
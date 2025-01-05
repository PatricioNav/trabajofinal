# trabajo con streamlit
# Autor: Patricio Navarrete Freire
import streamlit as st
import pandas as pd
import ModuloCD as m1
import ModuloCarga as m2
import ModuloML as m3

# ************************ Modulos
app_mode = st.sidebar.selectbox('Selecciona Proceso',['Presentación','Cargar_Datos_EDA','Regresiones'])
if app_mode=="Presentación":
    m1.generar_portada()   
elif app_mode=='Cargar_Datos_EDA': 
    m2.cargar_datos()
elif app_mode=='Regresiones':
    m3.generar_modelos()
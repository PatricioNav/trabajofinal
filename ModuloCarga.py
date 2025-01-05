import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
def cargar_datos():
    st.empty()
   
    def generar_estadisticas(df):
        """
        Genera Estadísticas descriptivas de un dataframe
         """
        #st.write(df.columns)
         
        col1, col2 = st.columns(2)
        

        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
        if numeric_columns:
            with col1:
                row_count = df.shape[0]
                column_count = df.shape[1]
                st.write("Datos generales:")
                table_format = f"""
                | Descripción | Valor| 
                |---|---|
                | Número de Filas   | {row_count} |
                | Número de columnas | {column_count} |
                """
                st.markdown(table_format)  
                st.subheader("Seleccionar columna para análisis")
                column = st.selectbox("Selecciona una columna numérica:", numeric_columns)
                st.write(f"Estadísticas descriptivas para la columna: {column}")
                st.write(df[column].describe())
            with col2:
                st.subheader(f"Histograma de la columna: {column}")
            # Crear el histograma usando matplotlib
                fig, ax = plt.subplots()
                ax.hist(df[column], bins=20, color="skyblue", edgecolor="black")
                ax.set_title(f"Histograma de {column}")
                ax.set_xlabel("Valores")
                ax.set_ylabel("Frecuencia")
            # Mostrar el gráfico en Streamlit
                st.pyplot(fig)
        else:
            st.warning("El dataset no contiene columnas numéricas.")
        return df.describe()

    def exportar_excel(df):
        """
        Exporta un dataframe a un archivo excel
        """
        with open("estadisticas_descriptivas.xlsx", "wb") as archivo:
            df.to_excel(archivo, index=False, sheet_name = "Estadísticas")
    
        return "estadisticas_descriptivas.xlsx"

    st.title("Cargar Archivos y explorar Datos en Archivos Excel o CSV")
    st.divider()

    archivo_subido = st.file_uploader("Sube tu archivo excel o csv", type = ["xlsx", "xls", "csv"])

    if archivo_subido is not None:

        st.write("El archivo ha sido cargado")

        if archivo_subido.name.endswith("csv"):
            df = pd.read_csv(archivo_subido)

        else:
            df = pd.read_excel(archivo_subido)

        st.write(" ### DataFrame Original")
        st.dataframe(df)
        st.divider()
        df_estadisticas = generar_estadisticas(df)
        #st.dataframe(df_estadisticas)
       
        ruta_archivo = exportar_excel(df_estadisticas)

        with open(ruta_archivo, "rb") as archivo:

            st.download_button (
                label= "Descargar Resultados en Excel",
                data= archivo,
                file_name = "estadisticas_descriptivas.xlsx",
                mime= "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else: 
        st.write("Por favor cargar el archivo")

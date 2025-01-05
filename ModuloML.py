import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
#import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
# import tempfile

def generar_modelos():
    st.empty()
    st.markdown("<h1 style='text-align: center;'>Modelos de Regresión</h1>", unsafe_allow_html=True)
    st.divider()
    with st.expander("Ver detalles del DataSet"):
        texto_largo = """
        Este conjunto de datos es una representación sintética del rendimiento de los estudiantes, diseñado para imitar 
        situaciones del mundo real al considerar factores clave como los hábitos de estudio, los patrones de sueño, el contexto 
        socioeconómico y la asistencia a clases. Cada fila representa un estudiante hipotético y el conjunto de datos incluye tanto 
        las características de entrada como la variable objetivo calculada (calificaciones).
        Fuente remota: https://www.kaggle.com/code/stealthtechnologies/creating-student-performance-dataset
        """
        st.write(texto_largo)
    #st.caption("Se trabaja con un archivo determinado que a continuacion se visualiza:")
    # conectar con archivo de excel
    df=pd.read_csv("Data/data.csv")
    st.write(df)
    # Limpieza de datos (quitar filas con valores nulos)
    df = df[["Socioeconomic Score", "Grades"]].dropna()
    
    # Seleccionar las columnas
    X = df[["Socioeconomic Score"]].values  # Variable independiente
    y = df["Grades"].values  # Variable dependiente
    st.divider()
    op_modo= st.radio("Seleccione el método de Regresion que se va a aplicar 👇",["Lineal", "RandomForest", "Ridge"])
    #st.write(op_modo)
    st.divider()
    if op_modo=='Lineal':
        # Crear el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)

        # Predicciones
        y_pred = model.predict(X)
       
        # Métricas de evaluación
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.subheader("Resultados del modelo Regresión Lineal")
        #st.write(f"**Coeficiente:** = {model.coef_[0]:.4f}') 
        st.write(f"**Coeficiente:** = {model.coef_[0]:.4f}")
        st.write(f"**Intercepto:** = {model.intercept_:.4f}")
        st.write(f"**Ecuación de la recta:** y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
        st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")
        st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")
        df["Prediccion"]=model.predict(X)
        st.write(df)
        # Gráfico de regresión
        fig, ax = plt.subplots()
        ax.scatter(X, y, color="blue", label="Datos reales")
        ax.plot(X, y_pred, color="red", label="Línea de regresión")
        ax.set_title("Regresión Lineal: Socioeconomic Score vs Grade")
        ax.set_xlabel("Socieconomic Score")
        ax.set_ylabel("Grade")
        ax.legend()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        
        if st.button("Exportar a PDF"):
            # export_to_pdf()
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Resultados de la Regresión Lineal", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(250, 10, f"Ecuación de la recta: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}", ln=True)
            pdf.cell(0, 10, f"Error cuadrático medio (MSE): {mse:.4f}", ln=True)
            pdf.cell(0, 10, f"Coeficiente de determinación (R²): {r2:.4f}", ln=True)
	        # Exportar gráfico al PDF
            # buffer = io.BytesIO()
            # fig.savefig(buffer, format='png')
            # buffer.seek(0)
            # image_bytes = buffer.getvalue()
            # pdf.image(image_bytes, x=10, y=60, w=100)
            
            # Guardar PDF
            pdf_file = "regression_results.pdf"
            pdf.output(pdf_file)
            st.success(f"Archivo exportado como {pdf_file}")
            with open(pdf_file, "rb") as file:
                st.download_button(label="Descargar PDF", data=file, file_name=pdf_file, mime="application/pdf")

       
    elif op_modo=='RandomForest':
        #Regression Método Random Forrest
        #st.title("Random Forrest")
        rf_model = RandomForestRegressor(n_estimators= 100, random_state = 42)
        rf_model.fit(X,y)
        df["Prediccion_RF"]=rf_model.predict(X)
       

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear el modelo de Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas de evaluación
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Resultados del Modelo Random Forest")
        st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")
        st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")
        st.write(df)

# Crear un gráfico interactivo con Plotly
        fig = go.Figure()

        # Agregar puntos reales
        fig.add_trace(go.Scatter(
            x=X_test.flatten(),
            y=y_test,
            mode='markers',
            name='Valores Reales',
            marker=dict(color='blue')
        ))

        # Agregar puntos predichos
        fig.add_trace(go.Scatter(
             x=X_test.flatten(),
            y=y_pred,
             mode='markers',
             name='Predicciones',
            marker=dict(color='red', symbol='circle-open')
        ))

        # Configurar el diseño del gráfico
        fig.update_layout(
            title="Random Forest: Socioeconomic Score vs Grades",
            xaxis_title="Socioeconomic Score",
            yaxis_title="Grades",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Mostrar el gráfico 
        st.plotly_chart(fig)

        # Botón para exportar a PDF
        if st.button("Exportar a PDF"):
            # Guardar gráfica como imagen temporal
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                # fig.write_image(temp_img.name, format="png")

                # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Resultados Metodo de Regresion RandomForest", ln=True, align="C")
            pdf.cell(250, 10, f"Error cuadrático medio (MSE): {mse:.4f}", ln=True)
            pdf.cell(0, 10, f"Coeficiente de determinación (R²): {r2:.4f}", ln=True)

            # Guardar PDF
            pdf_file = "regression_randomforest_results.pdf"
            pdf.output(pdf_file)
            st.success(f"Archivo exportado como {pdf_file}")
            with open(pdf_file, "rb") as file:
                st.download_button(label="Descargar PDF", data=file, file_name=pdf_file, mime="application/pdf")


    elif op_modo=='Ridge':
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Parámetro alpha para Ridge
        alpha = st.sidebar.slider("Elige el valor de alpha (penalización):", 0.01, 10.0, 1.0)

        # Crear y entrenar el modelo Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
    
        # Predicciones
        y_pred = ridge.predict(X_test)
        df["Prediccion_R"] = ridge.predict(X) #nuevo
        # Métricas de evaluación
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Mostrar resultados
        st.subheader("Resultados del Modelo Ridge")
        st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")
        st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")
        st.write(df)
        # Crear gráfico interactivo con Plotly
        fig = go.Figure()

        # Valores reales
        fig.add_trace(go.Scatter(
            x=X_test.flatten(),
            y=y_test,
            mode='markers',
            name='Valores Reales',
            marker=dict(color='blue', size=8)
        ))

        # Predicciones
        fig.add_trace(go.Scatter(
            x=X_test.flatten(),
            y=y_pred,
            mode='lines',
            name='Predicciones',
            line=dict(color='red', width=2)
        ))

        # Configurar diseño del gráfico
        fig.update_layout(
            title="Regresión Ridge: Valores Reales vs Predicciones",
            xaxis_title="Socioeconomic Score",
            yaxis_title="Grades",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Mostrar gráfico en Streamlit
        st.plotly_chart(fig)
    
        # Botón para exportar a PDF
        if st.button("Exportar a PDF"):
            # Guardar gráfica como imagen temporal
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                # fig.write_image(temp_img.name, format="png")

                # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Resultados Metodo de Regresion Ridge", ln=True, align="C")
            pdf.cell(250, 10, f"Error cuadrático medio (MSE): {mse:.4f}", ln=True)
            pdf.cell(0, 10, f"Coeficiente de determinación (R²): {r2:.4f}", ln=True)

            # Guardar PDF
            pdf_file = "regression_ridge_results.pdf"
            pdf.output(pdf_file)
            st.success(f"Archivo exportado como {pdf_file}")
            with open(pdf_file, "rb") as file:
                st.download_button(label="Descargar PDF", data=file, file_name=pdf_file, mime="application/pdf")
    
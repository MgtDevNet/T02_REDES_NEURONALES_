"""
Aplicación web de la Parte 1. Desde la raíz del repositorio:

    streamlit run app/app.py

Usa los modelos guardados en models/ (no reentrena). El modelo principal y los enlaces al
reporte y al video se configuran en la sección [app] de config.ini.
"""

import streamlit as st

st.set_page_config(page_title="¿Me prestas? · Simulador de crédito", page_icon=":material/account_balance:", layout="wide")

paginas = st.navigation([
    st.Page("evaluar.py", title="Evaluar crédito", icon=":material/credit_score:", default=True),
    st.Page("como_funciona.py", title="Cómo funciona", icon=":material/menu_book:"),
])
paginas.run()

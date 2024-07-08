
import streamlit as st
import pandas as pd
from utils.functions_streamlit import eda, clustering, generativas, rag
# Creamos la configuración de la página, y el título
st.set_page_config(page_title='Clustering y generativas', layout='wide', page_icon="🌐")

menu = st.sidebar.selectbox(label='Fase del proyecto:', options=('EDA', 'Clustering', 'Generativas', 'RAG'))

if menu == 'EDA':
    eda()

if menu == 'Clustering':
    clustering()


if menu == 'Generativas':
    api_key = st.sidebar.text_input('API Key')
    generativas(api_key)

if menu == 'RAG':
    rag()


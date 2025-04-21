import streamlit as st

st.set_page_config(
    page_title="Thailand Accident Analysis", page_icon="📊", layout="wide"
)


st.title("Welcome to Data Analytic Page🚗")
st.markdown(
    """
    This page shows the process of data analytic from preprocessing until the model training.
    """
)

with open("src/notebook/Data-Analysis-Notebook.md", "r", encoding="utf-8") as f:
    markdown_content = f.read()

st.markdown(markdown_content, unsafe_allow_html=True)



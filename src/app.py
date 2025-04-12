import streamlit as st
import pandas as pd
import numpy as np

st.title("🎈 My new app")

with st.sidebar:
    st.header("About")
    st.write("This is my first app")

st.header("Header with divider", divider="rainbow")

st.markdown("This is a streamlit markdown cell.")

st.subheader("st.columns")
col1, col2 = st.columns(2)
with col1:
    x = st.slider("Select a number", 0, 100, 50)
with col2:
    st.write("Value of :rainbow[**x**] is:", x)

st.subheader("st.area_chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.area_chart(chart_data)
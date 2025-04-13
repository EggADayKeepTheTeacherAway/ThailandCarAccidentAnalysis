import streamlit as st
import pandas as pd
import os

# constants
DATA_FOLDER = "data"
CSV_FILE = [file for file in os.listdir(DATA_FOLDER) if file.endswith(".csv")]
YEAR = list(x for x in range(2012, 2025))

# page
st.title("Thailand Car Accident Analysis")

year = st.selectbox("Select year", YEAR)

st.header(f"Accident data year: :red[{year}]", divider="rainbow")

for file in CSV_FILE:
    if str(year) in file:
        filepath = os.path.join(DATA_FOLDER, file)
        try:
            df = pd.read_csv(filepath)
            df = df.fillna("")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")

if df is not None:
    st.markdown(
        f'''
        ## Summary
        - Total accidents: {len(df)}
        - Total fatalities: {df['จำนวนผู้เสียชีวิต'].sum()}
        - Total injuries: {df['รวมจำนวนผู้บาดเจ็บ'].sum()}
        '''
    )

import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import os

st.set_page_config(page_title="Main Dashboard", layout="wide")

# constants
DATA_FOLDER = "data"
CSV_FILE = [file for file in os.listdir(DATA_FOLDER) if file.endswith(".csv")]
YEAR = list(x for x in range(2012, 2025))
MAP_OPTION = ["2D (Simple Map)", "3D (Pydeck)"]

# page
st.title("Thailand Car Accident Analysis")

selected_year = st.selectbox("Select year", YEAR)

st.header(f"Accident data year: :red[{selected_year}]", divider="rainbow")

data_tab, graph_tab, map_tab, summary_tab = st.tabs(
    ["Data 📈", "Graph 📊", "Map 🗺️", "Summary 📚"]
)


@st.cache_data
def get_df(year: int = 2012):
    for file in CSV_FILE:
        if str(year) in file:
            filepath = os.path.join(DATA_FOLDER, file)
            try:
                df = pd.read_csv(filepath)
                numeric_col = df.select_dtypes(include=["number"]).columns
                df[numeric_col] = df[numeric_col].fillna(-1)
                df = df.fillna("-")
                return df
            except Exception as e:
                st.error(f"Error reading file: {e}")


df = get_df(year=selected_year)

with data_tab:
    if df is not None:
        st.dataframe(df)

with graph_tab:
    if df is not None:
        line_col, bar_col = st.columns(2)
        
        # line graph
        with line_col:
            st.markdown(
                f"""
                ### Line graph of injuries per month in 
                year {selected_year}
                """
            )
            month_df = df.copy()
            month_df["MONTH"] = pd.DataFrame(
                month_df["วันที่เกิดเหตุ"].str.split("/").str[1]
            )
            st.line_chart(
                month_df.groupby("MONTH")["รวมจำนวนผู้บาดเจ็บ"]
                .sum()
                .sort_index(ascending=True)
            )
        # bar graph
        with bar_col:
            bar_graph = st.selectbox(
                "Select Bar x-axis", ["จังหวัด", "มูลเหตุสันนิษฐาน", "บริเวณที่เกิดเหตุ/ลักษณะทาง"]
            )
            try:
                st.markdown(
                    f"""
                    ### Bar graph of injuries per 
                    #### {bar_graph}
                    """
                )
                graph_df = df[df[bar_graph] != "-"]
                st.bar_chart(
                    graph_df[["รวมจำนวนผู้บาดเจ็บ", bar_graph]].groupby(bar_graph).sum()
                )
            except Exception as e:
                st.error(f"Error displaying bar graph: {e}")

        # pie chart
        weather_count = df["สภาพอากาศ"].value_counts().reset_index()
        weather_count.columns = ["Weather", "Count"]
        
        st.markdown(
            """
            ### Pie chart of accidents by weather condition
            """
        )
        fig = px.pie(weather_count, names="Weather", values="Count")
        st.plotly_chart(fig)

with map_tab:
    if df is not None:
        copy_df = df.copy()
        try:
            copy_df = copy_df[["LATITUDE", "LONGITUDE", "รวมจำนวนผู้บาดเจ็บ"]]
            copy_df = copy_df[
                (copy_df["LATITUDE"] != -1) & (copy_df["LONGITUDE"] != -1)
            ]
            copy_df = copy_df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})
        except Exception as e:
            st.error(f"Error displaying map: {e}")
        view_mode = st.radio(
            "### Select map view:", (MAP_OPTION[0], MAP_OPTION[1]), horizontal=True
        )
        if view_mode == MAP_OPTION[0]:
            st.map(copy_df[["lat", "lon"]], zoom=6, use_container_width=True)
        else:
            ele_scale_col, pitch_col = st.columns(2)
            with ele_scale_col:
                ele_scale = st.slider(
                    "Elevation scale", min_value=1, max_value=10000, value=1000
                )
            with pitch_col:
                pitch = st.slider("Angle", min_value=0, max_value=60, value=50, step=1)
            layer = [
                pdk.Layer(
                    "ColumnLayer",
                    data=copy_df,
                    get_position=["lon", "lat"],
                    get_elevation="รวมจำนวนผู้บาดเจ็บ",
                    elevation_scale=ele_scale,
                    radius=1000,
                    get_fill_color=[255, 0, 0, 200],
                    pickable=True,
                    auto_highlight=True,
                ),
            ]

            view_state = pdk.ViewState(
                latitude=copy_df["lat"].mean(),
                longitude=copy_df["lon"].mean(),
                zoom=6,
                pitch=pitch,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=layer,
                    initial_view_state=view_state,
                    tooltip={"text": "Injuries: {รวมจำนวนผู้บาดเจ็บ}"},
                )
            )

with summary_tab:
    if df is not None:
        st.markdown(
            f"""
            ## Summary
            - Total accidents: {len(df)}
            - Total fatalities: {df['จำนวนผู้เสียชีวิต'].sum()}
            - Total injuries: {df['รวมจำนวนผู้บาดเจ็บ'].sum()}
            """
        )

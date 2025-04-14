import streamlit as st
import pandas as pd
import pydeck as pdk
import os

# constants
DATA_FOLDER = "data"
CSV_FILE = [file for file in os.listdir(DATA_FOLDER) if file.endswith(".csv")]
YEAR = list(x for x in range(2012, 2025))
MAP_OPTION = ["2D (Simple Map)", "3D (Pydeck)"]

# page
st.title("Thailand Car Accident Analysis")

selected_year = st.selectbox("Select year", YEAR)

st.header(f"Accident data year: :red[{selected_year}]", divider="rainbow")

data_tab, graph_tab, map_tab, summary_tab = st.tabs(["Data", "Graph", "Map", "Summary"])


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
    # TODO: create a graph displaying the accident statistical data
    pass

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
            "Select map view:", (MAP_OPTION[0], MAP_OPTION[1]), horizontal=True
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

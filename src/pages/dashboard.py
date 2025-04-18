import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import os

st.set_page_config(
    page_title="Thailand Accident Analysis", page_icon="📈", layout="wide"
)


class DashboardPage:
    def __init__(self):
        self.DATA_FOLDER = "data"
        self.CSV_FILE = [
            file for file in os.listdir(self.DATA_FOLDER) if file.endswith(".csv")
        ]
        self.YEAR = list(range(2012, 2025))
        self.MAP_OPTION = ["2D (Simple Map)", "3D (Pydeck)"]
        self.df = None

    def get_df(self, year: int) -> pd.DataFrame | None:
        for file in self.CSV_FILE:
            if str(year) in file:
                filepath = os.path.join(self.DATA_FOLDER, file)
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    df.drop(columns=["วันที่รายงาน"], inplace=True, errors="ignore")
                    numeric_col = df.select_dtypes(include=["number"]).columns
                    df[numeric_col] = df[numeric_col].fillna(-1)
                    df = df.fillna("-")
                    return df
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        return None

    def show_data_tab(self):
        if self.df is not None:
            st.dataframe(self.df)

    def show_graph_tab(self, year):
        if self.df is None:
            return

        line_col, bar_col = st.columns(2)

        # Line Chart
        with line_col:
            st.markdown(f"### Line graph of injuries per month in year {year}")
            month_df = self.df.copy()
            month_df["MONTH"] = month_df["วันที่เกิดเหตุ"].str.split("/").str[1]
            st.line_chart(
                month_df.groupby("MONTH")["รวมจำนวนผู้บาดเจ็บ"].sum().sort_index()
            )

        # Bar Chart
        with bar_col:
            bar_graph = st.selectbox(
                "Select Bar x-axis", ["จังหวัด", "มูลเหตุสันนิษฐาน", "บริเวณที่เกิดเหตุ/ลักษณะทาง"]
            )
            try:
                st.markdown(f"### Bar graph of injuries per {bar_graph}")
                graph_df = self.df[self.df[bar_graph] != "-"]
                st.bar_chart(graph_df.groupby(bar_graph)[["รวมจำนวนผู้บาดเจ็บ"]].sum())
            except Exception as e:
                st.error(f"Error displaying bar graph: {e}")

        # Pie Chart
        st.markdown("### Pie chart of accidents by weather condition")
        weather_count = self.df["สภาพอากาศ"].value_counts().reset_index()
        weather_count.columns = ["Weather", "Count"]
        fig = px.pie(weather_count, names="Weather", values="Count")
        st.plotly_chart(fig)

    def show_map_tab(self):
        if self.df is None:
            return

        try:
            copy_df = self.df[["LATITUDE", "LONGITUDE", "รวมจำนวนผู้บาดเจ็บ"]]
            copy_df = copy_df[
                (copy_df["LATITUDE"] != -1) & (copy_df["LONGITUDE"] != -1)
            ]
            copy_df = copy_df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})
        except Exception as e:
            st.error(f"Error displaying map: {e}")
            return

        view_mode = st.radio("### Select map view:", self.MAP_OPTION, horizontal=True)
        if view_mode == self.MAP_OPTION[0]:
            st.map(copy_df[["lat", "lon"]], zoom=6, use_container_width=True)
        else:
            ele_scale_col, pitch_col = st.columns(2)
            with ele_scale_col:
                ele_scale = st.slider(
                    "Elevation scale", min_value=1, max_value=10000, value=1000
                )
            with pitch_col:
                pitch = st.slider("Angle", min_value=0, max_value=60, value=50)

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
                )
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

    def show_summary_tab(self):
        if self.df is not None:
            st.markdown(
                f"""
                ## Summary
                - Total accidents: {len(self.df)}
                - Total fatalities: {self.df['จำนวนผู้เสียชีวิต'].sum()}
                - Total injuries: {self.df['รวมจำนวนผู้บาดเจ็บ'].sum()}
            """
            )

    def render(self):
        st.set_page_config(page_title="Main Dashboard", layout="wide")
        st.title("Welcome to Dashboard Page🚗")

        selected_year = st.selectbox("Select year", self.YEAR)
        st.header(f"Accident data year: :red[{selected_year}]", divider="rainbow")
        self.df = self.get_df(selected_year)

        data_tab, graph_tab, map_tab, summary_tab = st.tabs(
            ["Data 📈", "Graph 📊", "Map 🗺️", "Summary 📚"]
        )

        with data_tab:
            self.show_data_tab()

        with graph_tab:
            self.show_graph_tab(selected_year)

        with map_tab:
            self.show_map_tab()

        with summary_tab:
            self.show_summary_tab()


if __name__ == "__main__":
    page = DashboardPage()
    page.render()

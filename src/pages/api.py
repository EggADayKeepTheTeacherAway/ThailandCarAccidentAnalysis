import streamlit as st
from streamlit_js_eval import get_geolocation
from api.api_app import create_app
from fastapi.testclient import TestClient

app = create_app()
client = TestClient(app)

MAX_YEAR = 2025
MIN_YEAR = 2012

st.set_page_config(
    page_title="Thailand Accident Analysis", page_icon="📍", layout="wide"
)

st.title("Welcome to API Page:car:")
st.markdown(
    """
    This page helps explore the API with the example of the results after calling the API.  
    Open the expander to see the result of each API and your current geolocation.
    """
)

st.write(
    "To use the API make sure that the API server is running in the port 8000. Then fetch the API using the following URL :grey-badge[http://127.0.0.1:8000/<API Endpoint>] or if you are hosting the API on the cloud server make sure to replace the URL with the cloud server URL."
)

with st.expander(":grey-badge[Geolocation]"):
    result = get_geolocation()
    if result and "coords" in result:
        latitude = result["coords"].get("latitude")
        longitude = result["coords"].get("longitude")
        st.success(f"Your Location\n\nLatitude: {latitude}\n\nLongitude: {longitude}")
    else:
        st.error(
            "Unable to get geolocation data. Please check your browser settings or change the browser."
        )

with st.expander(":green-badge[GET]:grey-badge[/accidents/{year}]"):
    year = st.number_input(
        "Enter year between 2012 to 2025 for accident data",
        min_value=MIN_YEAR,
        max_value=MAX_YEAR,
        value=None,
        placeholder="Type in a year",
    )
    response = client.get(f"/accidents/{year}")
    if response.status_code == 200:
        data = response.json()
        st.json(data)
    else:
        st.error(f"Error: {response.json().get('detail', 'No data found')}")

with st.expander(":green-badge[GET]:grey-badge[/accidents/{year}/summary]"):
    year = st.number_input(
        "Enter year between 2012 to 2025 for summary data",
        min_value=MIN_YEAR,
        max_value=MAX_YEAR,
        value=None,
        placeholder="Type in a year",
    )
    response = client.get(f"/accidents/{year}/summary")
    if response.status_code == 200:
        data = response.json()
        st.json(data)
    else:
        st.error(f"Error: {response.json().get('detail', 'No data found')}")

with st.expander(":green-badge[GET]:grey-badge[/predict/accident]"):
    latitude = st.number_input(
        "Enter latitude to predict accident",
        value=None,
        placeholder="Type in a latitude",
    )
    longitude = st.number_input(
        "Enter longitude to predict accident",
        value=None,
        placeholder="Type in a longitude",
    )
    response = client.get(f"/predict/accident?lat={latitude}&lon={longitude}")
    if response.status_code == 200:
        data = response.json()
        st.json(data)
    else:
        st.error(
            f"{response.json().get('detail')[0].get('msg', 'No data found').capitalize()}"
        )

with st.expander(":green-badge[GET]:grey-badge[/predict/injuries]"):
    latitude = st.number_input(
        "Enter latitude to predict injuries",
        value=None,
        placeholder="Type in a latitude",
    )
    longitude = st.number_input(
        "Enter longitude to predict injuries",
        value=None,
        placeholder="Type in a longitude",
    )
    response = client.get(f"/predict/accident?lat={latitude}&lon={longitude}")
    if response.status_code == 200:
        data = response.json()
        st.json(data)
    else:
        st.error(
            f"{response.json().get('detail')[0].get('msg', 'No data found').capitalize()}"
        )

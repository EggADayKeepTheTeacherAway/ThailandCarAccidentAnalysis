import streamlit as st
from streamlit_js_eval import get_geolocation

st.set_page_config(
    page_title="Thailand Accident Analysis", page_icon="📍", layout="wide"
)

st.title("Welcome to API Page:car:")
st.markdown(
    """
    This page helps explore the API with the example of the results after calling the API.  
    Use the sidebar to select a specific analysis page.
    """
)

result = get_geolocation()
if result and "coords" in result:
    latitude = result["coords"].get("latitude")
    longitude = result["coords"].get("longitude")
    st.success(
        f"Your Location\n\n" f"Latitude: {latitude}\n\n" f"Longitude: {longitude}"
    )
else:
    st.error(
        "Unable to get geolocation data. Please check your browser settings or change the browser."
    )

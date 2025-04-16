import streamlit as st
from streamlit_js_eval import get_geolocation

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
    st.write(f"Your Location:\nLatitude: {latitude}\nLongitude: {longitude}")
else:
    st.error("Unable to get geolocation data. Please check your browser settings or change the browser.")
    

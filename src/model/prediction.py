from tensorflow.keras.models import load_model
import joblib
import os
import pandas as pd
import numpy as np
import datetime
import requests

BASE_DIR = os.path.dirname(__file__)

model = load_model(os.path.join(BASE_DIR, 'accident_prediction_model_20.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'accident_prediction_scaler_20.save'))
features = joblib.load(os.path.join(BASE_DIR, 'accident_prediction_features_20.save'))
weather_features = [col for col in features if col.startswith('weather_')]
time_steps = 10


def convert_weather_status_to_thai(weather):
    weather = weather.lower()
    if weather in ['sunny', 'clear', 'partly cloudy', 'cloudy']:
        return 'แจ่มใส'
    elif 'rain' in weather or 'drizzle' in weather or 'shower' in weather or 'storm' in weather:
        return 'ฝนตก'
    elif 'fog' in weather or 'smoke' in weather or 'mist' in weather or 'haze' in weather or 'dust' in weather:
        return 'มีหมอก/ควัน/ฝุ่น'
    elif 'overcast' in weather or 'gloomy' in weather:
        return 'มืดครึ้ม'
    elif 'thunder' in weather or 'flood' in weather or 'cyclone' in weather or 'tornado' in weather:
        return 'ภัยธรรมชาติ เช่น พายุ น้ำท่วม'
    elif 'other' in weather:
        return 'อื่นๆ'
    else:
        return 'อื่นๆ'

def get_user_prediction(user_lat, user_lon):
    datetime_now = datetime.datetime.now()
    query = f"{user_lat},{user_lon}"

    # Fetch the data from the Openweather API
    try:
        response = requests.get(
            f'http://api.weatherapi.com/v1/current.json?key=a85505792f084a93971181237251504&q={query}&aqi=no'
        )
        response.raise_for_status()
        response_json = response.json()
        current_data = response_json.get('current')
        if not current_data or 'condition' not in current_data or 'text' not in current_data['condition']:
            raise ValueError("Missing 'current.condition.text' in API response")

        current_weather = convert_weather_status_to_thai(current_data['condition']['text'])

    except Exception as e:
        print(f"Weather API call failed: {e}")
        current_weather = 'อื่นๆ'  # Fallback weather category

    user_df = pd.DataFrame({
        'LATITUDE': [user_lat],
        'LONGITUDE': [user_lon],
        'hour': [datetime_now.hour],
        'dayofweek': [datetime_now.weekday()],
        'month': [datetime_now.month],
        'accident_lag1': [0],
        'rolling_mean_3': [0],
    })

    # One-hot encode weather column
    weather_one_hot = pd.DataFrame(np.zeros((1, len(weather_features))), columns=weather_features)
    weather_col_name = f'weather_{current_weather}'
    if weather_col_name in weather_one_hot.columns:
        weather_one_hot.at[0, weather_col_name] = 1
    else:
        print(f"Weather '{current_weather}' not in training all weather features will be zero.")

    user_input_df = pd.concat([user_df, weather_one_hot], axis=1)

    # Ensure all features are present
    for col in features:
        if col not in user_input_df.columns:
            user_input_df[col] = 0
    user_input_df = user_input_df[features]

    user_input_scaled = scaler.transform(user_input_df)
    user_input_sequence = np.repeat(user_input_scaled[np.newaxis, :, :], time_steps, axis=1)

    prediction = model.predict(user_input_sequence)
    return prediction[0][0]

if __name__ == '__main__':
    user_lat = 18.838155899963
    user_lon = 100.74123321561
    prediction = get_user_prediction(user_lat, user_lon)
    datetime_now = datetime.datetime.now().strftime("%d/%m/%Y %I:%M %p")
    print(f"Accident Risk at (Lat: {user_lat:.2f}  Lon: {user_lon:.2f} Date Time: {datetime_now}) is {prediction:.4f}")
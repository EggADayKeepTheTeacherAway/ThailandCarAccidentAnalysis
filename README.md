# ThailandCarAccidentAnalysis
## **Project Overview**
ThailandCarAccidentAnalysis is a project that aims to predict the possibility of a car accident in Thailand using the timestamp and location of the user by inputting the designated location with a timestamp.

## **Primary data source(s)**
The primary data are collected from the user, such as

 - **location and timestamp**, example `103.1`, `16.4`, `2025-01-02 06:11:00`

## **Secondary data source(s)**
Secondary data are from both Kaggle and the weather API.

 - **Weather Data**: Real-time data from [OpenWeather API](https://www.weatherapi.com/weather/q/bangkok-2366981?utm_source=chatgpt.com) for fog, rain, or other weather conditions that affect driving safety.
 - **Historical Accident Data**:  [Thailand Road Accident 2019-2022](https://www.kaggle.com/datasets/thaweewatboy/thailand-road-accident-2019-2022?utm_source=chatgpt.com) The dataset from Kaggle provides past road accident data in Thai used to train the prediction model for risk analysis and to improve the accuracy of predictions.
 - **[TRAMS](https://trams.mot.go.th/dashboard_report)** (TRansport Accident Management System): The ministry of transport dashboards.

## **API to be provided to users**
 - Website to input the location and time to predict the accident and show the historical data from the database that can be sorted by location

## Initial set up
1. Clone the repository
    ```shell
    git clone https://github.com/EggADayKeepTheTeacherAway/ThailandCarAccidentAnalysis.git
    ```
2. Change directory to the project root directory
    ```shell
    cd ThailandCarAccidentAnalysis
    ```

## Install requirements and run the app.
 1. Create virtual environment
    1. Create venv
          ```shell
          python -m venv venv
          ```
    2. Activate venv
       1. For Windows
            ```shell
            # In cmd.exe
            venv\Scripts\activate.bat
            # In PowerShell
            venv\Scripts\Activate.ps1
            ```
       2. For MacOS and Linux
            ```shell
            source myvenv/bin/activate
            ```
 2. Install the requirements
     ```shell
     pip install -r requirements.txt
     ```
     Install playwright required browser
     ```shell
     playwright install
     ```
 3. Run the app
     1. Run Main webpage
          ```shell
          streamlit run src/home.py
          ```
     2. Run FastAPI
          ```shell
          uvicorn src.api.api_main:app --reload   
          ```

## Run the test
```shell
pytest .\tests\
```

![egg accident](https://media.discordapp.net/attachments/811995210225156148/1363924955192102932/image.jpg?ex=6807cd99&is=68067c19&hm=daf289948c195f32797454ab0ae7aee05fb362ebbf4952fa355badd5a678c729&=&format=webp&width=1200&height=960)
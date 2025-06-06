# src/api/api_app.py
import os
import sys
import pandas as pd
import datetime
from fastapi import FastAPI, HTTPException, Query

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.prediction import get_user_prediction

TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATASET_NEED = [
    "ปีที่เกิดเหตุ",
    "วันที่เกิดเหตุ",
    "เวลา",
    "วันที่รายงาน",
    "เวลาที่รายงาน",
    "จังหวัด",
    "บริเวณที่เกิดเหตุ/ลักษณะทาง",
    "สภาพอากาศ",
    "LATITUDE",
    "LONGITUDE",
]


def create_app():
    app = FastAPI()
    AccidentAPI(app)
    return app


class AccidentAPI:
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_routes()

    @staticmethod
    def format_record(record):
        return {
            "date": str(record["วันที่เกิดเหตุ"].values[0]),
            "time": str(record["เวลา"].values[0]),
            "road_feature": str(record["บริเวณที่เกิดเหตุ/ลักษณะทาง"].values[0]),
            "location": {
                "province": str(record["จังหวัด"].values[0]),
                "weather": str(record["สภาพอากาศ"].values[0]),
                "lat": float(record["LATITUDE"].values[0]),
                "lon": float(record["LONGITUDE"].values[0]),
            },
        }

    def setup_routes(self):
        @self.app.get("/accidents/{year}")
        async def get_accidents_year(year: str = ""):
            for filename in os.listdir("data"):
                if filename.endswith(".csv") and year in filename:
                    filepath = os.path.join("data", filename)
                    try:
                        df = pd.read_csv(filepath)
                        df = df[DATASET_NEED].fillna("")
                        return df.to_dict(orient="records")
                    except Exception as e:
                        raise HTTPException(
                            status_code=500, detail=f"Error reading file: {e}"
                        )
            return HTTPException(
                status_code=404, detail=f"No data file found for year {year}"
            )

        @self.app.get("/accidents/{year}/summary")
        async def get_accidents_year_summary(year: str = ""):
            for filename in os.listdir("data"):
                if filename.endswith(".csv") and year in filename:
                    filepath = os.path.join("data", filename)
                    try:
                        df = pd.read_csv(filepath)
                        first_rc = df.head(1)
                        last_rc = df.tail(1)
                        print(f"File name: {filename}")
                        return {
                            "year": year,
                            "total_accidents": int(df.shape[0]),
                            "total_deaths": int(df["จำนวนผู้เสียชีวิต"].sum()),
                            "total_injuries": int(df["รวมจำนวนผู้บาดเจ็บ"].sum()),
                            "average_deaths": float(df["จำนวนผู้เสียชีวิต"].mean()),
                            "average_injuries": float(df["รวมจำนวนผู้บาดเจ็บ"].mean()),
                            "first_record": self.format_record(first_rc),
                            "last_record": self.format_record(last_rc),
                        }
                    except Exception as e:
                        raise HTTPException(
                            status_code=500, detail=f"Error reading file: {e}"
                        )
            return HTTPException(
                status_code=404, detail=f"File not found for year {year}"
            )

        @self.app.get("/predict/accident")
        async def get_accidents_year_summary(
                lat: float = Query(...), lon: float = Query(...)
        ):
            prediction = get_user_prediction(lat, lon)
            time = datetime.datetime.now().strftime(TIME_FORMAT)
            return {
                "lat": lat,
                "lon": lon,
                "time": time,
                "prediction": float(prediction),
            }

        @self.app.get("/predict/injuries")
        async def get_accidents_year_summary(
                lat: float = Query(...), lon: float = Query(...)
        ):
            prediction = "injuries prediction"
            time = datetime.datetime.now().strftime(TIME_FORMAT)
            # TODO: insert model here
            return {
                "lat": lat,
                "lon": lon,
                "time": time,
                "prediction": prediction,
            }

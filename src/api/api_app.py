# src/api/api_app.py
import os
import pandas as pd
from fastapi import FastAPI, HTTPException

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
                            "first_record": self.format_record(first_rc),
                            "last_record": self.format_record(last_rc),
                        }
                    except Exception as e:
                        raise HTTPException(
                            status_code=500, detail=f"Error reading file: {e}"
                        )
            return HTTPException(
                status_code=404, detail="File not found for year {year}"
            )

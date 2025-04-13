# src/api/api_app.py
import os
from fastapi import FastAPI, HTTPException


class AccidentAPI:
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/accidents/{year}")
        async def get_accidents(year: str = ""):
            for filename in os.listdir("data"):
                if filename.endswith(".csv") and year in filename:
                    return {"filename", filename}
            return HTTPException(status_code=404, detail="File not found for year {year}")

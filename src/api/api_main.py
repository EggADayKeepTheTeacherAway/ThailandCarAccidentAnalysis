# main.py

from fastapi import FastAPI
from src.api.api_app import AccidentAPI

app = FastAPI()
AccidentAPI(app)

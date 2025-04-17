# main.py

from fastapi import FastAPI
from src.api.api_app import create_app

app = create_app()

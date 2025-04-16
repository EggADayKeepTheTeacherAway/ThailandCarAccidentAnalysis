import unittest

from src.api.api_app import create_app
from fastapi.testclient import TestClient

VALID_YEAR = "2020"
INVALID_YEAR = "2000"


class TestAccidentAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_get_accident_year_no_data(self):
        response = self.client.get("/accidents/")
        self.assertEqual(response.status_code, 404)

    def test_get_valid_accident_year(self):
        response = self.client.get(f"/accidents/{VALID_YEAR}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_get_invalid_accident_year(self):
        response = self.client.get(f"/accidents/{INVALID_YEAR}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status_code"), 404)
        self.assertEqual(
            data.get("detail"), f"No data file found for year {INVALID_YEAR}"
        )

class TestSummaryAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_get_accident_summary_no_data(self):
        response = self.client.get("/accidents/summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("detail"), f"No data file found for year summary")

    def test_get_valid_accident_year(self):
        response = self.client.get(f"/accidents/{VALID_YEAR}/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_get_invalid_accident_year(self):
        response = self.client.get(f"/accidents/{INVALID_YEAR}/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status_code"), 404)
        self.assertEqual(
            data.get("detail"), f"File not found for year {INVALID_YEAR}"
        )
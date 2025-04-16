import time
import pytest
import subprocess
from playwright.sync_api import sync_playwright, expect


@pytest.fixture(scope="session", autouse=True)
def setup_streamlit():
    """Launch the Streamlit app."""
    process = subprocess.Popen(["streamlit", "run", "src/home.py"])
    time.sleep(3)
    yield
    return process


def test_home_page():
    """Test the home page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        expect(page).to_have_title("Thailand Accident Analysis")
        expect(page.locator("h1")).to_have_text(
            "Welcome to Thailand Accident Analysis Web Page🚗"
        )

        anchors = page.locator("a")
        count = anchors.count()
        for i in range(count):
            href = anchors.nth(i).get_attribute("href")
            text = anchors.nth(i).inner_text()
            print(f"Link {i+1}: Text = '{text}', Href = '{href}'")

        page.locator("text=api").click()
        time.sleep(2)
        expect(page.locator("h1")).to_have_text("Welcome to API Page🚗")
        page.screenshot(path="./screenshots/home_page.png")

        page.locator("text=dashboard").click()
        time.sleep(2)
        expect(page.locator("h1")).to_have_text("Welcome to Dashboard Page🚗")
        page.screenshot(path="./screenshots/home_page.png")

        page.locator("text=data analytic").click()
        time.sleep(2)
        expect(page.locator("h1")).to_have_text("Welcome to Data Analytic Page🚗")
        page.screenshot(path="./screenshots/home_page.png")

        time.sleep(2)
        page.close()
        browser.close()


if __name__ == "__main__":
    test_home_page()

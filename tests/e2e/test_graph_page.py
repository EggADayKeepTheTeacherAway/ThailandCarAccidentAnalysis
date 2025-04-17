import time
from playwright.sync_api import sync_playwright, expect

HEADLESS = False


def test_dashboard_data_tab():
    """Test the dashboard data tab."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Click sidebar link to 'dashboard'
        page.get_by_role("link", name="dashboard").click()
        time.sleep(2)

        # Click the "Data 📈" tab
        try:
            button = page.locator('button:has-text("Data 📈")')
            print("Button found:", button.count())
            button.click()
        except Exception as e:
            page.get_by_label("Data 📈").click()
        time.sleep(2)

        expect(page.locator(".stDataFrame")).to_be_visible()
        print("Data frame is visible")
        expect(page.locator(".stDataFrame")).to_have_count(1)
        expect(page.locator(".stDataFrame", has_text="2012")).to_be_visible()
        print("2012 Data frame year 2012 is visible")
        expect(page.locator(".stDataFrame", has_text="2013")).not_to_be_visible()
        print("2012 Data frame year 2013 is not visible")

        # Click the selectbox to change the year
        page.get_by_label("Select year").click()
        print("Clicked on the selectbox")
        time.sleep(1)

        page.locator("text=2013").click()
        time.sleep(1)

        expect(page.locator(".stDataFrame")).to_be_visible()
        print("Data frame is visible")
        expect(page.locator(".stDataFrame")).to_have_count(1)
        expect(page.locator(".stDataFrame", has_text="2013")).to_be_visible()
        print("2013 Data frame year 2013 is visible")
        expect(page.locator(".stDataFrame", has_text="2012")).not_to_be_visible()
        print("2012 Data frame year 2012 is not visible")

        page.screenshot(path="tests/e2e/screenshots/dashboard/data_tab.png")
        page.close()
        browser.close()


def test_dashboard_graph_tab():
    """Test the dashboard graph tab."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Click sidebar link to 'dashboard'
        page.get_by_role("link", name="dashboard").click()
        time.sleep(2)

        # Click the "Graph 📊" tab
        try:
            button = page.locator('button:has-text("Graph 📊")')
            print("Button found:", button.count())
            button.click()
        except Exception as e:
            page.get_by_label("Graph 📊").click()

        time.sleep(2)

        # Check Line chart
        total_canvas = page.locator("canvas")
        line_canvas = page.locator("canvas").first
        expect(line_canvas).to_be_visible()
        print("Line chart is visible")
        line_canvas.screenshot(path="tests/e2e/screenshots/dashboard/line_chart1.png")

        # Check Bar chart
        selectbox = page.get_by_label("Select Bar x-axis")
        expect(selectbox).to_be_visible()
        print("Select box is visible")
        for option in ["จังหวัด", "มูลเหตุสันนิษฐาน", "บริเวณที่เกิดเหตุ/ลักษณะทาง"]:
            selectbox.click()
            time.sleep(1)
            selectbox.fill(option)
            selectbox.press("Enter")
            time.sleep(1)

            canvas_count = page.locator("canvas")
            expect(canvas_count).to_have_count(total_canvas.count())
            print(f"Bar chart is visible for {option.replace('/', '_')}")
            page.locator("canvas").nth(1).screenshot(
                path=f"tests/e2e/screenshots/dashboard/bar_chart_{option.replace('/', '_')}.png"
            )
            time.sleep(1)

        page.close()
        browser.close()


def test_dashboard_map_tab():
    """Test the dashboard map tab."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Click sidebar link to 'dashboard'
        page.get_by_role("link", name="dashboard").click()
        time.sleep(2)

        # Click the "Map 🗺️" tab
        try:
            button = page.locator('button:has-text("Map 🗺️")')
            print("Button found:", button.count())
            button.click()
        except Exception as e:
            page.get_by_label("Map 🗺️").click()
        time.sleep(2)

        # Check Map
        map_2D = page.locator(
            "#view-default-view > div:nth-child(1) > div.mapboxgl-map"
        )
        expect(map_2D).to_be_visible()
        print("Map is visible")

        radio = page.get_by_label("Select map view:")
        expect(radio).to_be_visible()
        print("Radio button is visible total count:", radio.count())
        for option in ["2D (Simple Map)", "3D (Pydeck)"]:
            radio.locator("text=" + option).click()
            page.wait_for_timeout(750)
            map_23D = page.locator(
                "#view-default-view > div:nth-child(1) > div.mapboxgl-map"
            )
            expect(map_23D).to_be_visible()
            print(f"Map is visible for {option}")
            map_23D.screenshot(path=f"tests/e2e/screenshots/dashboard/map_{option}.png")

        page.close()
        browser.close()


def test_dashboard_summary_tab():
    """Test the dashboard summary tab."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Click sidebar link to 'dashboard'
        page.get_by_role("link", name="dashboard").click()
        time.sleep(2)

        button = page.locator('button:has-text("Summary 📚")')
        print("Button found:", button.count())
        button.click()

        selectbox = page.get_by_label("Select year")
        expect(selectbox).to_be_visible()
        print("Select box is visible")

        last_year_data = ""

        for year in range(2012, 2025, 2):
            selectbox.click()
            selectbox.fill(str(year))
            selectbox.press("Enter")
            page.wait_for_timeout(500)

            summary_data = page.get_by_label("Summary")
            print("found ul:", summary_data.count())
            if summary_data.count() == 0:
                print("No summary data found")
                continue
            if last_year_data == summary_data.inner_text():
                print("Summary data is same as last year")
                continue
            last_year_data = summary_data.inner_text()
            expect(summary_data).to_be_visible()
            expect(summary_data).to_have_count(1)
            print(f"Summary {year} data is visible")

            list_items = summary_data.locator("li")
            expect(list_items).to_have_count(3)

            page.screenshot(
                path=f"tests/e2e/screenshots/dashboard/summary/summary_{year}.png"
            )

        page.close()
        browser.close()


if __name__ == "__main__":
    test_dashboard_summary_tab()

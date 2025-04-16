import time
from playwright.sync_api import sync_playwright, expect


def test_dashboard_data_tab():
    """Test the dashboard data tab."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Click sidebar link to 'dashboard'
        page.get_by_role("link", name="dashboard").click()
        time.sleep(2)

        # Click the "Data 📈" tab
        page.locator("#tabs-bui2-tab-0").click()
        time.sleep(2)

        expect(page.locator(".stDataFrame")).to_be_visible()
        print("Data frame is visible")
        expect(page.locator(".stDataFrame")).to_have_count(1)
        expect(page.locator(".stDataFrame", has_text="2012")).to_be_visible()
        print("2012 Data frame year 2012 is visible")
        expect(page.locator(".stDataFrame", has_text="2013")).not_to_be_visible()
        print("2012 Data frame year 2013 is not visible")

        # Click the selectbox to change the year
        page.locator(
            "#root > div:nth-child(1) > div.withScreencast > div > div > section.stMain.st-emotion-cache-bm2z3a.eht7o1d1 > div.stMainBlockContainer.block-container.st-emotion-cache-t1wise.eht7o1d4 > div > div > div > div:nth-child(2) > div > div"
        ).click()
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
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Click sidebar link to 'dashboard'
        page.get_by_role("link", name="dashboard").click()
        time.sleep(2)

        # Click the "Graph 📊" tab
        page.locator("#tabs-bui2-tab-1").click()
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
            print(f"Bar chart is visible for {option.replace("/", "_")}")
            page.locator("canvas").nth(1).screenshot(
                path=f"tests/e2e/screenshots/dashboard/bar_chart_{option.replace("/", "_")}.png"
            )
            time.sleep(1)

        page.close()
        browser.close()


if __name__ == "__main__":
    test_dashboard_graph_tab()

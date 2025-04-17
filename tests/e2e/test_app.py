import time
import requests


def test_for_server(url="http://localhost:8501", timeout=50):
    start_time = time.time()
    status = False
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            assert response.status_code == 200
            status = True
            time.sleep(1)
            break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    if not status:
        raise RuntimeError(f"❌ Server not ready after {timeout} seconds.")

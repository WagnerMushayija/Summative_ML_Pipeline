import requests
import time

url = "https://summative-api.onrender.com/"

while True:
    try:
        r = requests.get(url, timeout=10)
        print(f"Pinged {url} - Status: {r.status_code}")
    except Exception as e:
        print(f"Ping failed: {e}")
    time.sleep(600)  # every 10 minutes
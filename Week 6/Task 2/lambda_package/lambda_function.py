import json
import requests
from bs4 import BeautifulSoup

MAX_BYTES = 1_000_000  # 1 MB limit
TIMEOUT = 10

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts & styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())

    return text[:5000]  # limit output size

def lambda_handler(event, context):
    """
    Expected input from Bedrock Agent:
    {
      "url": "https://example.com"
    }
    """

    url = event.get("url")
    if not url:
        return {"error": "URL not provided"}

    try:
        headers = {
            "User-Agent": "Bedrock-Web-Scraper/1.0",
            "Accept-Encoding": "gzip, deflate"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=TIMEOUT,
            allow_redirects=True,
            stream=True
        )

        content = response.raw.read(MAX_BYTES, decode_content=True)

        text = clean_html(content)

        return {
            "url": url,
            "text": text
        }

    except Exception as e:
        return {
            "error": str(e)
        }

from fastapi import FastAPI
from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
import re

app = FastAPI(title="MCP Server")

SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly"
]

creds = service_account.Credentials.from_service_account_file(
    "credentials/service_account.json", scopes=SCOPES
)

FOLDER_ID = "1dItMGOzlnxQBG7ZNfJyK8JkJbVLLt2rH"

drive_service = build("drive", "v3", credentials=creds)
docs_service = build("docs", "v1", credentials=creds)


class ToolRequest(BaseModel):
    input: str

@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {"name": "google_docs_search", "description": "Search insurance-related Google Docs"}
        ]
    }


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


@app.post("/tools/google_docs_search/invoke")
def invoke_google_docs(req: ToolRequest):
    keyword = req.input.lower()

    results = drive_service.files().list(
        q = f"'{FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.document'",
        pageSize=10,
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])

    matched_docs = []

    for file in files:
        doc = docs_service.documents().get(
            documentId=file["id"]
        ).execute()

        content = ""
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if paragraph:
                for elem in paragraph.get("elements", []):
                    text_run = elem.get("textRun")
                    if text_run:
                        content += text_run.get("content", "")

        content_n = normalize(content)
        name_n = normalize(file["name"])
        keyword_n = normalize(keyword)

        combined = name_n + " " + content_n
        words = keyword_n.split()

        if all(word in combined for word in words):
            matched_docs.append(
                f"{file['name']}\n{content[:800]}"
            )


    if not matched_docs:
        return {
            "output": f"No documents found for keyword: {req.input}"
        }

    return {
        "output": "\n\n".join(matched_docs)
    }

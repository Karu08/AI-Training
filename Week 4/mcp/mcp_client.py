import requests
from langchain.tools import Tool

MCP_SERVER_URL = "http://127.0.0.1:3333"

def load_mcp_tools():
    resp = requests.get(f"{MCP_SERVER_URL}/tools").json()
    tools = []
    for t in resp["tools"]:
        name = t["name"]
        description = t["description"]
        def wrapper(query, name=name):
            r = requests.post(f"{MCP_SERVER_URL}/tools/{name}/invoke", json={"input": query})
            return r.json()["output"]
        tools.append(Tool(name=name, func=wrapper, description=description))
    return tools

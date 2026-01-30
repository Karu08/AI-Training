from langchain_community.tools import DuckDuckGoSearchRun

def web_search(query:str)->str:
    """
    Does a web search to return results for general user queries

    Args:
        query: string
    
    Returns:
        the result for user query as a string
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)



import json
from pathlib import Path
from typing import Dict, Any, Union

from pageindex.client import PageIndexClient

class PlatformPageIndex:
    """
    A minimal and highly integrated core wrapper around PageIndexClient
    for an LLM-based smart learning dialog platform.

    This class exposes the core capabilities:
    1. Indexing a document (PDF or Markdown) to generate a hierarchical tree.
    2. Retrieving the tree structure to allow an LLM to reason about document contents.
    3. Retrieving specific page/line content to use as precise knowledge references
       for source tracing, allowing verification of the LLM's answers.
    """
    def __init__(self, workspace: str = "./platform_workspace", api_key: str = None, model: str = None):
        """
        Initialize the PageIndex client tailored for the learning platform.

        Args:
            workspace (str): The directory to store indexed documents.
            api_key (str, optional): OpenAI or LiteLLM API key. Will check environment variables if not provided.
            model (str, optional): The LLM model to use for indexing (summarization, tree generation).
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.client = PageIndexClient(api_key=api_key, model=model, workspace=str(self.workspace))

    def index_document(self, file_path: str, mode: str = "auto") -> str:
        """
        Index a document and return its Document ID.

        Args:
            file_path (str): The path to the PDF or Markdown document.
            mode (str): "auto", "pdf", or "md".

        Returns:
            str: The unique Document ID (UUID string).
        """
        return self.client.index(file_path, mode=mode)

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Get metadata about the indexed document.

        Args:
            doc_id (str): The Document ID.

        Returns:
            dict: A dictionary containing 'doc_name', 'type', 'page_count' (or 'line_count'), etc.
        """
        result_json = self.client.get_document(doc_id)
        return json.loads(result_json)

    def get_structure(self, doc_id: str) -> list:
        """
        Get the hierarchical tree structure of the document without raw text.
        This should be fed into the LLM as the "Table-of-Contents" so it can reason
        and decide which sections/pages to look up.

        Args:
            doc_id (str): The Document ID.

        Returns:
            list: The tree structure as a list of nodes.
        """
        result_json = self.client.get_document_structure(doc_id)
        return json.loads(result_json)

    def get_content(self, doc_id: str, pages: str) -> list:
        """
        Get exact content for specific pages or lines.
        This provides the accurate knowledge reference sources (source tracing)
        needed to verify if the LLM's answers are correct.

        Args:
            doc_id (str): The Document ID.
            pages (str): The page or line ranges. Format examples: "5-7", "3,8", "12".

        Returns:
            list: A list of dicts, each with 'page' and 'content' keys.
        """
        result_json = self.client.get_page_content(doc_id, pages)
        return json.loads(result_json)

if __name__ == "__main__":
    pass

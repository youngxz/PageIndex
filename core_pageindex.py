import json
from pathlib import Path
from typing import Dict, Any, List, Union

from pageindex.client import PageIndexClient

class PageIndexError(Exception):
    """Exception raised for errors in the PageIndex platform wrapper."""
    pass

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

    def _check_error(self, result: dict) -> dict:
        """Helper to check if the returned result contains an error."""
        if 'error' in result:
            raise PageIndexError(result['error'])
        return result

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

    def list_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all indexed documents in the workspace.

        Returns:
            Dict[str, Dict]: A mapping from document ID to its lightweight metadata.
        """
        # _rebuild_meta or _read_meta logic is handled internally, but we can access `documents`
        meta = {}
        for doc_id, doc_info in self.client.documents.items():
            meta[doc_id] = {
                'doc_name': doc_info.get('doc_name', ''),
                'doc_description': doc_info.get('doc_description', ''),
                'type': doc_info.get('type', ''),
                'path': doc_info.get('path', '')
            }
        return meta

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Get metadata about the indexed document.

        Args:
            doc_id (str): The Document ID.

        Returns:
            dict: A dictionary containing 'doc_name', 'type', 'page_count' (or 'line_count'), etc.
        """
        result_json = self.client.get_document(doc_id)
        result = json.loads(result_json)
        return self._check_error(result)

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
        result = json.loads(result_json)
        if isinstance(result, dict) and 'error' in result:
             raise PageIndexError(result['error'])
        return result

    def get_content(self, doc_id: str, pages: str) -> List[Dict[str, Any]]:
        """
        Get exact content for specific pages or lines as structured data.

        Args:
            doc_id (str): The Document ID.
            pages (str): The page or line ranges. Format examples: "5-7", "3,8", "12".

        Returns:
            list: A list of dicts, each with 'page' and 'content' keys.
        """
        result_json = self.client.get_page_content(doc_id, pages)
        result = json.loads(result_json)
        if isinstance(result, dict) and 'error' in result:
             raise PageIndexError(result['error'])
        return result

    def format_reference_source(self, doc_id: str, pages: str) -> str:
        """
        Get formatted reference source text for presenting to the student or checking LLM answers.
        This provides the accurate knowledge reference sources (source tracing)
        needed to verify if the LLM's answers are correct.

        Args:
            doc_id (str): The Document ID.
            pages (str): The page or line ranges (e.g. "5-7").

        Returns:
            str: Human-readable formatted reference string.
        """
        content_items = self.get_content(doc_id, pages)
        meta = self.get_document_metadata(doc_id)
        doc_type = meta.get('type', 'doc')
        label = "Page" if doc_type == "pdf" else "Line"

        formatted_parts = []
        for item in content_items:
            num = item.get('page')
            text = item.get('content', '').strip()
            formatted_parts.append(f"--- [{label} {num}] ---\n{text}")

        return "\n\n".join(formatted_parts)

    def get_llm_tool_instructions(self) -> str:
        """
        Get system prompt instructions for the platform's LLM to guide it
        on how to use the structure and retrieve exact content.

        Returns:
            str: System prompt instructions string.
        """
        return (
            "You are an intelligent knowledge assistant. You have access to document tree structures "
            "and can fetch specific pages or lines for detailed context.\n"
            "1. First, review the document's hierarchical structure (Table-of-Contents).\n"
            "2. Identify the relevant sections based on the user's query.\n"
            "3. Retrieve the exact content using the 'get_content' tool by passing tight ranges (e.g. '5-7'). "
            "For PDF documents, use the physical page numbers (deduced from start_index/end_index). "
            "For Markdown documents, use the line numbers (line_num).\n"
            "4. Base your final answer strictly on the retrieved source content, and cite the exact page/line."
        )

if __name__ == "__main__":
    pass

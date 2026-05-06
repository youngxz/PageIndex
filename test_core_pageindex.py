import os
import json
from core_pageindex import PlatformPageIndex, PageIndexError

def test():
    print("Testing PlatformPageIndex integration...")
    # Initialize the core platform page index wrapper
    platform_idx = PlatformPageIndex(workspace="./test_workspace")

    # Create a dummy markdown file to test indexing
    dummy_md = "dummy.md"
    with open(dummy_md, "w", encoding="utf-8") as f:
        f.write("# Main Topic\n\nThis is the main topic introduction.\n\n## Subtopic 1\n\nDetailed info about subtopic 1.\n\n## Subtopic 2\n\nDetailed info about subtopic 2.\n")

    try:
        # 1. Index document
        print("Indexing dummy document...")
        doc_id = platform_idx.index_document(dummy_md, mode="md")
        print(f"Document ID: {doc_id}")

        # 2. Test Error Handling
        try:
            platform_idx.get_document_metadata("invalid_id")
            assert False, "Should have raised PageIndexError for invalid ID"
        except PageIndexError as e:
            print("Successfully caught expected error:", e)

        # 3. List Documents
        docs = platform_idx.list_documents()
        assert doc_id in docs, "Newly indexed document not found in list."
        print(f"Listed {len(docs)} documents.")

        # 4. Get Metadata
        metadata = platform_idx.get_document_metadata(doc_id)
        print("Metadata:", json.dumps(metadata, indent=2))

        # 5. Get Structure
        structure = platform_idx.get_structure(doc_id)
        print("Structure length:", len(structure))

        # 6. Test Format Reference Source
        formatted_source = platform_idx.format_reference_source(doc_id, "1-7")
        print("Formatted Reference Source:")
        print(formatted_source)
        assert "--- [Line 1] ---" in formatted_source

        # 7. Test LLM Instructions
        instructions = platform_idx.get_llm_tool_instructions()
        assert "Table-of-Contents" in instructions

        print("PlatformPageIndex tests passed successfully.")

    finally:
        # Cleanup
        if os.path.exists(dummy_md):
            os.remove(dummy_md)
        # Clean up test workspace to prevent dirty git trees
        if os.path.exists("./test_workspace"):
            import shutil
            shutil.rmtree("./test_workspace")

if __name__ == "__main__":
    test()
import os
import json
from core_pageindex import PlatformPageIndex

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

        # 2. Get Metadata
        metadata = platform_idx.get_document_metadata(doc_id)
        print("Metadata:", json.dumps(metadata, indent=2))

        # 3. Get Structure
        structure = platform_idx.get_structure(doc_id)
        print("Structure length:", len(structure))
        print("Structure snippet:", json.dumps(structure, indent=2, ensure_ascii=False)[:300], "...")

        # 4. Get Content for source tracing
        content = platform_idx.get_content(doc_id, "1-7")
        print("Content retrieved for source tracing:")
        print(json.dumps(content, indent=2))

        assert len(content) > 0, "Content retrieval failed."
        print("PlatformPageIndex tests passed successfully.")

    finally:
        # Cleanup
        if os.path.exists(dummy_md):
            os.remove(dummy_md)

if __name__ == "__main__":
    test()
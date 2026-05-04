import os
import sys
import time
from pathlib import Path

# Make sure the package is importable from project root
sys.path.insert(0, str(Path(__file__).parent))

from vectorless_rag.index_builder import build_catalog_tree

# ── Paths ─────────────────────────────────────────────────────────────────────
JSON_PATH  = "data/flipkart_fashion_products_dataset.json"
TREE_PATH  = "catalog_index/catalog_tree.json"


def main():
    print("=" * 60)
    print("  Flipkart Vectorless RAG — Index Builder")
    print("  (PageIndex-style hierarchical catalog tree)")
    print("=" * 60)

    # Check data file exists
    if not os.path.exists(JSON_PATH):
        print(f"\n❌  Dataset not found: {JSON_PATH}")
        print(f"    Please place your JSON file at: {os.path.abspath(JSON_PATH)}")
        sys.exit(1)

    file_size_mb = os.path.getsize(JSON_PATH) / (1024 * 1024)
    print(f"\n📂  Dataset : {JSON_PATH}  ({file_size_mb:.1f} MB)")
    print(f"📁  Output  : {TREE_PATH}")
    print()

    start = time.time()
    tree  = build_catalog_tree(JSON_PATH, TREE_PATH)
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"✅  Index built in {elapsed:.1f}s")
    print(f"    Total products : {tree['total_products']:,}")
    print(f"    Categories     : {tree['total_categories']}")
    print()
    print("  Sub-categories per category:")
    for cat, node in tree["categories"].items():
        sc_count = len(node.get("sub_categories", {}))
        print(f"    • {cat}: {sc_count} sub-categories, {node['total']:,} products")
    print()
    print("  Top global brands:")
    print("   ", ", ".join(tree["top_brands"][:10]))
    print("=" * 60)
    print()
    print("🚀  Ready! Run the chatbot with:")
    print("    streamlit run app_vectorless.py")
    print()


if __name__ == "__main__":
    main()
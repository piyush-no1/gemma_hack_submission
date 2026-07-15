import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.taxonomy_builder import build_taxonomy_json
from backend.rag.graph_builder import build_graph_from_taxonomy
from backend.rag.embed_store import build_vector_store
from backend.config import settings

def main():
    print("======================================================================")
    print("AI Surveillance Safety Platform - Seeding Long-Term Memory (RAG/KG)...")
    print("======================================================================")
    
    # 1. Generate taxonomy JSON
    taxonomy_path = "data/taxonomy_entries.json"
    build_taxonomy_json(taxonomy_path)
    
    # 2. Build networkx graph
    graph_path = "data/graph.json"
    build_graph_from_taxonomy(taxonomy_path, graph_path)
    
    # 3. Compute embeddings and write to LanceDB
    build_vector_store(graph_path, settings.VECTORSTORE_ROOT)
    
    print("======================================================================")
    print("Long-term knowledge base seeded successfully!")
    print("======================================================================")

if __name__ == "__main__":
    main()

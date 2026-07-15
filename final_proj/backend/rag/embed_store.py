import os
import json
import urllib.request
import networkx as nx
import lancedb
from ..config import settings

_transformer_model = None

def get_embedding_dim() -> int:
    # sentence-transformers MiniLM returns 384 dims.
    # Ollama embeddinggemma returns 3584 dims.
    if settings.OLLAMA_EMBEDDING_MODEL == "embeddinggemma" and os.getenv("EMBEDDING_PROVIDER") == "ollama":
        return 3584
    return 384

def embed(text: str) -> list[float]:
    global _transformer_model
    
    # Try Ollama if explicitly configured via environment variable
    if os.getenv("EMBEDDING_PROVIDER") == "ollama":
        url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
        payload = {
            "model": settings.OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                return res_data["embedding"]
        except Exception as e:
            print(f"[EmbedStore] Ollama embedding failed: {e}. Falling back to sentence-transformers.")
            
    # Fallback to local sentence-transformers (MiniLM is very small and fast)
    if _transformer_model is None:
        from sentence_transformers import SentenceTransformer
        print("[EmbedStore] Loading local sentence-transformers model ('all-MiniLM-L6-v2')...")
        _transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    return _transformer_model.encode(text).tolist()

def node_to_text(node_id: str, data: dict) -> str:
    node_type = data.get("type", "unknown")
    label = data.get("label", "")
    if node_type == "scene":
        behaviors = ", ".join(data.get("normal_behaviors", []))
        return f"Scene: {label}. Normal behaviors expected: {behaviors}."
    return f"{node_type}: {label}"

def build_vector_store(graph_path: str, vectorstore_root: str):
    print(f"[EmbedStore] Reading graph from {graph_path}...")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph links data not found at {graph_path}")
        
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        
    g = nx.node_link_graph(graph_data)
    
    os.makedirs(vectorstore_root, exist_ok=True)
    db = lancedb.connect(vectorstore_root)
    
    rows = []
    total = g.number_of_nodes()
    print(f"[EmbedStore] Connecting to LanceDB and embedding {total} nodes...")
    
    for i, (node_id, data) in enumerate(g.nodes(data=True)):
        text = node_to_text(node_id, data)
        vector = embed(text)
        
        rows.append({
            "node_id": node_id,
            "type": data["type"],
            "label": data["label"],
            "text": text,
            "vector": vector,
            "collection": "taxonomy"
        })
        
    # Create or overwrite lancedb table
    db.create_table("knowledge_nodes", data=rows, mode="overwrite")
    print(f"[EmbedStore] Vector store indexing completed. Wrote {len(rows)} nodes to table 'knowledge_nodes'.")

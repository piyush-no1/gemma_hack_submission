import os
import json
import time
import networkx as nx
import lancedb
from ..config import settings
from .embed_store import embed

_db_connection = None
_graph_cache = None
_graph_mtime = None

def get_db():
    global _db_connection
    if _db_connection is None:
        _db_connection = lancedb.connect(settings.VECTORSTORE_ROOT)
    return _db_connection

def get_graph(graph_path: str = "./data/graph.json") -> nx.DiGraph:
    global _graph_cache, _graph_mtime
    resolved_path = os.path.abspath(graph_path)
    if not os.path.exists(resolved_path):
        return None
        
    current_mtime = os.path.getmtime(resolved_path)
    if _graph_cache is None or _graph_mtime != current_mtime:
        with open(resolved_path, "r", encoding="utf-8") as f:
            _graph_cache = nx.node_link_graph(json.load(f))
        _graph_mtime = current_mtime
        
    return _graph_cache

def retrieve_anomaly_knowledge(query_text: str, top_k: int = 3, graph_path: str = "./data/graph.json") -> dict:
    """
    Called by the Judge agent node. Combines semantic vector similarity search
    in LanceDB with NetworkX graph structure queries to find UCF anomalies,
    needed evidence, and recommended inspection tools.
    """
    start_time = time.time()
    db = get_db()
    
    table_name = "knowledge_nodes"
    if table_name not in db.table_names():
        return {
            "query": query_text,
            "matches": [],
            "error": "Vector database table not initialized.",
            "retrieval_time_ms": int((time.time() - start_time) * 1000)
        }
        
    table = db.open_table(table_name)
    
    # 1. Vector Search
    query_vector = embed(query_text)
    matches = table.search(query_vector).limit(top_k).to_list()
    
    # 2. Graph Traversal
    g = get_graph(graph_path)
    if g is None:
        return {
            "query": query_text,
            "matches": [],
            "error": "Knowledge graph data not compiled.",
            "retrieval_time_ms": int((time.time() - start_time) * 1000)
        }

    results = []
    for match in matches:
        node_id = match["node_id"]
        entry = {
            "node_id": node_id,
            "matched_node": match["label"],
            "type": match["type"],
            "text": match["text"],
            "distance": float(match.get("_distance", 0.0)),
            "collection": match.get("collection", "taxonomy")
        }
        
        # Pull precise node relations if they exist in networkx
        if g.has_node(node_id):
            if match["type"] == "scene":
                # Successors are anomalies
                anomalies = [g.nodes[n]["label"] for n in g.successors(node_id)
                             if g.nodes[n].get("type") == "anomaly"]
                entry["possible_anomalies"] = anomalies
                entry["normal_behaviors"] = g.nodes[node_id].get("normal_behaviors", [])
                
            elif match["type"] == "anomaly":
                # Successors are evidence and tools
                evidence = [g.nodes[n]["label"] for n in g.successors(node_id)
                            if g.nodes[n].get("type") == "evidence"]
                tools = [g.nodes[n]["label"] for n in g.successors(node_id)
                         if g.nodes[n].get("type") == "tool"]
                entry["evidence_needed"] = evidence
                entry["recommended_tools"] = tools

        results.append(entry)
        
    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "query": query_text,
        "matches": results,
        "retrieval_time_ms": elapsed_ms
    }

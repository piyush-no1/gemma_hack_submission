import os
import json
import networkx as nx

def build_graph_from_taxonomy(taxonomy_path: str, graph_path: str):
    print(f"[GraphBuilder] Reading taxonomy from {taxonomy_path}...")
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Taxonomy entries file not found at {taxonomy_path}")
        
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
        
    g = nx.DiGraph()

    for entry in entries:
        scene_name = entry["scene"]
        scene_id = f"scene::{scene_name}"
        g.add_node(
            scene_id,
            type="scene",
            label=scene_name,
            normal_behaviors=entry["normal_behaviors"]
        )

        for anomaly in entry["anomaly_types"]:
            anomaly_id = f"anomaly::{anomaly}"
            g.add_node(anomaly_id, type="anomaly", label=anomaly)
            g.add_edge(scene_id, anomaly_id, relation="has_anomaly")

            for evidence in entry["evidence_needed"]:
                evidence_id = f"evidence::{evidence}"
                g.add_node(evidence_id, type="evidence", label=evidence)
                g.add_edge(anomaly_id, evidence_id, relation="needs_evidence")

            for tool in entry["recommended_tools"]:
                tool_id = f"tool::{tool}"
                g.add_node(tool_id, type="tool", label=tool)
                g.add_edge(anomaly_id, tool_id, relation="triggers_tool")

    # Serialize using node_link_data
    data = nx.node_link_data(g)
    
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"[GraphBuilder] Wrote graph with {g.number_of_nodes()} nodes, {g.number_of_edges()} edges to {graph_path}")

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    print("[Test] Verifying imports...")
    try:
        from backend.config import settings
        from backend.graph.build_graph import build_graph
        from backend.jobs.job_manager import job_manager
        print("[Test] Imports completed successfully.")
        
        print("[Test] Compiling StateGraph...")
        graph = build_graph()
        print("[Test] StateGraph compiled successfully!")
        
        # Test basic retrieval tool loading
        from backend.rag.retrieval_tool import retrieve_anomaly_knowledge
        print("[Test] Running retrieval test query...")
        res = retrieve_anomaly_knowledge("suspicious person loitering in parking lot at night")
        print(f"[Test] Retrieval success! Matches found: {len(res.get('matches', []))}")
        for match in res.get('matches', [])[:2]:
            print(f"  - Match: {match['matched_node']} ({match['type']})")
            
        print("[Test] Verification checks passed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Test] Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()

import json
from typing import Dict, List
from fastapi import WebSocket

class WebSocketProgressManager:
    def __init__(self):
        # Map job_id to a list of active web socket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        print(f"[WebSocket] Connected client for job {job_id}")

    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        print(f"[WebSocket] Disconnected client for job {job_id}")

    async def broadcast(self, job_id: str, data: dict):
        if job_id not in self.active_connections:
            return
        message = json.dumps(data, default=str)
        # Iterate over a snapshot to avoid issues if disconnect happens mid-broadcast
        dead = []
        for ws in list(self.active_connections.get(job_id, [])):
            try:
                await ws.send_text(message)
            except Exception as e:
                print(f"[WebSocket] Broadcast error for job {job_id}: {e}")
                dead.append(ws)
        # Clean up dead connections
        for ws in dead:
            self.disconnect(job_id, ws)

# Singleton Instance
ws_progress = WebSocketProgressManager()

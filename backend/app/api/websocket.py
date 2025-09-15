"""
WebSocket endpoints for real-time updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass

manager = ConnectionManager()

@router.websocket("/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to TelemetryHealthCare backend"
        }))
        
        # Keep connection alive and handle messages
        while True:
            # Wait for any message from client (heartbeat)
            data = await websocket.receive_text()
            
            # Echo back heartbeat
            if data == "ping":
                await websocket.send_text("pong")
            else:
                # Handle other messages if needed
                logger.info(f"Received: {data}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def broadcast_health_update(patient_id: str, data: dict):
    """Broadcast health update to all connected dashboard clients"""
    message = json.dumps({
        "type": "health_update",
        "patient_id": patient_id,
        "data": data
    })
    await manager.broadcast(message)

async def broadcast_alert(alert: dict):
    """Broadcast alert to all connected dashboard clients"""
    message = json.dumps({
        "type": "alert",
        **alert
    })
    await manager.broadcast(message)
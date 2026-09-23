"""
WebSocket Live Telemetry Streaming Endpoint.
"""

import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from backend.api.session import session_manager

logger = logging.getLogger("backend.api.websocket")
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint broadcasting real-time serializable telemetry packets per frame.
    Clients can also send lightweight JSON commands (e.g. ping/start/stop).
    """
    await session_manager.register_client(websocket)
    logger.info("WebSocket client connected. Total clients: %d", len(session_manager._active_clients))
    
    try:
        while True:
            # We wait for messages but if we don't receive anything for a long time, it's fine.
            # Client usually sends a ping every 5 seconds.
            data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
            command = data.get("command")
            
            if command == "ping":
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
            elif command == "start":
                fps = float(data.get("fps", 15.0))
                resp = await session_manager.start_stream(fps=fps)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "command_response", "command": "start", "result": resp})
            elif command == "stop":
                resp = await session_manager.stop_stream()
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "command_response", "command": "stop", "result": resp})
            elif command == "get_status":
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "status", "status": session_manager.get_status()})
                
    except asyncio.TimeoutError:
        logger.warning("WebSocket client heartbeat timeout (60s without ping). Closing connection.")
        session_manager.unregister_client(websocket)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Heartbeat timeout")
    except WebSocketDisconnect:
        session_manager.unregister_client(websocket)
        logger.info("WebSocket client disconnected normally. Remaining clients: %d", len(session_manager._active_clients))
    except Exception as e:
        session_manager.unregister_client(websocket)
        logger.warning("WebSocket client error/disconnect: %s", str(e))
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1011)
            except Exception:
                pass

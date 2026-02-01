from __future__ import annotations

import asyncio
import random
import time
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


def fake_tree_debug() -> Dict[str, List]:
    candidates = ["cat", "dog", "car"]
    accepted = [True, False, False]
    return {"candidates": candidates, "accepted": accepted, "speedup": 2.4}


@app.websocket("/generate_stream")
async def generate_stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        for token in [" The", " quick", " brown", " fox"]:
            payload = {"text": token, "tree_debug": fake_tree_debug(), "done": False}
            await ws.send_json(payload)
            await asyncio.sleep(0.2)
        await ws.send_json({"text": "", "tree_debug": fake_tree_debug(), "done": True})
    except WebSocketDisconnect:
        return
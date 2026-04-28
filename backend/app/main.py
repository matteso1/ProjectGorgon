"""Project Gorgon — Streaming Inference Backend.

FastAPI + WebSocket backend that runs the actual speculative decoding
engine and streams tokens with tree debug info to the frontend.

Usage:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Project Gorgon",
    description="Speculative Decoding Inference Engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ────────────────────────────────────────────────────
_engine: Optional[Dict[str, Any]] = None
_loading = False


class EngineConfig(BaseModel):
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_heads: int = 4
    top_k: int = 4
    max_new_tokens: int = 128
    checkpoint_path: Optional[str] = None


def _load_engine(config: EngineConfig) -> Dict[str, Any]:
    """Load the backbone + Medusa heads. Called once on first request."""
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import torch
    from gorgon.models.backbone import load_backbone_4bit

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Gorgon] Loading backbone: {config.model_name}")
    model, tokenizer, heads = load_backbone_4bit(
        model_name=config.model_name,
        num_heads=config.num_heads,
        token=hf_token,
        device_map=device,
    )

    # Load trained heads if available
    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
        heads.load_state_dict(ckpt["heads_state_dict"])
        print(f"[Gorgon] Loaded head checkpoint: {config.checkpoint_path}")
    else:
        # Check for default latest checkpoint
        default_ckpt = ROOT / "checkpoints" / "medusa_heads_latest.pt"
        if default_ckpt.exists():
            ckpt = torch.load(str(default_ckpt), map_location="cpu", weights_only=False)
            heads.load_state_dict(ckpt["heads_state_dict"])
            print(f"[Gorgon] Loaded default checkpoint: {default_ckpt}")

    heads = heads.to(device)
    print(f"[Gorgon] Engine ready on {device}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "heads": heads,
        "device": device,
        "config": config,
    }


def get_engine() -> Dict[str, Any]:
    global _engine, _loading
    if _engine is None:
        if _loading:
            raise RuntimeError("Engine is still loading")
        _loading = True
        _engine = _load_engine(EngineConfig())
        _loading = False
    return _engine


# ─── REST endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok" if _engine is not None else "loading",
        "engine": "loaded" if _engine is not None else "not_loaded",
    }


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    if _engine is None:
        return {"status": "not_loaded"}
    cfg = _engine["config"]
    return {
        "model_name": cfg.model_name,
        "num_heads": cfg.num_heads,
        "top_k": cfg.top_k,
        "max_new_tokens": cfg.max_new_tokens,
        "device": _engine["device"],
    }


# ─── WebSocket streaming ─────────────────────────────────────────────

@app.websocket("/generate_stream")
async def generate_stream(ws: WebSocket) -> None:
    await ws.accept()

    try:
        # Wait for prompt from client
        data = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
        prompt = data.get("prompt", "The quick brown fox")
        max_tokens = data.get("max_new_tokens", 128)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        # Fallback: use default prompt for backwards compatibility
        prompt = "The quick brown fox"
        max_tokens = 64

    try:
        engine = get_engine()
    except RuntimeError:
        await ws.send_json({"error": "Engine is loading, try again"})
        await ws.close()
        return

    import sys
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import torch
    from gorgon.inference.gorgon_loop import speculative_generate
    from gorgon.inference.tree_candidates import build_candidate_tree

    model = engine["model"]
    tokenizer = engine["tokenizer"]
    heads = engine["heads"]
    device = engine["device"]
    top_k = engine["config"].top_k

    # Run speculative generation token by token, streaming results
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        generated: List[int] = []
        total_drafted = 0
        total_accepted = 0
        iterations = 0
        start_time = time.perf_counter()

        # Initial forward
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][:, -1:, :]

        # First token
        first_token = int(torch.argmax(outputs.logits[0, -1]).item())
        generated.append(first_token)
        text = tokenizer.decode([first_token])
        elapsed = time.perf_counter() - start_time

        await ws.send_json({
            "text": text,
            "tree_debug": {
                "candidates": [text],
                "accepted": [True],
                "speedup": 0.0,
            },
            "done": False,
        })

        current_ids = torch.cat([
            input_ids,
            torch.tensor([[first_token]], device=device, dtype=input_ids.dtype),
        ], dim=1)

        while len(generated) < max_tokens:
            iterations += 1

            # Draft
            head_device = next(heads.parameters()).device
            head_dtype = next(heads.parameters()).dtype
            hidden_for_heads = hidden.to(device=head_device, dtype=head_dtype)

            tree = build_candidate_tree(heads, hidden_for_heads, top_k=top_k)
            num_candidates = len(tree.tokens)
            total_drafted += num_candidates

            # Verify
            from gorgon.inference.gorgon_loop import _verify_tree_candidates
            accepted_tokens, bonus_token, next_hidden = _verify_tree_candidates(
                model, current_ids, tree,
            )
            total_accepted += len(accepted_tokens)

            new_tokens = accepted_tokens + [bonus_token]
            candidates_text = [tokenizer.decode([t]) for t in tree.tokens[:min(5, len(tree.tokens))].tolist()]
            accepted_flags = [True] * len(accepted_tokens) + [False] * (len(candidates_text) - len(accepted_tokens))
            accepted_flags = accepted_flags[:len(candidates_text)]

            for tok in new_tokens:
                if len(generated) >= max_tokens:
                    break
                generated.append(tok)

                text = tokenizer.decode([tok])
                elapsed = time.perf_counter() - start_time
                tps = len(generated) / elapsed if elapsed > 0 else 0
                acceptance = total_accepted / total_drafted if total_drafted > 0 else 0

                await ws.send_json({
                    "text": text,
                    "tree_debug": {
                        "candidates": candidates_text,
                        "accepted": accepted_flags,
                        "speedup": acceptance,
                        "tokens_per_second": round(tps, 2),
                        "acceptance_rate": round(acceptance, 3),
                    },
                    "done": False,
                })
                await asyncio.sleep(0.01)  # Yield to event loop

                if eos_token_id is not None and tok == eos_token_id:
                    break

            if eos_token_id is not None and generated[-1] == eos_token_id:
                break

            # Update state
            new_t = torch.tensor([new_tokens], device=device, dtype=input_ids.dtype)
            current_ids = torch.cat([current_ids, new_t], dim=1)
            hidden = next_hidden

        # Done
        elapsed = time.perf_counter() - start_time
        tps = len(generated) / elapsed if elapsed > 0 else 0
        await ws.send_json({
            "text": "",
            "tree_debug": {
                "candidates": [],
                "accepted": [],
                "speedup": total_accepted / total_drafted if total_drafted > 0 else 0,
                "tokens_per_second": round(tps, 2),
                "total_tokens": len(generated),
                "total_drafted": total_drafted,
                "total_accepted": total_accepted,
                "iterations": iterations,
            },
            "done": True,
        })

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"error": str(e), "done": True})
        except Exception:
            pass
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

import aiosqlite
import faiss
import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- CONFIGURATION ULTIME ---
class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/tinyllama.gguf")
    VECTOR_DIM: int = 384
    RERANK_THRESHOLD: float = 0.82 
    L1_EXPIRE: int = 604800  # 1 semaine
    DB_PATH: str = "eco_cache/vault_v18.db"
    FAISS_PATH: str = "eco_cache/vault_v18.index"
    
    # Gestion des ressources
    MAX_CONCURRENT_LLM: int = 1  # Crucial pour la VRAM
    LLM_MAX_TOKENS: int = 1024
    CONTEXT_WINDOW: int = 4096
    DEVICE: str = "cuda" if os.getenv("USE_CUDA") else "cpu"
    
    # Sauvegarde périodique
    SAVE_INTERVAL: int = 300 # 5 minutes

settings = Settings()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("EcoLLM-V18-Final")

class ChatQuery(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=2000)

class GlobalEngine:
    def __init__(self):
        self.encoder = None
        self.reranker = None
        self.llm = None
        self.index = None
        self.db = None
        self.redis = None
        self.is_ready = False
        self._index_lock = asyncio.Lock()
        self._inference_sem = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM)
        self.index_dirty = False

    async def setup(self):
        Path("eco_cache").mkdir(exist_ok=True)
        loop = asyncio.get_running_loop()

        logger.info("🚀 Loading Neural Engines...")
        self.encoder = await loop.run_in_executor(None, lambda: SentenceTransformer("all-MiniLM-L6-v2", device=settings.DEVICE))
        self.reranker = await loop.run_in_executor(None, lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=settings.DEVICE))
        
        self.llm = Llama(
            model_path=settings.MODEL_PATH,
            n_ctx=settings.CONTEXT_WINDOW,
            n_gpu_layers=-1 if settings.DEVICE == "cuda" else 0,
            flash_attn=True,
            verbose=False
        )

        # SQLite avec optimisation WAL
        self.db = await aiosqlite.connect(settings.DB_PATH, isolation_level=None)
        await self.db.execute("PRAGMA journal_mode=WAL;")
        await self.db.execute("PRAGMA synchronous=NORMAL;")
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS vault (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                p_hash TEXT UNIQUE, 
                prompt TEXT, 
                resp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # FAISS
        if Path(settings.FAISS_PATH).exists():
            self.index = await loop.run_in_executor(None, faiss.read_index, settings.FAISS_PATH)
        else:
            # HNSW pour la rapidité de recherche
            self.index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(settings.VECTOR_DIM, 32))

        self.redis = redis.from_url("redis://localhost:6379", decode_responses=True)
        
        # Warm-up
        _ = self.encoder.encode("warmup")
        self.is_ready = True
        
        # Lancer le daemon de sauvegarde
        asyncio.create_task(self._autosave_daemon())
        logger.info("💎 System Status: 10/10 READY")

    async def _autosave_daemon(self):
        """Sauvegarde l'index périodiquement si des changements ont eu lieu."""
        while True:
            await asyncio.sleep(settings.SAVE_INTERVAL)
            if self.index_dirty:
                async with self._index_lock:
                    await asyncio.to_thread(faiss.write_index, self.index, settings.FAISS_PATH)
                    self.index_dirty = False
                    logger.info("💾 FAISS index auto-saved.")

    async def get_semantic_match(self, query: str) -> Optional[str]:
        q_vec = await asyncio.to_thread(lambda: self.encoder.encode([query], normalize_embeddings=True).astype("float32"))
        
        async with self._index_lock:
            distances, indices = self.index.search(q_vec, 5)
        
        valid_ids = [int(i) for i in indices[0] if i >= 0]
        if not valid_ids: return None

        # Fetch bulk
        async with self.db.execute(f"SELECT prompt, resp FROM vault WHERE id IN ({','.join('?'*len(valid_ids))})", valid_ids) as cursor:
            rows = await cursor.fetchall()
        
        if not rows: return None
        
        # Reranking pour précision chirurgicale
        pairs = [[query, r[0]] for r in rows]
        scores = await asyncio.to_thread(self.reranker.predict, pairs)
        best_idx = int(np.argmax(scores))
        
        if scores[best_idx] > settings.RERANK_THRESHOLD:
            return rows[best_idx][1]
        return None

engine = GlobalEngine()
app = FastAPI(title="EcoLLM Platinum V18")

@app.on_event("startup")
async def startup():
    await engine.setup()

@app.post("/chat")
async def chat(query: ChatQuery, request: Request, background_tasks: BackgroundTasks):
    start_t = time.perf_counter()
    p_norm = " ".join(query.prompt.lower().strip().split())
    p_hash = hashlib.blake2b(p_norm.encode(), digest_size=16).hexdigest()

    async def stream_logic():
        # --- L1: REDIS ---
        try:
            cached = await engine.redis.get(f"v18:{p_hash}")
            if cached:
                yield f"data: {json.dumps({'src': 'L1', 't': cached, 'ms': (time.perf_counter()-start_t)*1000})}\n\n"
                return
        except Exception: pass

        # --- L2: SEMANTIC ---
        semantic = await engine.get_semantic_match(p_norm)
        if semantic:
            background_tasks.add_task(engine.redis.setex, f"v18:{p_hash}", settings.L1_EXPIRE, semantic)
            yield f"data: {json.dumps({'src': 'L2', 't': semantic, 'ms': (time.perf_counter()-start_t)*1000})}\n\n"
            return

        # --- L3: INFERENCE (AVEC SEMAPHORE) ---
        full_text = ""
        async with engine._inference_sem:
            loop = asyncio.get_running_loop()
            # Template adaptable
            prompt_formatted = f"### System: Etre précis et concis.\n### User: {query.prompt}\n### Assistant: "
            
            tokens = await loop.run_in_executor(
                None, 
                lambda: engine.llm(
                    prompt=prompt_formatted,
                    stream=True, 
                    max_tokens=settings.LLM_MAX_TOKENS,
                    stop=["###", "User:"]
                )
            )

            for chunk in tokens:
                if await request.is_disconnected(): break
                t = chunk["choices"][0]["text"]
                full_text += t
                yield f"data: {json.dumps({'src': 'L3', 't': t})}\n\n"

        if len(full_text.strip()) > 5:
            background_tasks.add_task(finalize_storage, p_hash, p_norm, full_text.strip())

    return StreamingResponse(stream_logic(), media_type="text/event-stream")

async def finalize_storage(h: str, p: str, r: str):
    try:
        # Utilisation de aiosqlite pour ne pas bloquer
        async with engine.db.execute("INSERT OR IGNORE INTO vault (p_hash, prompt, resp) VALUES (?, ?, ?)", (h, p, r)) as cursor:
            if cursor.rowcount > 0:
                rid = cursor.lastrowid
                vec = await asyncio.to_thread(lambda: engine.encoder.encode([p], normalize_embeddings=True).astype("float32"))
                async with engine._index_lock:
                    engine.index.add_with_ids(vec, np.array([rid], dtype=np.int64))
                    engine.index_dirty = True
        
        await engine.redis.setex(f"v18:{h}", settings.L1_EXPIRE, r)
    except Exception as e:
        logger.error(f"Finalize error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)

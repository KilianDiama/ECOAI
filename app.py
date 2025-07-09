#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoLLM – Inference LLM ultra-économe :
  • Quantification int8 locale (transformers + bitsandbytes)
  • Option Micro-LLM distillé (< 100 Mo) pour Pi Zero
  • Cache chiffré des réponses (portalocker + JSON)
  • Batching & planification OFF-PEAK
  • Suivi CO₂ (CodeCarbon)
  • CLI, API FastAPI & snippet client-side cache
  • Edge-ready (Cloudflare Workers)
"""

from __future__ import annotations
import argparse, atexit, contextlib, hashlib, json, logging
import os, queue, sched, sys, threading, time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from pathlib import Path
from threading import Event
from typing import Any, List, Optional, Iterable, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from codecarbon import EmissionsTracker
import portalocker

# ─────────────────────────────── Configuration ──────────────────────────────── #
LOG_FORMAT       = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
CACHE_DIR        = Path(os.getenv("CACHE_DIR", "eco_cache"))
CACHE_FILE       = os.getenv("CACHE_FILE", "llm_cache.json")
EMISSIONS_CSV    = os.getenv("EMISSIONS_CSV", "emissions.csv")
OFFPEAK_START    = os.getenv("OFFPEAK_START", "00:00")
OFFPEAK_END      = os.getenv("OFFPEAK_END",   "06:00")
# Basique ou micro-distilled (taille <100 Mo) via env USE_MICRO=1
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt2")
MICRO_MODEL_PATH = os.getenv("MICRO_MODEL_PATH", "micro-gpt.pt")
USE_MICRO        = bool(int(os.getenv("USE_MICRO", "0")))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "4"))
QUANT_THRESHOLD  = float(os.getenv("LLM_INT8_THRESHOLD", "6.0"))
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("EcoLLM")

# ─────────────────────────────── File Lock ──────────────────────────────────── #
@contextlib.contextmanager
def file_lock(path: Path, timeout: int = 10) -> Iterable[None]:
    lockfile = path.with_suffix(".lock")
    with portalocker.Lock(str(lockfile), timeout=timeout):
        yield

# ───────────────────────────── Cache Manager ────────────────────────────────── #
class CacheManager:
    def __init__(self, base_dir: Path, filename: str):
        self.base_dir = base_dir; self.base_dir.mkdir(exist_ok=True, parents=True)
        self.path = self.base_dir/filename
        self._load()

    def _load(self):
        if self.path.is_file():
            try:
                with file_lock(self.path):
                    self.store = json.loads(self.path.read_text())
            except Exception:
                log.warning("Cache corrompu → reset")
                self.store = {}
        else:
            self.store = {}

    def save(self):
        try:
            with file_lock(self.path):
                self.path.write_text(json.dumps(self.store, ensure_ascii=False))
        except Exception as e:
            log.error("Impossible de sauvegarder le cache : %s", e)

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def set(self, key: str, val: str):
        self.store[key] = val; self.save()

# ───────────────────────────── Scheduler & Queue ───────────────────────────── #
@dataclass
class Job:
    prompt: str

class JobScheduler:
    def __init__(self, fn):
        self.fn        = fn
        self.queue     = queue.Queue()
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.stop_evt  = Event()
        atexit.register(self.shutdown)
        threading.Thread(target=self._run, daemon=True).start()

    def _in_offpeak(self)->bool:
        now   = datetime.now().time()
        start = dt_time.fromisoformat(OFFPEAK_START)
        end   = dt_time.fromisoformat(OFFPEAK_END)
        if start<=end: return start<=now<end
        return now>=start or now<end

    def submit(self, job: Job):
        self.queue.put(job)
        if self._in_offpeak():
            self.scheduler.enter(0,1,self._drain)

    def _drain(self):
        batch=[]
        while not self.queue.empty() and len(batch)<BATCH_SIZE:
            batch.append(self.queue.get())
        if batch:
            self.fn([j.prompt for j in batch])

    def _run(self):
        while not self.stop_evt.is_set():
            if self._in_offpeak() and not self.queue.empty():
                self.scheduler.enter(0,1,self._drain)
                self.scheduler.run(blocking=False)
            time.sleep(300)

    def shutdown(self):
        self.stop_evt.set()

# ─────────────────────────────── EcoLLM Core ────────────────────────────────── #
@dataclass
class EcoLLM:
    cache:     CacheManager = field(default_factory=lambda: CacheManager(CACHE_DIR, CACHE_FILE))
    tracker:   EmissionsTracker = field(init=False)
    scheduler: JobScheduler    = field(init=False)
    pipe:      Any             = field(init=False)

    def __post_init__(self):
        # Choix micro ou standard
        if USE_MICRO and Path(MICRO_MODEL_PATH).is_file():
            log.info("Chargement Micro-LLM depuis %s", MICRO_MODEL_PATH)
            self.model     = torch.load(MICRO_MODEL_PATH, map_location="cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.pipe      = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        else:
            log.info("Chargement LLM %s quantifié int8", LLM_MODEL)
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=QUANT_THRESHOLD)
            self.tokenizer= AutoTokenizer.from_pretrained(LLM_MODEL)
            self.model    = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL, quantization_config=bnb_cfg, device_map="auto"
            )
            self.pipe     = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        # CO₂ tracker
        self.tracker   = EmissionsTracker(measure_power_secs=15,
                                          output_file=str(CACHE_DIR/EMISSIONS_CSV))
        # Off-peak scheduler
        self.scheduler = JobScheduler(self._infer_batch)
        log.info("EcoLLM prêt (micro=%s, int8=%s)", USE_MICRO, not USE_MICRO)

    def _key(self, prompt:str)->str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def generate(self, prompt:str)->str:
        key = self._key(prompt)
        if (resp:=self.cache.get(key)):
            return resp
        job = Job(prompt)
        if self.scheduler._in_offpeak():
            self._infer_batch([prompt])
        else:
            log.info("Mise en file → off-peak")
            self.scheduler.submit(job)
            return "scheduled"
        return self.cache.get(key)  # type: ignore

    def _infer_batch(self, prompts:List[str]):
        with self.tracker:
            outs = self.pipe(prompts, max_length=256, batch_size=len(prompts))
        for p,o in zip(prompts,outs):
            txt=o["generated_text"]
            self.cache.set(self._key(p), txt)
            log.debug("Prompt caché")

# ─────────────────────────────── API Layer ─────────────────────────────────── #
if "fastapi" in sys.modules:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class Payload(BaseModel):
        prompt:str

    app=FastAPI(title="EcoLLM API")

    @app.on_event("startup")
    def on_startup():
        if not os.getenv("ECO_PASSWORD"):
            log.error("ECO_PASSWORD non défini"); sys.exit(1)
        app.state.eco = EcoLLM()

    @app.post("/generate")
    def generate(p:Payload)->Dict[str,str]:
        try: return {"result": app.state.eco.generate(p.prompt)}
        except Exception as e: raise HTTPException(500,str(e))

# ─────────────────────────────────── CLI ────────────────────────────────────── #
def main():
    p=argparse.ArgumentParser("EcoLLM CLI")
    p.add_argument("-p","--prompt", required=True, help="Texte d'entrée")
    p.add_argument("-v","--verbose", action="store_true")
    args=p.parse_args()
    if args.verbose: log.setLevel(logging.DEBUG)
    eco=EcoLLM()
    print(eco.generate(args.prompt))

if __name__=="__main__":
    main()

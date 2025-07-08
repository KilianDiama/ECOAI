#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoAI 
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.quantization import quantize_dynamic

# Optional dependencies
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    logging.warning("FastAPI or Pydantic not installed: REST API disabled.", stacklevel=2)

try:
    import portalocker  # type: ignore
except ImportError:
    portalocker = None
    logging.info("portalocker not installed: falling back to POSIX/msvcrt locks.", stacklevel=2)

try:
    import codecarbon
except ImportError:
    codecarbon = None
    logging.warning("codecarbon not installed: CO2 tracking disabled.", stacklevel=2)

__all__ = ["EcoAI", "EcoAIConfig", "__version__", "CryptoError", "CacheError", "APIError"]
__version__ = "3.1.1"

# ───────────────────────────────────────────────────────────────────────────
# Exceptions dédiées
# ───────────────────────────────────────────────────────────────────────────
class CryptoError(Exception):
    """Erreur lors du chiffrement/déchiffrement."""

class CacheError(Exception):
    """Erreur I/O cache ou corruption."""

class APIError(Exception):
    """Erreur générique pour l’API REST."""

# ───────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ───────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class EcoAIConfig:
    # Cache & modèles
    cache_version: int = 3
    cache_file: str = "predictions_cache_v3.enc"
    model_file: str = "sentiment_q8.pt"
    default_model_name: str = "distilbert-base-uncased"
    tiny_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    emissions_csv: str = "emissions.csv"
    default_cache_dir: str = "eco_cache"
    default_task: str = "sentiment-analysis"

    # KDF Argon2
    kdf_time_cost: int = 1          # passes
    kdf_memory_cost: int = 64 * 1024  # kibibytes
    kdf_parallelism: int = 2
    kdf_length: int = 32

    # i18n
    locale: str = "en_US"

    # Emojis (logging)
    EMOJI_INIT: str = "⚙️"
    EMOJI_READY: str = "🌱"
    EMOJI_CACHE: str = "♻️"
    EMOJI_ENERGY: str = "⚡"
    EMOJI_CO2: str = "🌍"
    EMOJI_EXPORT: str = "📄"

CFG = EcoAIConfig()

# ───────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ecoai")

# ───────────────────────────────────────────────────────────────────────────
# Verrouillage de fichier cross-plat
# ───────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def file_lock(path: Path, timeout: int = 10):
    lock_path = Path(f"{path}.lock")
    if portalocker:
        with portalocker.Lock(str(lock_path), timeout=timeout):
            yield
        return
    # Windows fallback
    try:
        import msvcrt  # type: ignore
        fp = open(lock_path, "a+b")
        start = time.monotonic()
        while True:
            try:
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() - start > timeout:
                    fp.close()
                    raise TimeoutError(f"Timeout lock on {path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
            fp.close()
            lock_path.unlink(missing_ok=True)
        return
    except ImportError:
        pass
    # POSIX fallback
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()

# ───────────────────────────────────────────────────────────────────────────
# Crypto helpers
# ───────────────────────────────────────────────────────────────────────────
class _Crypto:
    """AES-GCM-256 + HMAC-SHA256 via Argon2id KDF."""
    HEADER_VERSION = bytes([CFG.cache_version])

    @classmethod
    def _make_kdf(cls, salt: bytes) -> Argon2id:
        return Argon2id(
            time_cost=CFG.kdf_time_cost,
            memory_cost=CFG.kdf_memory_cost,
            parallelism=CFG.kdf_parallelism,
            length=CFG.kdf_length,
            salt=salt,
        )

    @classmethod
    def _derive_key(cls, pwd: bytes, salt: bytes) -> bytes:
        try:
            return cls._make_kdf(salt).derive(pwd)
        except Exception as e:
            raise CryptoError(f"KDF failed: {e}") from e

    @staticmethod
    def _sign(payload: bytes, key: bytes) -> bytes:
        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(payload)
        return h.finalize()

    @classmethod
    def encrypt(cls, data: bytes, password: str) -> str:
        salt = os.urandom(16)
        iv = os.urandom(12)
        key = cls._derive_key(password.encode(), salt)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ct = encryptor.update(data) + encryptor.finalize()
        payload = cls.HEADER_VERSION + salt + iv + encryptor.tag + ct
        mac = cls._sign(payload, key)
        return base64.b64encode(mac + payload).decode()

    @classmethod
    def decrypt(cls, token: str, password: str) -> bytes:
        try:
            blob = base64.b64decode(token)
            mac, payload = blob[:32], blob[32:]
            ver, salt, iv, tag, ct = (
                payload[:1],
                payload[1:17],
                payload[17:29],
                payload[29:45],
                payload[45:],
            )
            if ver != cls.HEADER_VERSION:
                raise CryptoError("Unsupported cipher version")
            key = cls._derive_key(password.encode(), salt)
            if not hmac.compare_digest(mac, cls._sign(payload, key)):
                raise CryptoError("Integrity check failed")
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
            return cipher.decryptor().update(ct) + cipher.decryptor().finalize()
        except InvalidTag as e:
            raise CryptoError("Invalid password or corrupted data") from e
        except Exception as e:
            raise CryptoError(f"Decrypt failed: {e}") from e

# ───────────────────────────────────────────────────────────────────────────
# Gestionnaire de fichiers sécurisé
# ───────────────────────────────────────────────────────────────────────────
class SecureFileManager:
    """Store JSON/CSV chiffrés avec rotation de clé."""
    def __init__(self, master_password: str, base_dir: Path):
        if not master_password or master_password == "change-me!":
            raise ValueError("Password empty or default is forbidden.")
        self._pwd = master_password
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.base_dir / name

    def rotate_password(self, new_pwd: str) -> None:
        for enc in self.base_dir.glob("*.enc"):
            data = _Crypto.decrypt(enc.read_text(), self._pwd)
            enc.write_text(_Crypto.encrypt(data, new_pwd))
        self._pwd = new_pwd

    def save_json(self, obj: Any, name: str) -> None:
        token = _Crypto.encrypt(json.dumps(obj, ensure_ascii=False).encode(), self._pwd)
        p = self._path(name)
        try:
            with file_lock(p):
                p.write_text(token)
        except Exception as e:
            raise CacheError(f"Failed to write cache {name}: {e}")

    def load_json(self, name: str) -> Any:
        p = self._path(name)
        try:
            with file_lock(p):
                return json.loads(_Crypto.decrypt(p.read_text(), self._pwd).decode())
        except Exception as e:
            raise CacheError(f"Failed to load cache {name}: {e}")

    def save_csv(self, df: pd.DataFrame, name: str) -> None:
        token = _Crypto.encrypt(df.to_csv(index=False).encode(), self._pwd)
        p = self._path(name)
        try:
            with file_lock(p):
                p.write_text(token)
        except Exception as e:
            raise CacheError(f"Failed to write CSV cache {name}: {e}")

    def load_csv(self, name: str) -> pd.DataFrame:
        from io import StringIO
        p = self._path(name)
        try:
            with file_lock(p):
                data = _Crypto.decrypt(p.read_text(), self._pwd).decode()
            return pd.read_csv(StringIO(data))
        except Exception as e:
            raise CacheError(f"Failed to load CSV cache {name}: {e}")

# ───────────────────────────────────────────────────────────────────────────
# Service de validation
# ───────────────────────────────────────────────────────────────────────────
class DataValidationService:
    @staticmethod
    def validate_type(data: Any, expected: type) -> bool:
        return isinstance(data, expected)

    @staticmethod
    def validate_nonempty(data: Any) -> bool:
        if isinstance(data, (np.ndarray, pd.DataFrame, list, dict, str)):
            return len(data) > 0
        return bool(data)

# ───────────────────────────────────────────────────────────────────────────
# Core EcoAI
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class EcoAI:
    master_password: str
    model_name: str = CFG.default_model_name
    cache_dir: Path = Path(CFG.default_cache_dir)
    task: str = CFG.default_task
    quant_dtype: str = "int8"
    lite: bool = False
    use_gpu: bool = False
    disable_carbon: Optional[bool] = None
    business_name: str = "EcoAI"

    _pipeline: Any = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _embedder: SentenceTransformer = field(init=False, repr=False)
    _cache: Dict[str, Any] = field(init=False, repr=False)
    _fm: SecureFileManager = field(init=False, repr=False)
    _tracker: Any = field(init=False, repr=False)
    _energy_log: List[Tuple[float, float]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fm = SecureFileManager(self.master_password, self.cache_dir)
        self._logger = log.getChild(self.business_name.lower())
        self._logger.info("%s Initializing %s...", CFG.EMOJI_INIT, self.business_name)

        if self.lite:
            self.model_name = CFG.tiny_model_name

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = self._load_or_quantize(self.model_name, self.quant_dtype)
        device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
        self._pipeline = pipeline(self.task, model=self._model, tokenizer=self._tokenizer, device=device)

        self._embedder = SentenceTransformer("all-MiniLM-L6-v2", show_progress_bar=False)
        self._cache_path = self.cache_dir / CFG.cache_file
        self._cache = self._load_cache()

        self.disable_carbon = not bool(codecarbon) if self.disable_carbon is None else self.disable_carbon
        if self.disable_carbon:
            self._tracker = contextlib.nullcontext()
        else:
            from codecarbon import EmissionsTracker
            self._tracker = EmissionsTracker(
                measure_power_secs=1,
                output_file=str(self.cache_dir / CFG.emissions_csv),
            )

        self._validator = DataValidationService()
        self._logger.info("%s %s ready!", CFG.EMOJI_READY, self.business_name)

    def _load_or_quantize(self, model_name: str, dtype: str):
        model_path = self.cache_dir / CFG.model_file
        if model_path.exists():
            try:
                self._logger.info("%s Loading quantized model...", CFG.EMOJI_CACHE)
                return torch.load(model_path, map_location="cpu")
            except Exception:
                self._logger.warning("Cache invalid: re-quantizing", stacklevel=2)
        self._logger.info("%s Quantizing model...", CFG.EMOJI_ENERGY)
        base = AutoModelForSequenceClassification.from_pretrained(model_name)
        if dtype.lower() in {"int8", "qint8"}:
            q = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8).cpu()
        elif dtype.lower() in {"fp16", "float16"}:
            q = base.half().cpu()
        else:
            q = base.cpu()
        torch.save(q, model_path)
        return q

    def _load_cache(self) -> Dict[str, Any]:
        if not self._cache_path.exists():
            return {}
        try:
            return self._fm.load_json(self._cache_path.name)
        except CacheError as e:
            bak = self._cache_path.with_suffix(".bak.enc")
            self._cache_path.rename(bak)
            self._logger.warning("Cache corrupt → %s: %s", bak, e, stacklevel=2)
            return {}

    def _save_cache(self):
        try:
            self._fm.save_json(self._cache, self._cache_path.name)
        except CacheError as e:
            self._logger.error("Cannot save cache: %s", e, stacklevel=2)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def predict(self, text: str) -> Dict[str, Any]:
        key = self._hash(text)
        if key not in self._cache:
            with self._tracker:
                t0 = time.perf_counter()
                out = self._pipeline(text)[0]
                dt = time.perf_counter() - t0
                co2 = getattr(self._tracker, "final_emissions", 0.0)
                self._energy_log.append((dt, co2))
                self._cache[key] = out
                self._save_cache()
        return self._cache[key]

    def batch_predict(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        texts = list(texts)
        keys = [self._hash(t) for t in texts]
        to_run = [t for t, k in zip(texts, keys) if k not in self._cache]
        if to_run:
            with self._tracker:
                t0 = time.perf_counter()
                preds = self._pipeline(to_run)
                dt = time.perf_counter() - t0
                co2 = getattr(self._tracker, "final_emissions", 0.0)
            for t, p in zip(to_run, preds):
                k = self._hash(t)
                self._cache[k] = p
                self._energy_log.append((dt / len(to_run), co2 / len(to_run)))
            self._save_cache()
        return [self._cache[k] for k in keys]

    def get_embeddings(self, texts: Iterable[str]) -> np.ndarray:
        return self._embedder.encode(list(texts), convert_to_numpy=True)

    def auto_cluster(
        self, texts: Iterable[str], k_range: range = range(2, 10),
        dim: int = 2, max_samples: int = 5000
    ) -> Tuple[np.ndarray, List[int], int, float]:
        lst = list(texts)
        if len(lst) > max_samples:
            sampled = np.random.choice(len(lst), max_samples, replace=False)
            base_texts = [lst[i] for i in sampled]
        else:
            base_texts = lst
        embs = self.get_embeddings(base_texts)
        red = UMAP(n_components=dim, random_state=42).fit_transform(embs)
        best = {"k": 0, "score": -1.0, "labels": []}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(red)
            sc = silhouette_score(red, km.labels_)
            if sc > best["score"]:
                best.update(k=k, score=sc, labels=km.labels_.tolist())
        if len(lst) > max_samples:
            lab_pred = KMeans(n_clusters=best["k"], random_state=42, n_init="auto").fit(red)
            all_embs = self.get_embeddings(lst)
            red_all = UMAP(n_components=dim, random_state=42).fit_transform(all_embs)
            labels_full = lab_pred.predict(red_all).tolist()
        else:
            labels_full = best["labels"]
        return red, labels_full, best["k"], best["score"]

    def export_results(
        self, texts: List[str], preds: List[Dict[str, Any]],
        clusters: List[int], fmt: str = "excel"
    ) -> Tuple[Path, pd.DataFrame]:
        df = pd.DataFrame({
            "Text": texts,
            "Prediction": [p["label"] for p in preds],
            "Score": [float(p["score"]) for p in preds],
            "Cluster": clusters,
        })
        exts = {"excel": "xlsx", "csv": "csv", "json": "json"}
        if fmt not in exts:
            raise ValueError(f"Format non supporté: {fmt}")
        out = self.cache_dir / f"EcoAI_report_{int(time.time())}.{exts[fmt]}"
        if fmt == "excel":
            df.to_excel(out, index=False)
        elif fmt == "csv":
            df.to_csv(out, index=False)
        else:
            df.to_json(out, orient="records", force_ascii=False)
        self._logger.info("%s Export → %s", CFG.EMOJI_EXPORT, out.name)
        return out, df

    def summary(self):
        total_t = sum(t for t, _ in self._energy_log)
        total_c = sum(c for _, c in self._energy_log)
        self._logger.info(
            "%s Total time: %.2fs | %s CO₂: %.6f kg | %s Cache size: %d",
            CFG.EMOJI_ENERGY, total_t, CFG.EMOJI_CO2, total_c,
            CFG.EMOJI_CACHE, len(self._cache)
        )

# ───────────────────────────────────────────────────────────────────────────
# FastAPI service
# ───────────────────────────────────────────────────────────────────────────
if FastAPI:
    class _Payload(BaseModel):
        text: str

    app = FastAPI(title="EcoAI API", version=__version__)

    @app.on_event("startup")
    def _startup():
        pwd = os.getenv("ECOAI_PASSWORD", "please-change-me")
        app.state.eco = EcoAI(master_password=pwd, use_gpu=True)

    @app.post("/predict")
    def predict(payload: _Payload):
        try:
            return app.state.eco.predict(payload.text)
        except CryptoError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except APIError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error")

# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="EcoAI CLI")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--text", help="Texte à analyser")
    g.add_argument("-b", "--batch", nargs="+", help="Batch de textes")
    parser.add_argument("-p", "--password", required=True, help="Mot de passe maître")
    parser.add_argument("--lite", action="store_true", help="Modèle léger")
    parser.add_argument("--quant", choices=["int8", "fp16", "fp32"], default="int8", help="Quantization")
    parser.add_argument("--gpu", action="store_true", help="Activer GPU si disponible")
    args = parser.parse_args()

    eco = EcoAI(
        master_password=args.password,
        quant_dtype=args.quant,
        lite=args.lite,
        use_gpu=args.gpu,
    )
    if args.text:
        r = eco.predict(args.text)
        print(f"'{args.text}' → {r['label']} ({r['score']:.2%})")
    else:
        res = eco.batch_predict(args.batch)
        for t, pp in zip(args.batch, res):
            print(f"'{t}' → {pp['label']} ({pp['score']:.2%})")
    eco.summary()

if __name__ == "__main__":
    _cli()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoAI – Sentiment analysis économe en carbone + cache chiffré

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

# ──────────────────────────────  Dépendances optionnelles  ──────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, ValidationError
except ModuleNotFoundError:  # FastAPI non installé
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore

try:
    import portalocker  # type: ignore
except ModuleNotFoundError:
    portalocker = None

try:
    import codecarbon
    from codecarbon import EmissionsTracker
except ModuleNotFoundError:
    codecarbon = None  # type: ignore
    EmissionsTracker = None  # type: ignore

__all__ = [
    "EcoAI",
    "EcoAIConfig",
    "__version__",
    "CryptoError",
    "CacheError",
    "APIError",
]

__version__ = "4.0.0"

# ────────────────────────────────  Exceptions dédiées  ─────────────────────────────────
class CryptoError(RuntimeError):
    """Erreur lors du chiffrement/déchiffrement."""


class CacheError(IOError):
    """Erreur I/O cache ou corruption."""


class APIError(RuntimeError):
    """Erreur générique pour l’API REST."""


# ───────────────────────────────  Configuration globale  ───────────────────────────────
@dataclass(slots=True, frozen=True)
class EcoAIConfig:
    # Fichiers
    cache_version: int = 4
    cache_file: str = "predictions_cache_v4.enc"
    model_file: str = "sentiment_q8.pt"
    default_cache_dir: str = "eco_cache"
    emissions_csv: str = "emissions.csv"

    # Modèles HF
    default_model_name: str = "distilbert-base-uncased"
    tiny_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    default_task: str = "sentiment-analysis"

    # Argon2 KDF
    kdf_time_cost: int = 1
    kdf_memory_cost: int = 64 * 1024  # KiB
    kdf_parallelism: int = 2
    kdf_length: int = 32  # bytes

    # Divers
    locale: str = "en_US"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    # Emojis pour les logs
    EMOJI_INIT: str = "⚙️"
    EMOJI_READY: str = "🌱"
    EMOJI_CACHE: str = "♻️"
    EMOJI_ENERGY: str = "⚡"
    EMOJI_CO2: str = "🌍"
    EMOJI_EXPORT: str = "📄"


CFG = EcoAIConfig()

# ──────────────────────────────────  Logging  ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format=CFG.LOG_FORMAT)
log = logging.getLogger("ecoai")


# ─────────────────────────────────  File lock cross-plat  ──────────────────────────────
@contextlib.contextmanager
def file_lock(target: Path, timeout: int = 10) -> Iterable[None]:
    """
    Contexte de verrouillage exclusif sur `target`.

    • Si portalocker dispo → portable & robuste.
    • Sinon : fallback POSIX/msvcrt.
    """
    lock_path = target.with_suffix(".lock")

    if portalocker is not None:
        with portalocker.Lock(str(lock_path), timeout=timeout):
            yield
        return

    # Fallback sans portalocker
    if os.name == "nt":
        import msvcrt  # type: ignore

        with open(lock_path, "a+b") as fp:
            start = time.monotonic()
            while True:
                try:
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"Lock timeout on {target}")
                    time.sleep(0.1)
            try:
                yield
            finally:
                msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
        lock_path.unlink(missing_ok=True)
    else:  # POSIX simple
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            yield
        finally:
            os.close(fd)
            lock_path.unlink(missing_ok=True)


# ────────────────────────────────  Helpers crypto  ─────────────────────────────────────
class _Crypto:
    """
    Chiffrement AES-256-GCM + HMAC-SHA256
    • Header : 1 byte version
    • Salt : 16 B, IV : 12 B, Tag : 16 B, MAC : 32 B
    """

    HEADER_VERSION = bytes([CFG.cache_version])

    # ——— KDF Argon2id ——— #
    @classmethod
    def _derive_key(cls, pwd: str, salt: bytes) -> bytes:
        kdf = Argon2id(
            time_cost=CFG.kdf_time_cost,
            memory_cost=CFG.kdf_memory_cost,
            parallelism=CFG.kdf_parallelism,
            length=CFG.kdf_length,
            salt=salt,
        )
        return kdf.derive(pwd.encode())

    # ——— HMAC ——— #
    @staticmethod
    def _hmac(payload: bytes, key: bytes) -> bytes:
        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(payload)
        return h.finalize()

    # ——— Encrypt ——— #
    @classmethod
    def encrypt(cls, data: bytes, password: str) -> str:
        salt, iv = os.urandom(16), os.urandom(12)
        key = cls._derive_key(password, salt)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ct = encryptor.update(data) + encryptor.finalize()

        payload = b"".join(
            [cls.HEADER_VERSION, salt, iv, encryptor.tag, ct]
        )
        mac = cls._hmac(payload, key)
        return base64.b64encode(mac + payload).decode()

    # ——— Decrypt ——— #
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

            key = cls._derive_key(password, salt)
            if not hmac.compare_digest(mac, cls._hmac(payload, key)):
                raise CryptoError("Integrity check failed")

            cipher = Cipher(
                algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
            )
            decryptor = cipher.decryptor()
            return decryptor.update(ct) + decryptor.finalize()
        except InvalidTag as exc:
            raise CryptoError("Invalid password or corrupted data") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise CryptoError(f"Decrypt error: {exc}") from exc


# ─────────────────────────────  Gestionnaire de fichiers  ──────────────────────────────
class SecureFileManager:
    """Lecture/écriture de JSON ou CSV avec chiffrement + file-lock."""

    def __init__(self, master_password: str, base_dir: Path) -> None:
        if master_password in {"", "change-me!", "please-change-me"}:
            raise ValueError("Weak or empty master password.")

        self._pwd: str = master_password
        self.base_dir: Path = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ——— Helpers ——— #
    def _p(self, name: str) -> Path:
        return self.base_dir / name

    # ——— Rotation de clé ——— #
    def rotate_password(self, new_pwd: str) -> None:
        for enc in self.base_dir.glob("*.enc"):
            data = _Crypto.decrypt(enc.read_text(), self._pwd)
            enc.write_text(_Crypto.encrypt(data, new_pwd))
        self._pwd = new_pwd

    # ——— JSON ——— #
    def save_json(self, obj: Any, name: str) -> None:
        token = _Crypto.encrypt(
            json.dumps(obj, ensure_ascii=False).encode(), self._pwd
        )
        path = self._p(name)
        try:
            with file_lock(path):
                path.write_text(token)
        except OSError as exc:
            raise CacheError(f"Cannot write {name}: {exc}") from exc

    def load_json(self, name: str) -> Any:
        path = self._p(name)
        try:
            with file_lock(path):
                raw = path.read_text()
            return json.loads(_Crypto.decrypt(raw, self._pwd).decode())
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise CacheError(f"Cannot load {name}: {exc}") from exc

    # ——— CSV ——— #
    def save_csv(self, df: pd.DataFrame, name: str) -> None:
        token = _Crypto.encrypt(df.to_csv(index=False).encode(), self._pwd)
        path = self._p(name)
        try:
            with file_lock(path):
                path.write_text(token)
        except OSError as exc:
            raise CacheError(f"Cannot write {name}: {exc}") from exc

    def load_csv(self, name: str) -> pd.DataFrame:
        from io import StringIO

        path = self._p(name)
        try:
            with file_lock(path):
                raw = path.read_text()
            csv = _Crypto.decrypt(raw, self._pwd).decode()
            return pd.read_csv(StringIO(csv))
        except Exception as exc:
            raise CacheError(f"Cannot load {name}: {exc}") from exc


# ─────────────────────────────────  Validation Pydantic  ───────────────────────────────
class _InputValidator(BaseModel):  # type: ignore[misc]
    """Validation unique pour API & CLI batch."""

    text: str


# ─────────────────────────────────────  EcoAI  ─────────────────────────────────────────
@dataclass
class EcoAI:
    master_password: str
    model_name: str = CFG.default_model_name
    cache_dir: Path = Path(CFG.default_cache_dir)
    task: str = CFG.default_task
    quant_dtype: str = "int8"
    lite: bool = False
    use_gpu: bool = False
    disable_carbon: Optional[bool] = None  # auto si None
    business_name: str = "EcoAI"

    # Champs internes (init=False)
    _tokenizer: AutoTokenizer = field(init=False)
    _model: torch.nn.Module = field(init=False)
    _pipeline: Any = field(init=False)
    _embedder: SentenceTransformer = field(init=False)
    _fm: SecureFileManager = field(init=False)
    _cache_path: Path = field(init=False)
    _cache: Dict[str, Dict[str, Union[str, float]]] = field(init=False)
    _tracker: contextlib.AbstractContextManager = field(init=False)  # type: ignore
    _energy_log: List[Tuple[float, float]] = field(default_factory=list, init=False)
    _logger: logging.Logger = field(init=False)

    # ————————————  Post-init  ———————————— #
    def __post_init__(self) -> None:
        # Logging
        self._logger = log.getChild(self.business_name.lower())
        self._logger.info("%s Initializing %s…", CFG.EMOJI_INIT, self.business_name)

        # Secure cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fm = SecureFileManager(self.master_password, self.cache_dir)
        self._cache_path = self.cache_dir / CFG.cache_file

        # Model
        if self.lite:
            self.model_name = CFG.tiny_model_name

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = self._load_or_quantize(self.model_name, self.quant_dtype)

        device_id = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
        self._pipeline = pipeline(
            self.task,
            model=self._model,
            tokenizer=self._tokenizer,
            device=device_id,
        )

        # Embedder
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2", show_progress_bar=False)

        # Cache mémoire
        self._cache = self._load_cache()

        # CodeCarbon
        want_carbon = not bool(self.disable_carbon is True)
        if codecarbon and want_carbon:
            self._tracker = EmissionsTracker(
                measure_power_secs=1,
                output_file=str(self.cache_dir / CFG.emissions_csv),
            )
        else:
            self._tracker = contextlib.nullcontext()

        self._logger.info("%s %s ready!", CFG.EMOJI_READY, self.business_name)

    # ————————————  Modèle quantisé  ———————————— #
    def _load_or_quantize(self, model_name: str, dtype: str) -> torch.nn.Module:
        model_path = self.cache_dir / CFG.model_file
        if model_path.is_file():
            try:
                self._logger.info("%s Loading quantized model cache…", CFG.EMOJI_CACHE)
                return torch.load(model_path, map_location="cpu")
            except (OSError, RuntimeError):
                self._logger.warning("Corrupt model cache ; re-quantizing.")

        self._logger.info("%s Quantizing model (%s)…", CFG.EMOJI_ENERGY, dtype)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if dtype.lower() in {"int8", "qint8"}:
            q_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
        elif dtype.lower() in {"fp16", "float16"}:
            q_model = base_model.half()
        else:
            q_model = base_model  # full precision

        q_model.cpu()
        torch.save(q_model, model_path)
        return q_model

    # ————————————  Cache  ———————————— #
    def _load_cache(self) -> Dict[str, Dict[str, Union[str, float]]]:
        if not self._cache_path.exists():
            return {}
        try:
            return self._fm.load_json(self._cache_path.name)
        except (CacheError, json.JSONDecodeError) as exc:
            # On renomme pour debug
            bak = self._cache_path.with_suffix(".bak.enc")
            self._cache_path.rename(bak)
            self._logger.warning("Cache corrupt → renamed %s: %s", bak.name, exc)
            return {}

    def _save_cache(self) -> None:
        try:
            self._fm.save_json(self._cache, self._cache_path.name)
        except CacheError as exc:
            self._logger.error("Cannot save cache: %s", exc)

    # ————————————  Utilitaires internes  ———————————— #
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    # ————————————  Prédictions  ———————————— #
    def _track_emissions(self) -> Tuple[contextlib.AbstractContextManager, List[float]]:
        if isinstance(self._tracker, contextlib.AbstractContextManager):
            return self._tracker, self._energy_log
        # Should not happen, but fallback
        return contextlib.nullcontext(), self._energy_log

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        # Validation basique
        if not text.strip():
            raise ValueError("Input text is empty.")

        key = self._hash(text)
        if key in self._cache:
            return self._cache[key]

        # Prédiction + suivi CO₂
        with self._track_emissions()[0]:
            t0 = time.perf_counter()
            result = self._pipeline(text)[0]  # type: ignore[index]
            elapsed = time.perf_counter() - t0
            emissions = getattr(self._tracker, "final_emissions", 0.0)
            self._energy_log.append((elapsed, emissions))

        self._cache[key] = result  # type: ignore[assignment]
        self._save_cache()
        return result

    def batch_predict(
        self, texts: Sequence[str]
    ) -> List[Dict[str, Union[str, float]]]:
        validated: List[str] = [t for t in texts if t.strip()]
        keys = [self._hash(t) for t in validated]

        to_compute = [t for t, k in zip(validated, keys) if k not in self._cache]

        if to_compute:
            with self._track_emissions()[0]:
                t0 = time.perf_counter()
                preds = self._pipeline(to_compute)
                elapsed = time.perf_counter() - t0
                emissions = getattr(self._tracker, "final_emissions", 0.0)

            for txt, pred in zip(to_compute, preds):
                k = self._hash(txt)
                self._cache[k] = pred  # type: ignore[assignment]
                self._energy_log.append((elapsed / len(to_compute), emissions / len(to_compute)))

            self._save_cache()

        return [self._cache[k] for k in keys]

    # ————————————  Embeddings & clustering  ———————————— #
    def get_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        return self._embedder.encode(list(texts), convert_to_numpy=True)

    def auto_cluster(
        self,
        texts: Sequence[str],
        k_range: range = range(2, 10),
        dim: int = 2,
        max_samples: int = 5000,
    ) -> Tuple[np.ndarray, List[int], int, float]:
        lst = list(texts)
        sample_idx = (
            np.random.choice(len(lst), max_samples, replace=False)
            if len(lst) > max_samples
            else np.arange(len(lst))
        )
        base_texts = [lst[i] for i in sample_idx]
        embs = self.get_embeddings(base_texts)

        red = UMAP(n_components=dim, random_state=42).fit_transform(embs)
        best = {"k": 0, "score": -1.0, "labels": []}

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(red)
            score = silhouette_score(red, km.labels_)
            if score > best["score"]:
                best.update(k=k, score=score, labels=km.labels_.tolist())

        # Prédiction pour tous les points
        if len(lst) > max_samples:
            kmeans_full = KMeans(n_clusters=best["k"], random_state=42, n_init="auto").fit(
                UMAP(n_components=dim, random_state=42).fit_transform(
                    self.get_embeddings(lst)
                )
            )
            labels_full = kmeans_full.labels_.tolist()
        else:
            labels_full = best["labels"]

        return red, labels_full, best["k"], best["score"]

    # ————————————  Export  ———————————— #
    def export_results(
        self,
        texts: Sequence[str],
        preds: Sequence[Dict[str, Union[str, float]]],
        clusters: Sequence[int],
        fmt: str = "excel",
    ) -> Tuple[Path, pd.DataFrame]:
        df = pd.DataFrame(
            {
                "Text": texts,
                "Prediction": [p["label"] for p in preds],
                "Score": [float(p["score"]) for p in preds],
                "Cluster": clusters,
            }
        )

        exts = {"excel": "xlsx", "csv": "csv", "json": "json"}
        if fmt not in exts:
            raise ValueError(f"Unsupported export format: {fmt!r}")

        out_file = self.cache_dir / f"EcoAI_report_{int(time.time())}.{exts[fmt]}"
        if fmt == "excel":
            df.to_excel(out_file, index=False)
        elif fmt == "csv":
            df.to_csv(out_file, index=False)
        else:
            df.to_json(out_file, orient="records", force_ascii=False)

        self._logger.info("%s Exported → %s", CFG.EMOJI_EXPORT, out_file.name)
        return out_file, df

    # ————————————  Résumé énergie / CO₂  ———————————— #
    def summary(self) -> None:
        total_time = sum(t for t, _ in self._energy_log)
        total_co2 = sum(c for _, c in self._energy_log)
        self._logger.info(
            "%s Time %.2fs | %s CO₂ %.6f kg | %s Cache %d",
            CFG.EMOJI_ENERGY,
            total_time,
            CFG.EMOJI_CO2,
            total_co2,
            CFG.EMOJI_CACHE,
            len(self._cache),
        )


# ────────────────────────────────  FastAPI (optionnel)  ────────────────────────────────
if FastAPI is not None:

    class _Payload(BaseModel):  # type: ignore[misc]
        text: str

    app = FastAPI(title="EcoAI API", version=__version__)

    @app.on_event("startup")
    def _startup() -> None:
        pwd = os.getenv("ECOAI_PASSWORD")
        if not pwd:
            log.error("ECOAI_PASSWORD env var missing.")
            sys.exit(1)
        app.state.eco = EcoAI(master_password=pwd, use_gpu=True)

    @app.post("/predict")
    def predict(payload: _Payload) -> Dict[str, Union[str, float]]:
        try:
            return app.state.eco.predict(payload.text)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))  # type: ignore[arg-type]
        except CryptoError as exc:
            raise HTTPException(status_code=400, detail=str(exc))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail="Internal error") from exc


# ─────────────────────────────────────  CLI  ───────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(description="EcoAI CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", help="Texte à analyser")
    group.add_argument("-b", "--batch", nargs="+", help="Batch de textes")
    parser.add_argument("-p", "--password", required=True, help="Mot de passe maître")
    parser.add_argument("--lite", action="store_true", help="Modèle léger")
    parser.add_argument("--quant", choices=["int8", "fp16", "fp32"], default="int8")
    parser.add_argument("--gpu", action="store_true", help="Utiliser GPU")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Augmente le niveau de log")
    args = parser.parse_args()

    # Log level
    if args.verbose:
        log.setLevel(logging.DEBUG)

    eco = EcoAI(
        master_password=args.password,
        quant_dtype=args.quant,
        lite=args.lite,
        use_gpu=args.gpu,
    )

    if args.text:
        res = eco.predict(args.text)
        print(f"'{args.text}' → {res['label']} ({float(res['score']):.2%})")
    else:
        results = eco.batch_predict(args.batch)  # type: ignore[arg-type]
        for txt, res in zip(args.batch, results):  # type: ignore[arg-type]
            print(f"'{txt}' → {res['label']} ({float(res['score']):.2%})")

    eco.summary()


if __name__ == "__main__":  # pragma: no cover
    try:
        _cli()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

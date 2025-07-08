
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from torch.quantization import quantize_dynamic
import torch

try:
    import portalocker  # type: ignore
except ImportError:  # graceful fallback
    portalocker = None

try:
    import msvcrt  # type: ignore  # noqa: E402  (only on Windows)
except ImportError:  # non-Windows
    msvcrt = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class EcoAIConfig:
    cache_version: int = 1
    cache_file: str = "predictions_cache_v1.enc"
    model_file: str = "sentiment_q8.pt"
    default_model_name: str = "distilbert-base-uncased"
    tiny_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    emissions_csv: str = "emissions.csv"
    default_cache_dir: str = "eco_cache"
    default_task: str = "sentiment-analysis"

    # emojis centralisés (logging uniformisé)
    EMOJI_INIT = "⚙️"
    EMOJI_READY = "🌱"
    EMOJI_CACHE = "♻️"
    EMOJI_ENERGY = "⚡"
    EMOJI_CO2 = "🌍"
    EMOJI_EXPORT = "📄"


CFG = EcoAIConfig()

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ecoai")


# ──────────────────────────────────────────────────────────────────────────────
# Fichier lock cross-platform
# ──────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def file_lock(path: Path, timeout: int = 10):
    """
    Context manager de verrou fichier.
    • portalocker si installé
    • Windows : msvcrt.locking
    • POSIX fallback : fichier .lock avec os.open + EEXIST
    """
    lock_path = Path(f"{path}.lock")

    if portalocker:
        with portalocker.Lock(lock_path, timeout=timeout):
            yield
        return

    if msvcrt:  # Windows sans portalocker
        fp = open(lock_path, "a+b")
        start = time.monotonic()
        while True:
            try:
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Lock timeout on {path}") from None
                time.sleep(0.1)
        try:
            yield
        finally:
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
            fp.close()
            lock_path.unlink(missing_ok=True)
        return

    # POSIX simple (non-atomique, mais dernier recours)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# SecureFileManager
# ──────────────────────────────────────────────────────────────────────────────
class SecureFileManager:
    """Lecture/écriture JSON & CSV chiffrés AES-GCM 256 bits (header versionné)."""

    HEADER_VERSION = bytes([CFG.cache_version])

    def __init__(self, master_password: str, base_dir: str | Path):
        self._pwd = master_password.encode()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------- crypto primitives
    def _generate_salt(self) -> bytes:
        return os.urandom(16)

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
            backend=default_backend(),
        )
        return kdf.derive(self._pwd)

    # ---------- public API
    def encrypt_data(self, data: bytes) -> str:
        salt = self._generate_salt()
        key = self._derive_key(salt)
        iv = os.urandom(12)

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()                     # ⇦ un seul encryptor
        ct = encryptor.update(data) + encryptor.finalize()
        tag = encryptor.tag

        payload = self.HEADER_VERSION + salt + iv + tag + ct
        return base64.b64encode(payload).decode()

    def decrypt_data(self, b64: str) -> bytes:
        raw = base64.b64decode(b64)
        ver, salt, iv, tag, ct = raw[:1], raw[1:17], raw[17:29], raw[29:45], raw[45:]

        if ver != self.HEADER_VERSION:
            raise ValueError("Unsupported cipher version")

        key = self._derive_key(salt)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        try:
            return decryptor.update(ct) + decryptor.finalize()
        except InvalidTag as exc:
            raise ValueError("Invalid password or corrupted data") from exc

    # ---------- helpers
    def _f(self, fn: str | Path) -> Path:
        return self.base_dir / fn

    def _write(self, text: str, fn: str | Path) -> None:
        with open(self._f(fn), "w", encoding="utf-8") as fp:
            fp.write(text)

    def _read(self, fn: str | Path) -> str:
        with open(self._f(fn), "r", encoding="utf-8") as fp:
            return fp.read()

    # ---------- JSON/CSV helpers
    def save_encrypted_json(self, data: Any, file_name: str | Path) -> None:
        txt = self.encrypt_data(json.dumps(data, ensure_ascii=False).encode())
        with file_lock(self._f(file_name)):
            self._write(txt, file_name)

    def load_encrypted_json(self, file_name: str | Path) -> Any:
        with file_lock(self._f(file_name)):
            payload = self._read(file_name)
        return json.loads(self.decrypt_data(payload).decode())

    def save_encrypted_csv(self, df: pd.DataFrame, file_name: str | Path) -> None:
        txt = self.encrypt_data(df.to_csv(index=False).encode())
        with file_lock(self._f(file_name)):
            self._write(txt, file_name)

    def load_encrypted_csv(self, file_name: str | Path) -> pd.DataFrame:
        from io import StringIO

        with file_lock(self._f(file_name)):
            payload = self._read(file_name)
        return pd.read_csv(StringIO(self.decrypt_data(payload).decode()))


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────
class DataValidationService:
    """Validations simples (type + non-vide)."""

    @staticmethod
    def validate_type(data: Any, expected: type) -> bool:
        return isinstance(data, expected)

    @staticmethod
    def validate_nonempty(data: Any) -> bool:
        if isinstance(data, np.ndarray):
            return data.size > 0
        if isinstance(data, (pd.DataFrame, list, dict, str)):
            return len(data) > 0
        return bool(data)


# ──────────────────────────────────────────────────────────────────────────────
# InteractionLogger
# ──────────────────────────────────────────────────────────────────────────────
class InteractionLogger:
    """Historique par utilisateur (optionnellement chiffré)."""

    def __init__(self, storage: Path, password: str, encrypted: bool = True):
        self.storage = storage
        self.encrypted = encrypted
        self.secure = SecureFileManager(password, storage)
        self.storage.mkdir(parents=True, exist_ok=True)

    def _file(self, uid: str) -> str:
        return f"{uid}_interactions.json"

    def log(self, uid: str, message: str, result: Dict[str, Any]) -> None:
        entry = {
            "user_id": uid,
            "message": message,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        name = self._file(uid)
        try:
            if self.encrypted and self.secure._f(name).exists():
                data = self.secure.load_encrypted_json(name)
            else:
                data = []
        except Exception as exc:
            log.warning("Logger read error → reset: %s", exc)
            data = []

        data.append(entry)
        if self.encrypted:
            self.secure.save_encrypted_json(data, name)
        else:
            self.secure._f(name).write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ──────────────────────────────────────────────────────────────────────────────
# EcoAI core
# ──────────────────────────────────────────────────────────────────────────────
class EcoAI:
    """Pipeline Sentiment → Cache → Embeddings → Clustering → Export."""

    def __init__(
        self,
        *,
        model_name: str = CFG.default_model_name,
        cache_dir: str | Path = CFG.default_cache_dir,
        task: str = CFG.default_task,
        master_password: str | None = None,
        business_name: str = "EcoAI",
    ):
        self.task = task
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.master_password = master_password or os.getenv("ECOAI_PASSWORD", "change-me!")
        self.fm = SecureFileManager(self.master_password, self.cache_dir)

        # ── logging
        self.logger = logging.getLogger(f"ecoai.{business_name}")
        self.logger.info("%s  Initialisation %s…", CFG.EMOJI_INIT, business_name)

        # ── model & pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_or_quantize(model_name)
        self.pipe = pipeline(task, model=self.model, tokenizer=self.tokenizer, device=-1)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # ── cache
        self.cache_path = self.cache_dir / CFG.cache_file
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

        # ── misc
        self.validator = DataValidationService()
        from codecarbon import EmissionsTracker  # local import (= optional dep)

        self.tracker = EmissionsTracker(
            measure_power_secs=1,
            output_file=str(self.cache_dir / CFG.emissions_csv),
        )
        self.energy_log: list[tuple[float, float]] = []

        self.interactions = InteractionLogger(
            storage=self.cache_dir / "interactions", password=self.master_password
        )

        self.logger.info("%s  %s prêt !", CFG.EMOJI_READY, business_name)

    # ─────────────────────────────────────────────────────────── internal utils
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def _load_or_quantize(self, model_name: str):
        model_path = self.cache_dir / CFG.model_file
        if model_path.exists():
            try:
                self.logger.info("%s  Chargement modèle quantifié …", CFG.EMOJI_CACHE)
                return torch.load(model_path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Échec cache modèle : %s → requantification", exc)

        self.logger.info("%s  Quantization dynamique…", CFG.EMOJI_ENERGY)
        base = AutoModelForSequenceClassification.from_pretrained(model_name)
        q_model = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8).cpu()
        torch.save(q_model, model_path)
        return q_model

    # cache
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path.exists():
            try:
                return self.fm.load_encrypted_json(self.cache_path.name)  # type: ignore[arg-type]
            except Exception as exc:
                self.logger.warning("Cache corrompu ou mauvais mot de passe : %s", exc)
        return {}

    def _save_cache(self) -> None:
        try:
            self.fm.save_encrypted_json(self.cache, self.cache_path.name)
        except Exception as exc:
            self.logger.error("Impossible de sauvegarder le cache : %s", exc)

    # micro-optim pipeline
    def _select_pipeline(self, text: str):
        if len(text.split()) < 6:
            tiny = CFG.tiny_model_name
            return pipeline(
                self.task,
                model=AutoModelForSequenceClassification.from_pretrained(tiny),
                tokenizer=AutoTokenizer.from_pretrained(tiny),
                device=-1,
            )
        return self.pipe

    # ───────────────────────────────────────────────────────────── public API
    def predict(self, text: str, *, user_id: str = "public") -> Dict[str, Any]:
        h = self._hash(text)

        if h not in self.cache:
            pipe = self._select_pipeline(text)
            self.tracker.start()
            tic = time.perf_counter()

            try:
                self.cache[h] = pipe(text)[0]  # type: ignore[index]
            finally:
                dur = time.perf_counter() - tic
                co2 = self.tracker.stop()
                self.energy_log.append((dur, co2))
                self._save_cache()

        self.interactions.log(user_id, text, self.cache[h])
        return self.cache[h]

    def batch_predict(self, texts: List[str], *, user_id: str = "public") -> List[Dict[str, Any]]:
        hashes = [self._hash(t) for t in texts]
        to_run = [t for t, h in zip(texts, hashes) if h not in self.cache]

        if to_run:
            self.tracker.start()
            tic = time.perf_counter()
            preds = self.pipe(to_run)
            for t, pred in zip(to_run, preds):
                self.cache[self._hash(t)] = pred

            per = len(to_run)
            dur = time.perf_counter() - tic
            co2 = self.tracker.stop()
            self.energy_log.extend([(dur / per, co2 / per)] * per)
            self._save_cache()

        for t, h in zip(texts, hashes):
            self.interactions.log(user_id, t, self.cache[h])

        return [self.cache[h] for h in hashes]

    # embeddings & clustering
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def reduce_and_cluster(
        self, feats: np.ndarray, *, dim: int = 2, n_clusters: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if not (
            self.validator.validate_type(feats, np.ndarray)
            and self.validator.validate_nonempty(feats)
        ):
            raise ValueError("Invalid features for clustering")

        reducer = UMAP(n_components=dim, random_state=42)
        red = reducer.fit_transform(feats)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(red)
        score = silhouette_score(red, labels)
        return red, labels, score

    # export
    def export_results(
        self,
        texts: List[str],
        preds: List[Dict[str, Any]],
        clusters: np.ndarray,
        *,
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

        ext = {"excel": "xlsx", "csv": "csv", "json": "json"}[fmt]
        out = self.cache_dir / f"EcoAI_report_{int(time.time())}.{ext}"

        if fmt == "excel":
            df.to_excel(out, index=False)
        elif fmt == "csv":
            df.to_csv(out, index=False)
        elif fmt == "json":
            df.to_json(out, orient="records", force_ascii=False)
        else:
            raise ValueError("Unsupported export format")

        self.logger.info("%s  Export terminé → %s", CFG.EMOJI_EXPORT, out)
        return out, df

    # résumé énergie/cache
    def summary(self) -> None:
        t_tot = sum(t for t, _ in self.energy_log)
        c_tot = sum(c for _, c in self.energy_log)
        self.logger.info(
            "%s  Temps total : %.2f s | %s  CO₂ total : %.6f kg | %s  Cache : %d entrées",
            CFG.EMOJI_ENERGY,
            t_tot,
            CFG.EMOJI_CO2,
            c_tot,
            CFG.EMOJI_CACHE,
            len(self.cache),
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI rapide  →  `python -m ecoai -t "Some text"`
# ──────────────────────────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(description="EcoAI quick sentiment CLI")
    parser.add_argument("-t", "--text", required=True, help="Text to analyse")
    parser.add_argument("-p", "--password", default="change-me!", help="Master password")
    args = parser.parse_args()

    eco = EcoAI(master_password=args.password)
    res = eco.predict(args.text)
    print(f"{args.text!r} → {res['label']} ({res['score']:.2%})")
    eco.summary()


if __name__ == "__main__":
    _cli()

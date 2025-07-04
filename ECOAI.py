import os
import time
import hashlib
import json
import logging
import base64
import numpy as np
import pandas as pd
from typing import List, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.quantization import quantize_dynamic
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

# --- Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcoAI")

# --- Secure file manager
class SecureFileManager:
    def __init__(self, master_password: str, base_dir: str = "files"):
        self.master_password = master_password.encode()
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _generate_salt(self) -> bytes:
        return os.urandom(16)

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=salt,
            iterations=100000, backend=default_backend()
        )
        return kdf.derive(self.master_password)

    def encrypt_data(self, data: bytes) -> str:
        salt = self._generate_salt()
        key = self._derive_key(salt)
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(data) + encryptor.finalize()
        return base64.b64encode(salt + iv + encryptor.tag + encrypted).decode("utf-8")

    def decrypt_data(self, encrypted_data: str) -> bytes:
        decoded = base64.b64decode(encrypted_data)
        salt, iv, tag, ciphertext = decoded[:16], decoded[16:28], decoded[28:44], decoded[44:]
        key = self._derive_key(salt)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        return cipher.decryptor().update(ciphertext) + cipher.decryptor().finalize()

    def save_encrypted_json(self, data: dict, file_name: str):
        content = json.dumps(data).encode("utf-8")
        encrypted = self.encrypt_data(content)
        with open(os.path.join(self.base_dir, file_name), "w") as f:
            f.write(encrypted)

    def load_encrypted_json(self, file_name: str) -> dict:
        with open(os.path.join(self.base_dir, file_name), "r") as f:
            encrypted = f.read()
        content = self.decrypt_data(encrypted)
        return json.loads(content.decode("utf-8"))

    def save_encrypted_csv(self, df: pd.DataFrame, file_name: str):
        encrypted = self.encrypt_data(df.to_csv(index=False).encode("utf-8"))
        with open(os.path.join(self.base_dir, file_name), "w") as f:
            f.write(encrypted)

    def load_encrypted_csv(self, file_name: str) -> pd.DataFrame:
        with open(os.path.join(self.base_dir, file_name), "r") as f:
            encrypted = f.read()
        from io import StringIO
        return pd.read_csv(StringIO(self.decrypt_data(encrypted).decode("utf-8")))
# ----------- DATA VALIDATION -----------
class DataValidationService:
    @staticmethod
    def validate_type(data: Any, expected_type: type) -> bool:
        return isinstance(data, expected_type)

    @staticmethod
    def validate_nonempty(data: Any) -> bool:
        if isinstance(data, (list, dict, str, np.ndarray, pd.DataFrame)):
            return len(data) > 0
        return bool(data)

# ----------- AUDIT / USER LOGGER -----------
class InteractionLogger:
    def __init__(self, storage_path="interactions", encrypted=True, password="changeme"):
        self.storage_path = storage_path
        self.encrypted = encrypted
        self.password = password
        os.makedirs(storage_path, exist_ok=True)
        self.secure_manager = SecureFileManager(password, base_dir=storage_path)

    def _get_path(self, user_id: str):
        return f"{user_id}_interactions.json"

    def log(self, user_id: str, message: str, result: Any):
        entry = {
            "user_id": user_id,
            "message": message,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        file = self._get_path(user_id)
        try:
            if self.encrypted and os.path.exists(os.path.join(self.storage_path, file)):
                data = self.secure_manager.load_encrypted_json(file)
            else:
                data = []
        except Exception:
            data = []
        data.append(entry)
        if self.encrypted:
            self.secure_manager.save_encrypted_json(data, file)
        else:
            with open(os.path.join(self.storage_path, file), "w") as f:
                json.dump(data, f, indent=4)

# ---------- ECOAI CLASS ----------


class EcoAI:
    def __init__(self, model_name="distilbert-base-uncased", cache_dir="eco_cache",
                 task="sentiment-analysis", master_password=None, business_name="EcoAI_Business"):
        self.logger = logger
        self.logger.info("Init EcoAI...")

        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).cpu()
        self.pipe = pipeline(task, model=self.model, tokenizer=self.tokenizer)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "predictions_cache_encrypted.json")

        self.master_password = master_password or os.getenv("MASTER_PASSWORD", "defaultpassword")
        self.file_manager = SecureFileManager(self.master_password, base_dir=cache_dir)
        self.cache = self._load_cache()

        self.tracker = EmissionsTracker(measure_power_secs=1, output_file=os.path.join(cache_dir, "emissions.csv"))
        self.energy_log = []
        self.validator = DataValidationService()
        self.interaction_logger = InteractionLogger(storage_path=os.path.join(cache_dir, "interactions"),
                                                    encrypted=True, password=self.master_password)
        self.business_name = business_name
        self.logger.info(f"🌱 {business_name} ready to save energy and make money!")

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                return self.file_manager.load_encrypted_json("predictions_cache_encrypted.json")
            except Exception as e:
                self.logger.warning(f"Cache load error: {e}")
        return {}

    def _save_cache(self):
        try:
            self.file_manager.save_encrypted_json(self.cache, "predictions_cache_encrypted.json")
        except Exception as e:
            self.logger.warning(f"Cache save error: {e}")

    def _select_model(self, text: str):
        if len(text.split()) < 6:
            tiny_model = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(tiny_model)
            model = AutoModelForSequenceClassification.from_pretrained(tiny_model)
            return pipeline(self.task, model=model, tokenizer=tokenizer)
        return self.pipe

    def predict(self, text: str, user_id="public") -> dict:
        key = self._hash(text)
        if key in self.cache:
            result = self.cache[key]
        else:
            pipe = self._select_model(text)
            self.tracker.start()
            start = time.time()
            try:
                result = pipe(text)[0]
            finally:
                duration = time.time() - start
                emissions = self.tracker.stop()
            self.cache[key] = result
            self._save_cache()
            self.energy_log.append((duration, emissions))
        self.interaction_logger.log(user_id, text, result)
        return result

    def batch_predict(self, texts: List[str], user_id="public") -> List[dict]:
        normalized = [t.strip().lower() for t in texts]
        keys = [self._hash(t) for t in normalized]
        results, to_predict, to_indices = [], [], []

        for i, key in enumerate(keys):
            if key in self.cache:
                results.append(self.cache[key])
            else:
                results.append(None)
                to_predict.append(texts[i])
                to_indices.append(i)

        if to_predict:
            self.tracker.start()
            start = time.time()
            for txt, idx in zip(to_predict, to_indices):
                pipe = self._select_model(txt)
                pred = pipe(txt)[0]
                self.cache[keys[idx]] = pred
                results[idx] = pred
            duration = time.time() - start
            emissions = self.tracker.stop()
            per_item_time = duration / len(to_predict)
            per_item_emissions = emissions / len(to_predict)
            for _ in to_predict:
                self.energy_log.append((per_item_time, per_item_emissions))
            self._save_cache()

        for t, r in zip(texts, results):
            self.interaction_logger.log(user_id, t, r)
        return results

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts)

    def reduce_and_cluster(self, features: np.ndarray, dim=2, n_clusters=3) -> Tuple[np.ndarray, np.ndarray, float]:
        if not self.validator.validate_type(features, np.ndarray) or not self.validator.validate_nonempty(features):
            raise ValueError("Invalid features for clustering.")
        reducer = UMAP(n_components=dim, random_state=42)
        reduced = reducer.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        return reduced, labels, score

    def visualize_clusters(self, reduced_features: np.ndarray, labels: np.ndarray, texts: List[str]):
        plt.figure(figsize=(8, 6))
        for i in range(len(reduced_features)):
            plt.scatter(reduced_features[i, 0], reduced_features[i, 1], label=f"Cluster {labels[i]}")
            plt.annotate(texts[i], (reduced_features[i, 0], reduced_features[i, 1]), fontsize=8)
        plt.title("Text Clusters")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True)
        plt.show()

    def export_results(self, texts: List[str], predictions: List[dict], clusters: np.ndarray, output_format="excel"):
        df = pd.DataFrame({
            "Text": texts,
            "Prediction": [p["label"] if p else None for p in predictions],
            "Score": [p["score"] if p else None for p in predictions],
            "Cluster": clusters,
        })
        out_name = f"EcoAI_report_{int(time.time())}"
        if output_format == "excel":
            out_path = os.path.join(self.cache_dir, f"{out_name}.xlsx")
            df.to_excel(out_path, index=False)
        elif output_format == "csv":
            out_path = os.path.join(self.cache_dir, f"{out_name}.csv")
            df.to_csv(out_path, index=False)
        elif output_format == "json":
            out_path = os.path.join(self.cache_dir, f"{out_name}.json")
            df.to_json(out_path, orient="records")
        else:
            raise ValueError("Unsupported output format.")
        self.logger.info(f"Results exported: {out_path}")
        return out_path

    def summary(self):
        total_time = sum(d for d, _ in self.energy_log)
        total_emissions = sum(e for _, e in self.energy_log)
        print(f"\n⚡ Total prediction time : {total_time:.2f} seconds")
        print(f"🌍 Total CO₂ emissions : {total_emissions:.6f} kg")
        print(f"♻️ Cached unique predictions : {len(self.cache)}")
        print(f"📚 Logs stored in : {self.interaction_logger.storage_path}")

# =========== EXAMPLE USAGE ===========
if __name__ == "__main__":
    texts = [
        "This is fantastic!",
        "Terrible experience.",
        "I love this!",
        "Worst product ever.",
        "This is fantastic!"
    ]
    eco_ai = EcoAI(master_password="supersecur3password", business_name="EcoAI MoneyMaker")
    features = np.random.rand(len(texts), 10)
    # features = eco_ai.get_embeddings(texts)  # Uncomment if sentence-transformers installed
    predictions = eco_ai.batch_predict(texts, user_id="user_demo")
    reduced, clusters, score = eco_ai.reduce_and_cluster(features)
    print(f"\n📊 Clustering (Silhouette Score: {score:.2f}):")
    for text, pred, label in zip(texts, predictions, clusters):
        print(f"[Cluster {label}] \"{text}\" → {pred['label']} ({pred['score']:.2f})")
    report_path = eco_ai.export_results(texts, predictions, clusters, output_format="excel")
    print(f"📁 Excel report exported: {report_path}")
    eco_ai.summary()

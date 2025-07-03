from ecoai import EcoAI
import numpy as np

# === Sample texts ===
texts = [
    "This is fantastic!",
    "Terrible experience.",
    "I love this!",
    "Worst product ever.",
    "This is fantastic!"  # Duplicate for cache test
]

# === Initialize EcoAI ===
eco_ai = EcoAI(
    master_password="supersecur3password",
    business_name="EcoAI Demo Business"
)

# === Generate predictions ===
print("\n🔍 Running sentiment predictions...")
predictions = eco_ai.batch_predict(texts, user_id="demo_user")

# === Generate embeddings or fake features for clustering ===
# Use real embeddings if sentence-transformers is installed:
# features = eco_ai.get_embeddings(texts)

# Otherwise use random data (for quick testing)
features = np.random.rand(len(texts), 10)

# === Cluster and reduce ===
print("\n🔬 Clustering...")
reduced, clusters, score = eco_ai.reduce_and_cluster(features)

# === Display results ===
print(f"\n📊 Clustering (Silhouette Score: {score:.2f})")
for text, pred, label in zip(texts, predictions, clusters):
    print(f"[Cluster {label}] \"{text}\" → {pred['label']} ({pred['score']:.2f})")

# === Export to Excel ===
report_path = eco_ai.export_results(texts, predictions, clusters, output_format="excel")
print(f"\n📁 Exported report: {report_path}")

# === Energy & prediction summary ===
eco_ai.summary()

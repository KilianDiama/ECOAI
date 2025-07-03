from ecoai import EcoAI
import numpy as np

texts = [
    "This is fantastic!",
    "Terrible experience.",
    "I love this!",
    "Worst product ever.",
    "This is fantastic!"
]

eco_ai = EcoAI(master_password="supersecur3password", business_name="EcoAI Demo")

# Use random features or use real embeddings:
features = np.random.rand(len(texts), 10)
# features = eco_ai.get_embeddings(texts)

predictions = eco_ai.batch_predict(texts, user_id="demo_user")
reduced, clusters, score = eco_ai.reduce_and_cluster(features)

print(f"\n📊 Clustering (Silhouette Score: {score:.2f})")
for text, pred, label in zip(texts, predictions, clusters):
    print(f"[Cluster {label}] \"{text}\" → {pred['label']} ({pred['score']:.2f})")

eco_ai.export_results(texts, predictions, clusters, output_format="excel")
eco_ai.summary()

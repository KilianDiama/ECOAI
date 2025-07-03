# 🌿 EcoAI — Green & Secure NLP Toolkit for Business

**EcoAI** is a production-ready, energy-efficient, and privacy-conscious NLP pipeline.  
It combines **secure data handling**, **low-carbon AI predictions**, and **business-ready exports** in one modular Python framework.

> ✅ Built for developers, data scientists, and companies that care about performance **and** the planet.

---

## 🚀 Features

### 🧠 Natural Language Processing
- Sentiment analysis using a quantized BERT model (faster & lighter on CPU)
- SHA-256 prediction caching to avoid redundant computation

### ⚡ Green AI Optimization
- Integrated [CodeCarbon](https://codecarbon.io/) CO₂ emission tracking
- Dynamic model quantization with PyTorch (up to 60% energy reduction)

### 🔐 Secure Data Storage
- AES-GCM 256-bit encryption for logs, results, and datasets
- Secure JSON/CSV file handling with password-protected access

### 📊 Business Intelligence
- UMAP + KMeans clustering for text segmentation
- Export results in Excel, CSV, or JSON formats
- User-level interaction logging for audits or traceability

---

## 📦 Installation


pip install -r requirements.txt
You need Python 3.8+ and packages listed in requirements.txt

## 🧪 Example Usage

from ecoai import EcoAI

texts = [
    "This is fantastic!",
    "Terrible experience.",
    "I love this!",
    "Worst product ever.",
]

eco = EcoAI(master_password="supersecur3password", business_name="EcoAI MoneyMaker")

predictions = eco.batch_predict(texts, user_id="user_demo")
features = eco.get_embeddings(texts)
reduced, clusters, score = eco.reduce_and_cluster(features)

eco.export_results(texts, predictions, clusters, output_format="excel")
eco.summary()
## 📈 Results Dashboard

python demo.py
Clusters your text

Tracks CO₂ emissions

Logs predictions

Saves an Excel business report

## 🏆 Why EcoAI?

CO₂ tracking	,
Secure file encryption	,
Quantized model (CPU)	,
Business export (Excel)	,
Audit logs per user	,
Clustering integration	,

## 🔒 License
This project is released under the MIT License.
For commercial licensing, extended support, or a private "Pro" version, please contact the author.

## 💬 Get Involved
⭐ Star this project to support green AI

📧 Contact for consulting, integration, or commercial license

❤️ Sponsor this project to support future development

## 🧑‍💻 Author
Developed by KilianDiama 
EcoAI is part of a broader initiative to bring Green AI to production.

"Make AI smarter, greener, and safer."

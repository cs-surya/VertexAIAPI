
# 🔍 GCP Semantic Search API for Physics Papers

A FastAPI-based service that leverages the **arXiv API**, **SciBERT embeddings**, and **Google Cloud Matching Engine** to perform blazing-fast, scalable **semantic search** over physics research papers.

## ⚡ Features

- 🚀 Pulls research papers from arXiv (Physics category)
- 🧠 Encodes paper abstracts with **SciBERT** (`allenai/scibert_scivocab_uncased`)
- 🔎 Performs semantic vector search using **Vertex AI Matching Engine**
- 🔄 Syncs paper metadata to **Firestore**
- 🧰 Built with **FastAPI**, deployable anywhere

---

## 🧱 Tech Stack

- **FastAPI** – Web API framework
- **SciBERT** – Scientific language transformer for embeddings
- **arXiv API** – Source of academic research data
- **Vertex AI Matching Engine** – High-performance vector search on Google Cloud
- **Firestore** – Metadata persistence layer
- **Python** – Obviously

---

## 🌐 API Endpoints

### `GET /`
**Health check**  
Returns `{"status": "ok"}`

---

### `GET /upsert_papers?query=<search_term>`
- 🔎 Fetches top arXiv papers matching the query (in the *physics* category)
- 🧠 Encodes paper abstracts with SciBERT
- 🗃️ Uploads vectors to **Matching Engine**
- 💾 Stores paper metadata in **Firestore**

✅ **Returns:** List of upserted paper IDs

---

### `GET /search?query=<query>&k=3`
- 🚀 Encodes your query with SciBERT
- 🔍 Finds top `k` semantically similar papers using Matching Engine
- 🧾 Returns metadata (title, abstract) from in-memory store or Firestore

✅ **Returns:** Top-k matched papers

---

## 🔐 Environment Setup

Create a `.env` file with the following:

```env
PROJECT_ID=your-gcp-project-id
INDEX_ID=your-matching-engine-index-id
ENDPOINT_ID=your-index-endpoint-id
DEPLOYED_INDEX_ID=your-deployed-index-id
REGION=us-central1
```

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/gcp-semantic-search
cd gcp-semantic-search
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```
fastapi
uvicorn
requests
python-dotenv
sentence-transformers
google-cloud-firestore
google-cloud-aiplatform
```

---

## 🚀 Run Locally

```bash
uvicorn main:app --reload --port 8004
```

Open the interactive docs at [http://localhost:8004/docs](http://localhost:8004/docs)

---

## 📖 Example Usage

### Upsert Papers
```bash
curl "http://localhost:8004/upsert_papers?query=quantum+mechanics"
```

### Search Papers
```bash
curl "http://localhost:8004/search?query=particle+physics&k=3"
```

---

## 🧠 How It Works

1. **Query arXiv** → pull latest physics research papers.
2. **Generate Embeddings** → use SciBERT to vectorize abstracts.
3. **Index in Vertex AI** → upsert vectors to Matching Engine.
4. **Metadata Sync** → store titles/abstracts in Firestore.
5. **Search** → encode query, match against index, return top results.

---

## 🙏 Acknowledgements

- Thanks to **[arXiv](https://arxiv.org/)** for providing open access to scientific research.
- Built on top of **Google Cloud Vertex AI Matching Engine**, making large-scale vector search seamless.


import os

import google
import requests
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from google.cloud import firestore
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1.services.index_service import IndexServiceClient
from google.cloud.aiplatform_v1.types import IndexDatapoint, UpsertDatapointsRequest

# Load .env
load_dotenv()

PROJECT_ID        = os.getenv("PROJECT_ID")
INDEX_ID          = os.getenv("INDEX_ID")
ENDPOINT_ID       = os.getenv("ENDPOINT_ID")
REGION            = os.getenv("REGION", "us-central1")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID")

# Init GCP clients
google.cloud.aiplatform.init(project=PROJECT_ID, location=REGION)
index_client = IndexServiceClient(client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"})
endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_ID)
db = firestore.Client()

app = FastAPI()
model = SentenceTransformer("allenai/scibert_scivocab_uncased")
ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=all:{}+AND+cat:physics*&max_results=10"

paper_store = {}

def fetch_arxiv(query: str):
    resp = requests.get(ARXIV_API_URL.format(query))
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        pid = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        papers.append((pid, title, abstract))
    return papers

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.get("/upsert_papers")
def upsert_papers(query: str = Query(...)):
    papers = fetch_arxiv(query)
    if not papers:
        return {"error": "No papers found"}

    ids, _, abstracts = zip(*papers)
    embeddings = [model.encode(text).tolist() for text in abstracts]

    datapoints = [
        IndexDatapoint(datapoint_id=pid, feature_vector=vec)
        for pid, vec in zip(ids, embeddings)
    ]
    index_client.upsert_datapoints(request=UpsertDatapointsRequest(index=INDEX_ID, datapoints=datapoints))

    batch = db.batch()
    for pid, title, abstract in papers:
        paper_store[pid] = {"id": pid, "title": title, "abstract": abstract}
        doc_ref = db.collection("papers").document(pid)
        batch.set(doc_ref, paper_store[pid])
    batch.commit()

    return {"upserted_ids": list(ids)}

@app.get("/search")
def search(query: str = Query(...), k: int = Query(3, ge=1, le=10)):
    query_vec = model.encode(query).tolist()
    resp = endpoint.match(deployed_index_id=DEPLOYED_INDEX_ID, queries=[query_vec], num_neighbors=k)
    results = []
    for neighbor in resp.nearest_neighbors[0]:
        # Fetch metadata from in‑memory store (or fallback to Firestore)
        doc = paper_store.get(neighbor.id) or db.collection("papers").document(neighbor.id).get().to_dict()
        results.append(doc)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)

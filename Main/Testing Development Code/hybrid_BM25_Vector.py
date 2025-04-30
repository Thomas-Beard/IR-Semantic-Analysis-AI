import requests
from sentence_transformers import SentenceTransformer
import json

bertModel = SentenceTransformer('all-MiniLM-L6-v2')

queryTarget = "neural networks for education"
queryVector = bertModel.encode(queryTarget).tolist()
vectorStr = ','.join([str(round(x, 6)) for x in queryVector])

SOLR_URL = 'http://localhost:8983/solr/research-papers/select'

params = {
    'q': 'abstract:information OR abstract:retrieval OR abstract:research', 
    'fl': 'id,title,score', 
    'rows': 100, 
    'wt': 'json',
    'rq': '{!rerank reRankQuery=$rvec reRankDocs=100 reRankWeight=1.0}', 
    'rvec': f'{{!knn f=vector topK=5}}[{vectorStr}]' 
}

responseRequest = requests.get(SOLR_URL, params=params)

if responseRequest.status_code == 200:
    docs = responseRequest.json()['response']['docs']
    print("🔁 Hybrid Search Results (BM25 + Vector):")
    for doc in docs:
        print(f"ID: {doc['id']}, Title: {doc['title']}, Score: {doc['score']:.3f}")
else:
    print(f"Error: {responseRequest.status_code}")
    print(responseRequest.text)

import requests
import json
from sentence_transformers import SentenceTransformer

bertModel = SentenceTransformer('all-MiniLM-L6-v2')

SOLR_URL = 'http://localhost:8983/solr/research-papers/update?commit=true'

documents = []
for i in range(1, 21):
    title = f"Example Research Paper {i}"
    abstract = f"This is the abstract of example research paper {i}, describing advanced research in information retrieval."
    authors = f"Author {i}"
    combined_text = f"{title}. {abstract}. {authors}"

    vector = bertModel.encode(combined_text).tolist()

    doc = {
        "id": f"paper{i:03}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": 2020 + (i % 5),
        "citations": i * 10,
        "vector": vector 
    }
    documents.append(doc)

headers = {'Content-type': 'application/json'}
responseRequest = requests.post(SOLR_URL, headers=headers, data=json.dumps(documents))

if responseRequest.status_code == 200:
    print("Documents with vectors uploaded successfully.")
else:
    print("Upload failed.")
    print(responseRequest.text)

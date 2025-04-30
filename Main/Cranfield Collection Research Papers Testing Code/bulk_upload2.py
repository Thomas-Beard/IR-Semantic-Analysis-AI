import requests
import json
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET

bertModel = SentenceTransformer('all-MiniLM-L6-v2')

SOLR_URL = 'http://localhost:8983/solr/research-papers/update?commit=true'

with open('cran.all.1400.xml', 'r', encoding='utf-8', errors='ignore') as f:
    raw_data = f.read()

wrappedData = f"<root>\n{raw_data}\n</root>"

root = ET.fromstring(wrappedData)

documents = []

for doc in root.findall('doc'):
    doc_id = doc.find('docno').text.strip()

    titleElem = doc.find('title')
    authorElem = doc.find('author')
    abstractElem = doc.find('text')

    title = titleElem.text.strip() if titleElem is not None and titleElem.text is not None else "No Title"
    author = authorElem.text.strip() if authorElem is not None and authorElem.text is not None else "Unknown Author"
    abstract = abstractElem.text.strip() if abstractElem is not None and abstractElem.text is not None else "No Abstract"

    combinedText = f"{title} {abstract}"
    vector = bertModel.encode(combinedText).tolist()

    document = {
        "id": doc_id,
        "title": title,
        "abstract": abstract,
        "authors": author,
        "year": 1960,
        "citations": 0,
        "vector": vector
    }
    documents.append(document)

batchSize = 100
headers = {'Content-type': 'application/json'}

for i in range(0, len(documents), batchSize):
    batch = documents[i:i + batchSize]
    response = requests.post(SOLR_URL, headers=headers, data=json.dumps(batch))
    if response.status_code == 200:
        print(f'Batch {i//batchSize + 1} uploaded successfully.')
    else:
        print(f'Batch {i//batchSize + 1} upload failed.')
        print(response.text)

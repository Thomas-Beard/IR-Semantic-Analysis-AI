import requests
import json

SOLR_URL = 'http://localhost:8983/solr/research-papers/update?commit=true'

sampleDocuments = []
for i in range(1, 21):
    doc = {
        "id": f"paper{i:03}",
        "title": f"Example Research Paper {i}",
        "abstract": f"This is the abstract of example research paper {i}, describing advanced research in information retrieval.",
        "authors": f"Author {i}",
        "year": 2020 + (i % 5),
        "citations": i * 10
    }
    sampleDocuments.append(doc)

headers = {'Content-type': 'application/json'}
responseRequest = requests.post(SOLR_URL, headers=headers, data=json.dumps(sampleDocuments))

# Check the response
if responseRequest.status_code == 200:
    print('Documents uploaded successfully.')
else:
    print(f'Failed to upload documents. Status code: {responseRequest.status_code}')
    print(responseRequest.text)

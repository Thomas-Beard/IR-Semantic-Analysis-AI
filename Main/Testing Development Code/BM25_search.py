import requests
from tabulate import tabulate

SOLR_SEARCH_URL = 'http://localhost:8983/solr/research-papers/select'

queryTarget = 'abstract:information retrieval'

params = {
    'q': queryTarget,
    'fl': 'id,title,score',
    'wt': 'json',
    'rows': 10,
    'sort': 'score desc'
}

responseRequest = requests.get(SOLR_SEARCH_URL, params=params)

if responseRequest.status_code == 200:
    docs = responseRequest.json()['response']['docs']
    if docs:
        maintable = [[doc['id'], doc['title'], round(doc['score'], 3)] for doc in docs]
        print(tabulate(maintable, headers=["ID", "Title", "Score"], tablefmt="grid"))
    else:
        print("No documents matched the query.")
else:
    print(f"Search failed. Status code: {responseRequest.status_code}")
    print(responseRequest.text)


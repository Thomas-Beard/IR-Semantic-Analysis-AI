import requests

SOLR_SEARCH_URL = 'http://localhost:8983/solr/research-papers/select'

queryTarget = 'year:(2021)'

params = {
    'q': queryTarget,
    'wt': 'json',
    'rows': 10
}

responseRequest = requests.get(SOLR_SEARCH_URL, params=params)

if responseRequest.status_code == 200:
    data = responseRequest.json()
    print(f"Found {data['response']['numFound']} documents:")
    for doc in data['response']['docs']:
        print(f"ID: {doc['id']}, Title: {doc['title']}")
else:
    print(f'Search failed. Status code: {responseRequest.status_code}')
    print(responseRequest.text)

import sys
import time
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer


SOLR_URL = "http://localhost:8990/solr"
COLLECTION_NAME = "research-papers"

bert_model = SentenceTransformer('all-MiniLM-L6-v2')



def check_collection_exists():
    print("[INFO] Checking if collection exists...")
    try:
        check_exist_request = requests.get(f"{SOLR_URL}/admin/collections?action=LIST")
        check_exist_request.raise_for_status()
        collections = check_exist_request.json().get("collections", [])
        if COLLECTION_NAME in collections:
            print(f"[INFO] Collection '{COLLECTION_NAME}' already exists.")
            return True
        return False
    except Exception as e:
        print(f"[INFO] Error checking collection: {e}")
        return False


def wait_for_collection_ready(name, retries=10, delay=3):
    print(f"[INFO] Waiting for collection '{name}' to become ready...")
    for _ in range(retries):
        try:
            check_ready_request = requests.get(f"{SOLR_URL}/admin/collections?action=LIST")
            check_ready_request.raise_for_status()
            if name in check_ready_request.json().get("collections", []):
                print(f"[INFO] Collection '{name}' is ready.")
                return True
        except Exception:
            pass
        time.sleep(delay)
    print(f"[ERROR] Collection '{name}' did not become ready in time.")
    return False


def wait_for_schema_ready(timeout=30):
    print("[INFO] Waiting for schema API to become available...")
    url = f"{SOLR_URL}/{COLLECTION_NAME}/schema/fields"
    for i in range(timeout):
        try:
            schema_request = requests.get(url)
            if schema_request.status_code == 200:
                print("[INFO] Schema API is available.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("[ERROR] Schema API did not become available.")
    return False


def wait_for_solr(max_attempts=10, delay=2):
    print("[INFO] Waiting for Solr to become available...")
    for attempt in range(max_attempts):
        try:
            solr_response = requests.get(f"{SOLR_URL}/admin/info/system", timeout=5)
            if solr_response.status_code == 200:
                print("[INFO] Solr is online.")
                return True
        except:
            print(f"[INFO] Attempt {attempt + 1} failed. Retrying in {delay}s...")
        time.sleep(delay)
    print("[ERROR] Solr did not become available.")
    return False


def create_collection():
    print(f"[INFO] Creating collection '{COLLECTION_NAME}'...")
    params = {
        'action': 'CREATE',
        'name': COLLECTION_NAME,
        'numShards': 1,
        'replicationFactor': 1,
        'collection.configName': '_default'
    }
    try:
        create_collection_request = requests.get(f"{SOLR_URL}/admin/collections", params=params)
        create_collection_request.raise_for_status()
        print("[INFO] Collection creation completed.")
    except Exception as e:
        print(f"[INFO] Failed to create collection: {e}")


def schema_field_exists(field_name):
    url = f"{SOLR_URL}/{COLLECTION_NAME}/schema/fields/{field_name}"
    check_schema_request = requests.get(url)
    return check_schema_request.status_code == 200

def schema_type_exists(type_name):
    url = f"{SOLR_URL}/{COLLECTION_NAME}/schema/fieldtypes/{type_name}"
    check_schema_FT_request = requests.get(url)
    return check_schema_FT_request.status_code == 200

def update_schema():
    print("[INFO] Updating schema fields...")

    if not schema_type_exists("knn_vector"):
        field_type_payload = {
            "add-field-type": {
                "name": "knn_vector",
                "class": "solr.DenseVectorField",
                "vectorDimension": 384,
                "similarityFunction": "cosine"
            }
        }
        update_request = requests.post(f"{SOLR_URL}/{COLLECTION_NAME}/schema", json=field_type_payload)
        update_request.raise_for_status()

    fields_to_add = [
        {"name": "title", "type": "text_general", "stored": True},
        {"name": "author", "type": "text_general", "stored": True},
        {"name": "text", "type": "text_general", "stored": True},
        {"name": "abstract", "type": "text_general", "stored": True},
        {"name": "vector", "type": "knn_vector", "indexed": True, "stored": False}
    ]

    for field in fields_to_add:
        if not schema_field_exists(field["name"]):
            create_field_request = requests.post(f"{SOLR_URL}/{COLLECTION_NAME}/schema",
                              json={"add-field": field},
                              headers={"Content-Type": "application/json"})
            create_field_request.raise_for_status()

    print("[INFO] Schema update completed.")


def upload_documents(xml_path):
    print("[INFO] Uploading documents...")
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = f.read()

        wrapped_data = f"<root>\n{raw_data}\n</root>"
        root = ET.fromstring(wrapped_data)

        docs = []
        for doc in root.findall("doc"):
            text = doc.findtext("text", "").strip()
            abstract = " ".join(text.split()[:50])
            embedding = bert_model.encode(text).tolist()
            docs.append({
                "id": doc.findtext("docno", "").strip(),
                "title": doc.findtext("title", "").strip(),
                "author": doc.findtext("author", "").strip(),
                "text": doc.findtext("text", "").strip(),
                "abstract": " ".join(doc.findtext("text", "").strip().split()[:50]),
                "vector": embedding  # Added vectors field for semantic search
            })

        update_url = f"{SOLR_URL}/{COLLECTION_NAME}/update?commit=true"
        document_upload_request = requests.post(update_url, json=docs, headers={"Content-Type": "application/json"})
        document_upload_request.raise_for_status()
        print(f"[INFO] Uploaded {len(docs)} documents.")
    except ET.ParseError as e:
        print(f"[ERROR] XML parsing failed: {e}")
    except Exception as e:
        print(f"[INFO] Failed to upload documents: {e}")


if __name__ == "__main__":
    check_exists = check_collection_exists()
    if not check_exists:
        create_collection()

    if not wait_for_solr():
        sys.exit(1)

    if wait_for_collection_ready(COLLECTION_NAME):
        XML_FILE = Path(__file__).resolve().parent / "cran.all.1400.xml"
        if wait_for_schema_ready():
            update_schema()
        else:
            print("[ERROR] Aborting schema update due to unavailable schema API.")
        upload_documents(XML_FILE)
    else:
        print("[ERROR] Collection did not become ready in time.")

    # xml_path = Path(__file__).resolve().parent / "cran.all.1400.xml"
    # upload_documents(str(xml_path))


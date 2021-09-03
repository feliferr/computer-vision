import os

from google.cloud import storage

# >>> from google.cloud import storage
# >>> client = storage.Client()
# >>> bucket = client.get_bucket('rec-alg')
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'client' is not defined
# >>> client = storage.Client()
# >>> bucket = client.get_bucket('rec-alg')
# >>> blob = bucket.get_blob("")

BUCKET_NAME = os.getenv("BUCKET_NAME")
SPLIT_PATTERN = f"gs://{BUCKET_NAME}/"
    
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

def download_gs_file(gs_file_path):
    query_path = gs_file_path.split(SPLIT_PATTERN)[1]
    blob = bucket.get_blob(query_path)
    os.makedirs(f"./{query_path}")
    with open(f"./{query_path}",'wb') as f:
        f.write(blob.download_as_bytes())
    
    return query_path

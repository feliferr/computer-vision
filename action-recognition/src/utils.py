import os

from google.cloud import storage

BUCKET_NAME = os.getenv("BUCKET_NAME")
SPLIT_PATTERN = f"gs://{BUCKET_NAME}/"
    
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

def download_gs_file(gs_file_path):
    query_path = gs_file_path.split(SPLIT_PATTERN)[1]
    blob = bucket.get_blob(query_path)

    os.makedirs(os.path.dirname(query_path))
    
    with open(f"./{query_path}",'wb') as f:
        f.write(blob.download_as_bytes())
    
    return query_path

def upload_to_gs(file_path):
    blob = bucket.blob(file_path)
    blob.upload_from_filename(filename=f"./{file_path}")

def list_gs_files(gs_path):
    query_path = gs_path.split(SPLIT_PATTERN)[1]
    blobs = list(bucket.list_blobs(prefix=query_path))

    gs_files_list = [f"gs://{BUCKET_NAME}/{blob.name}" for blob in blobs]

    return gs_files_list
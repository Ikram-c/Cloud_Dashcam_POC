import os
from google.cloud import storage

class MockBlob:
    def __init__(self, name):
        self.name = name

    def download_to_filename(self, destination_file_name):
        print(f"[MOCK GCP] Intercepted download request for blob: {self.name}")
        print(f"[MOCK GCP] Generating synthetic GPX file at: {destination_file_name}")
        
        # Write a minimal, valid GPX string so the gpxpy parser doesn't crash
        mock_gpx_data = """<?xml version="1.0" encoding="UTF-8"?>
        <gpx version="1.1" creator="MockGCP">
          <trk>
            <trkseg>
              <trkpt lat="53.801" lon="-1.554"><time>2026-07-11T12:00:00Z</time></trkpt>
              <trkpt lat="53.805" lon="-1.545"><time>2026-07-11T12:00:05Z</time></trkpt>
              <trkpt lat="53.810" lon="-1.540"><time>2026-07-11T12:00:10Z</time></trkpt>
            </trkseg>
          </trk>
        </gpx>
        """
        with open(destination_file_name, 'w') as f:
            f.write(mock_gpx_data)

class MockBucket:
    def __init__(self, name):
        self.name = name
        
    def blob(self, blob_name):
        return MockBlob(blob_name)

class MockStorageClient:
    """A drop-in replacement for google.cloud.storage.Client for local dev."""
    def bucket(self, bucket_name):
        return MockBucket(bucket_name)

def get_storage_client(use_mock=False):
    """
    Factory function to return the appropriate GCS client.
    """
    if use_mock:
        return MockStorageClient()
    
    # In production, this automatically picks up credentials from the 
    # GOOGLE_APPLICATION_CREDENTIALS environment variable.
    return storage.Client()

def download_gpx_track(client, bucket_name, blob_name, local_path):
    """
    Executes the download using whatever client (real or mock) was injected.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path
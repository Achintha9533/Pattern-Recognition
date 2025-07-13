# utils.py
import requests
import warnings
# No need for torch here if generate_images is removed, but keeping it for now
# in case other utility functions eventually use it.

# === Function to download file from Google Drive ===
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Raise an exception for bad status codes
    response.raise_for_status()

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print(f"Downloaded weights to {destination}")

# The generate_images function has been REMOVED from here.
# It now lives only in generate.py
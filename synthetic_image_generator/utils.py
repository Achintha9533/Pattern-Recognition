"""
Utility functions for file operations.

This module provides helper functions, primarily for downloading files
from Google Drive, which is useful for managing pre-trained model weights
or other large assets required by the project.

Functions
---------
download_file_from_google_drive(file_id, destination)
    Downloads a file from Google Drive to a specified local destination.

Notes
-----
- This function handles Google Drive's download warning for large files.
- It uses the `requests` library for HTTP communication.
- The `generate_images` function has been moved to `generate.py` and is
  no longer part of this utility module.
"""

import requests
import warnings
from pathlib import Path # Import Path for type hinting destination

# === Function to download file from Google Drive ===
def download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Downloads a file from Google Drive to a specified local destination.

    This function handles the common Google Drive download warning for large files
    by automatically confirming the download. It streams the content to the
    destination path.

    Parameters
    ----------
    file_id : str
        The Google Drive file ID of the file to be downloaded.
    destination : pathlib.Path
        The local path (including filename) where the downloaded file will be saved.

    Raises
    ------
    requests.exceptions.RequestException
        If the HTTP request fails (e.g., network error, bad status code).
    Exception
        For any other unexpected errors during the download process.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Assuming '1abc...' is a valid Google Drive file ID
    >>> # dummy_file_id = "1abc..."
    >>> # output_path = Path("my_downloaded_file.zip")
    >>> # try:
    >>> #     download_file_from_google_drive(dummy_file_id, output_path)
    >>> #     print(f"File downloaded to {output_path}")
    >>> # except requests.exceptions.RequestException as e:
    >>> #     print(f"Download failed: {e}")
    >>> # except Exception as e:
    >>> #     print(f"An unexpected error occurred: {e}")
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session() # Use a session for persistent connection and cookies

    # First request to get the download confirmation token (if file is large)
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # If a token is found, make a second request to confirm the download
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Raise an exception for HTTP errors (e.g., 404, 500)
    response.raise_for_status()

    # Write the downloaded content to the destination file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768): # Iterate in chunks for large files
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    # Print confirmation message
    print(f"Downloaded weights to {destination}")

# The generate_images function has been REMOVED from this module.
# It now lives only in generate.py, ensuring a clearer separation of concerns.
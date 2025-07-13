# tests/test_utils.py
import pytest
import requests
import requests_mock

# Adjust import path
from utils import download_file_from_google_drive

def test_download_file_from_google_drive_success(requests_mock, tmp_path):
    """
    GIVEN: a valid Google Drive file ID and a temporary destination path,
           and a mocked HTTP GET request that returns successful dummy content
    WHEN: the `download_file_from_google_drive` function is called
    THEN: the file should be successfully downloaded to the specified destination path,
          and its content should match the mocked response
    """
    file_id = "test_id"
    destination = tmp_path / "weights.pth"
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    # Mock the request to return some dummy content
    requests_mock.get(download_url, text="dummy_weights_content")
    
    download_file_from_google_drive(file_id, destination)
    
    assert destination.exists()
    assert destination.read_text() == "dummy_weights_content"

def test_download_file_from_google_drive_failure(requests_mock, tmp_path):
    """
    GIVEN: a Google Drive file ID and a temporary destination path,
           and a mocked HTTP GET request that returns an error status code (e.g., 404)
    WHEN: the `download_file_from_google_drive` function is called
    THEN: a `requests.exceptions.HTTPError` should be raised,
          and no file should be created at the destination path
    """
    file_id = "test_id_fail"
    destination = tmp_path / "weights_fail.pth"
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"

    # Mock the request to return a 404 Not Found error
    requests_mock.get(download_url, status_code=404)

    with pytest.raises(requests.exceptions.HTTPError):
        download_file_from_google_drive(file_id, destination)
        
    assert not destination.exists()
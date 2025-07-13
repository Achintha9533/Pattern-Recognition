# tests/test_utils.py
import pytest
import requests
import requests_mock

# Adjust import path
from utils import download_file_from_google_drive

def test_download_file_from_google_drive_success(requests_mock, tmp_path):
    """Test successful file download from Google Drive."""
    file_id = "test_id"
    destination = tmp_path / "weights.pth"
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    # Mock the request to return some dummy content
    requests_mock.get(download_url, text="dummy_weights_content")
    
    download_file_from_google_drive(file_id, destination)
    
    assert destination.exists()
    assert destination.read_text() == "dummy_weights_content"

def test_download_file_from_google_drive_failure(requests_mock, tmp_path):
    """Test download failure due to a bad HTTP status."""
    file_id = "test_id_fail"
    destination = tmp_path / "weights_fail.pth"
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"

    # Mock the request to return a 404 Not Found error
    requests_mock.get(download_url, status_code=404)

    with pytest.raises(requests.exceptions.HTTPError):
        download_file_from_google_drive(file_id, destination)
        
    assert not destination.exists()
import sys
import os
from unittest.mock import patch, MagicMock
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fetch_models import get_best_gguf_file

def test_get_best_gguf_file_selection():
    mock_hf_response = {
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "model-Q3_K_M.gguf"},
            {"rfilename": "model-Q4_K_M.gguf"},
            {"rfilename": "model-Q8_0.gguf"}
        ]
    }
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_hf_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        selected = get_best_gguf_file("unsloth/test-repo-GGUF")
        assert selected == "model-Q4_K_M.gguf"

def test_get_best_gguf_file_fallback():
    mock_hf_response = {
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "custom-quant.gguf"}
        ]
    }
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_hf_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        selected = get_best_gguf_file("unsloth/test-repo-GGUF")
        assert selected == "custom-quant.gguf"

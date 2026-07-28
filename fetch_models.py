#!/usr/bin/env python3
import argparse
import sys
import os
import urllib.request
import json
import subprocess

def get_best_gguf_file(repo_id: str) -> str:
    """
    Queries Hugging Face API to find available GGUF files and returns the best one.
    """
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching model metadata for {repo_id}: {e}")
        return None
        
    siblings = [s.get("rfilename") or s.get("rpath") for s in data.get("siblings", []) if s.get("rfilename") or s.get("rpath")]
    gguf_files = [f for f in siblings if f.lower().endswith(".gguf")]
    
    if not gguf_files:
        return None
        
    # We want to select the best quant. In order of preference:
    # Q4_K_M, Q5_K_M, IQ4_XS, Q8_0, Q3_K_M
    preferences = ["Q4_K_M", "Q5_K_M", "IQ4_XS", "Q8_0", "Q3_K_M", "Q4_K_S", "Q4_0"]
    for pref in preferences:
        for f in gguf_files:
            if pref in f:
                return f
                
    # Fallback to the first available gguf file
    return gguf_files[0]

def download_to_k7000(repo_id: str, filename: str) -> bool:
    """
    Spawns wget on k7000 via SSH to download the model file.
    """
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    target_path = "/home/opencode/llama.cpp/models/"
    
    print(f"Initiating remote download on k7000:")
    print(f"URL:  {download_url}")
    print(f"Dest: {target_path}{filename}")
    
    # We use wget with --continue (-c) to support resuming partial downloads.
    cmd = (
        f"mkdir -p {target_path} && "
        f"wget -c -P {target_path} '{download_url}'"
    )
    
    ssh_cmd = ["ssh", "opencode@192.168.1.171", cmd]
    
    # Run the SSH command
    try:
        res = subprocess.run(ssh_cmd)
        if res.returncode == 0:
            print(f"Successfully downloaded {filename} on k7000.")
            return True
        else:
            print(f"Download failed with exit code {res.returncode}.")
            return False
    except Exception as e:
        print(f"SSH execution error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AutoBench Model Fetcher: Download Unsloth GGUF models on k7000")
    parser.add_argument("repo", type=str, help="Hugging Face repository ID (e.g. unsloth/gemma-4-E2B-it-GGUF)")
    args = parser.parse_args()
    
    print(f"Fetching metadata for Hugging Face repo: {args.repo}...")
    best_file = get_best_gguf_file(args.repo)
    if not best_file:
        print(f"Error: No GGUF files found in repository {args.repo}.")
        sys.exit(1)
        
    print(f"Selected quant file: {best_file}")
    
    success = download_to_k7000(args.repo, best_file)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

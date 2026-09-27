import requests
import json
import torch
import sys

def diagnose_dataset_urls():
    print("--- 1. Diagnosing Dataset URLs ---")
    url_404 = "https://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz"
    print(f"Checking URL: {url_404}")
    try:
        res = requests.head(url_404, timeout=5)
        print(f"Result: {res.status_code} {res.reason}")
    except Exception as e:
        print(f"Error: {e}")
        
    hf_api_url = "https://huggingface.co/api/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet-S2-v1.0/tree/main"
    print(f"Checking HF gated API URL: {hf_api_url}")
    try:
        res = requests.get(hf_api_url, timeout=5)
        print(f"Result: {res.status_code} {res.reason}")
        print(f"Response: {res.text[:100]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_dataset_urls()

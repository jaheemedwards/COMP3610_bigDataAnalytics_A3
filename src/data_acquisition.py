import os
import requests

def download_data():
    dataset_path = 'data/raw/Amazon-Reviews-2023'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Downloading dataset to {dataset_path}")
        url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        # Add your download logic here
    else:
        print("Dataset already exists. Skipping download.")
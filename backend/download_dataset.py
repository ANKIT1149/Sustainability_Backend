import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset name and download path
dataset_name = "alistairking/recyclable-and-household-waste-classification"
download_path = "C:/Users/aryan/Downloads/"

# Download the dataset
api.dataset_download_files(dataset_name, path=download_path, unzip=True)

print(f"Dataset downloaded to {download_path}")

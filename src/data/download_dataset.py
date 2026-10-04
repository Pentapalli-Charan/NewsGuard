import os
import requests
import zipfile

url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
download_path = "data/raw/liar_dataset.zip"
extract_path = "data/raw/"

print("Downloading LIAR dataset...")
response = requests.get(url, verify=False)
if response.status_code == 200:
    with open(download_path, 'wb') as f:
        f.write(response.content)

    print("Extracting...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Downloaded and extracted:")
    print(os.listdir(extract_path))
else:
    print(f"Failed to download: {response.status_code}")

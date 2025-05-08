from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile

api = KaggleApi()
api.authenticate()

competition_name = 'sgh-x-mastercard-hackathon-may-2025'
download_path = './data/'

os.makedirs(download_path, exist_ok=True)

api.competition_download_files(competition_name, path=download_path)

for file in os.listdir(download_path):
    if file.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(download_path, file), 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(os.path.join(download_path, file))
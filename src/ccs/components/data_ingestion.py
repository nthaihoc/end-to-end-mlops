import os
import zipfile
import gdown
from ccs import logger
from ccs.utils.common import get_size
from ccs.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self) -> str:

        try:
            dataset_url = self.config.data_URL 
            zip_download_dir = self.config.local_data_file
            logger.info(f"Download data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
                raise e
    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

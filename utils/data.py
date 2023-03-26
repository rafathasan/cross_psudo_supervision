import os
import shutil
import zipfile
from tqdm import tqdm
from dataclasses import dataclass
from onedrivedownloader import download

@dataclass
class DatasetDownloader():
    filename: str
    source: str
    extract_to: str

    def download(self):
        download(url=self.source, filename=self.filename, unzip=True, unzip_path= self.extract_to, force_download=False, force_unzip=False, clean=False)
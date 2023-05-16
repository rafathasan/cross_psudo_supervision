import os
import requests
from tqdm import tqdm
import gdown
import onedrivedownloader
import zipfile

def download_data(url, filename, extract_to, download_dir='./data'):
    if not os.path.exists(extract_to):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        download_filepath = os.path.join(download_dir, filename)

        if 'drive.google.com' in url:
            gdown.download(url, download_filepath, quiet=False, fuzzy=True, resume=True)
        elif 'onedrive.live.com' or 'sharepoint.com' in url:
            onedrivedownloader.download(url=url, filename=download_filepath, unzip=False)
        else:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(download_filepath, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()

        with zipfile.ZipFile(download_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        os.remove(download_filepath)
    else:
        print("File or folder already exists. Skipping extraction.")

if __name__ == "__main__":
    url = "https://drive.google.com/file/d/111oHb3rn7ACQuCZHlz1Ygn1aEWaD4-Ub/view?usp=share_link"
    url2 = "https://iubedubd-my.sharepoint.com/:u:/g/personal/1731117_iub_edu_bd/EUpH9TyypDJEhpSzxPLHa64BxjVGO0OjplXfVrtG_Mo-sQ?e=DL0zPw"
    download_data(url2, "tmp.zip", "./data/alphamatting")

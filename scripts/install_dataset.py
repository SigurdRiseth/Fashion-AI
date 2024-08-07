import os
import urllib.request
from zipfile import ZipFile


URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

try:
    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}...')
        urllib.request.urlretrieve(URL, FILE)
    else:
        print(f'{FILE} already exists. Skipping download.')

    print('Unzipping images...')
    with ZipFile(FILE, 'r') as zip_images:
        zip_images.extractall(FOLDER)
    print('Done!')
except Exception as e:
    print(f'An error occurred: {e}')
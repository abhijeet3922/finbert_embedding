import os
import tarfile
import requests
import tensorflow as tf
from pathlib import Path

def download_file_from_google_drive(id, destination):

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_Finbert(location):

    model_path = Path.cwd()/'fin_model'

    if location == 'dropbox':
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            dataset = tf.keras.utils.get_file(fname=model_path/"fin_model.tar.gz",
            origin="https://www.dropbox.com/s/6oeprcqae7tc459/fin_model.tar.gz?dl=1")
            tar = tarfile.open(model_path/"fin_model.tar.gz")
            tar.extractall()

        else:
            if not os.path.exists(model_path/"fin_model.tar.gz"):
                dataset = tf.keras.utils.get_file(fname=model_path/"fin_model.tar.gz",
                origin="https://www.dropbox.com/s/6oeprcqae7tc459/fin_model.tar.gz?dl=1")
            if not os.path.exists(model_path/"pytorch_model.bin"):
                print("Extracting finbert model tar.gz")
                tar = tarfile.open(model_path/"fin_model.tar.gz")
                tar.extractall()

    if location == 'google drive':
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            print("Downloading the finbert model, will take a minute...")
            fileid = "1feMhKmiW2FNQ9107GLeMYbrijgp3Fv8l"
            download_file_from_google_drive(fileid, model_path/"fin_model.tar.gz")
            tar = tarfile.open(model_path/"fin_model.tar.gz")
            tar.extractall()
        else:
            if not os.path.exists(model_path/"fin_model.tar.gz"):
                download_file_from_google_drive(fileid, model_path/"fin_model.tar.gz")
            if not os.path.exists(model_path/"pytorch_model.bin"):
                print("Extracting finbert model tar.gz")
                tar = tarfile.open(model_path/"fin_model.tar.gz")
                tar.extractall()

    return model_path


if __name__ == "__main__":
    print("package from downloading finbert model")

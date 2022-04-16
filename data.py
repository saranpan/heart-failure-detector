import opendatasets as od
import json, os

def import_dataset(dataset_url):
    with open(r'C:\Users\Wallik\.kaggle\kaggle.json','r') as f:
        code = json.load(f)

    os.environ['KAGGLE_USERNAME'] = code['username']
    os.environ['KAGGLE_KEY'] = code['key']

    od.download(dataset_url, data_dir='.')




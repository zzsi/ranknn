import gzip
from utils import download_file, get_data_path


def get_files():
    return {
        'train': {
            'url': 'https://www.kaggle.com/c/expedia-hotel-recommendations/download/train.csv.gz',
            'local_path': 'expedia_hotels/train.csv.gz'
        },
        'test': {
            'url': 'https://www.kaggle.com/c/expedia-hotel-recommendations/download/train.csv.gz',
            'local_path': 'expedia_hotels/test.csv.gz'
        },
        'destinations': {
            'url': 'https://www.kaggle.com/c/expedia-hotel-recommendations/download/destinations.csv.gz',
            'local_path': 'expedia_hotels/destinations.csv.gz'
        }
    }


def data_generator(name):
    files = get_files()
    local_path = get_data_path(files[name]['local_path'])
    download_file(url=files['name']['url'], dest_path=local_path)
    with gzip.open(local_path) as in_file:
        for line in in_file:
            yield line


def train_data_generator():
    for item in data_generator('train'):
        yield item


def test_data_generator():
    for item in data_generator('test'):
        yield item

import itertools
import numpy as np
import zipfile
import scipy.sparse as sp
from utils import download_file, get_data_path


def get_files():
    return {
        'main': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'local_path': 'movielens/movielens.zip'
        }
    }


def _get_raw_movielens_data(name='main'):
    files = get_files()
    local_path = get_data_path(files[name]['local_path'])
    with zipfile.ZipFile(local_path) as datafile:
        return (datafile.read('ml-100k/ua.base').decode().split('\n'),
                datafile.read('ml-100k/ua.test').decode().split('\n'))


def _parse(data):
    """
    Parse movielens dataset lines.
    """

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        yield uid, iid, rating, timestamp


def _build_interaction_matrix(rows, cols, data):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:
        # Let's assume only really good things are positives
        if rating >= 4.0:
            mat[uid, iid] = 1.0

    return mat.tocoo()


def _get_movie_raw_metadata():
    """
    Get raw lines of the genre file.
    """

    _get_raw_movielens_data()
    files = get_files()
    local_path = get_data_path(files['main']['local_path'])

    with zipfile.ZipFile(local_path) as datafile:
        return datafile.read('ml-100k/u.item').decode(errors='ignore').split('\n')


def get_movielens_item_metadata(use_item_ids):
    """
    Build a matrix of genre features (no_items, no_features).

    If use_item_ids is True, per-item feeatures will also be used.
    """

    features = {}
    genre_set = set()

    for line in _get_movie_raw_metadata():

        if not line:
            continue

        splt = line.split('|')
        item_id = int(splt[0])

        genres = [idx for idx, val in
                  zip(range(len(splt[5:])), splt[5:])
                  if int(val) > 0]

        if use_item_ids:
            # Add item-specific features too
            genres.append(item_id)

        for genre_id in genres:
            genre_set.add(genre_id)

        features[item_id] = genres

    mat = sp.lil_matrix((len(features) + 1,
                         len(genre_set)),
                        dtype=np.int32)

    for item_id, genre_ids in features.items():
        for genre_id in genre_ids:
            mat[item_id, genre_id] = 1

    return mat


def get_dense_triplets(uids, pids, nids, num_users, num_items):

    user_identity = np.identity(num_users)
    item_identity = np.identity(num_items)

    return user_identity[uids], item_identity[pids], item_identity[nids]


def get_triplets(mat):

    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))


def get_movielens_data():
    """
    Return (train_interactions, test_interactions).
    """

    train_data, test_data = _get_raw_movielens_data()

    uids = set()
    iids = set()

    for uid, iid, rating, timestamp in itertools.chain(_parse(train_data),
                                                       _parse(test_data)):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return (_build_interaction_matrix(rows, cols, _parse(train_data)),
            _build_interaction_matrix(rows, cols, _parse(test_data)))

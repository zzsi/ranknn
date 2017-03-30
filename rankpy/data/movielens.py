"""
Adapted from: https://github.com/maciejkula/triplet_recommendations_keras
"""

import itertools
import numpy as np
import zipfile
import scipy.sparse as sp
from utils import download_file, get_data_absolute_path
from dataset import BaseDataset


class MovieLensDataset(BaseDataset):

    def __init__(self):
        self.__train, self.__test = get_movielens_data()

    def num_users(self):
        return self.__train.shape[0]

    def num_items(self):
        return self.__train.shape[1]

    def get_user_side_info(self):
        return get_user_side_info()

    def get_item_side_info(self):
        return get_item_side_info()

    def triplet_batches(self, mode='train', batch_size=100,
        include_user_side_info=False,
        include_item_side_info=False):
        train, test = get_movielens_data()
        if mode == 'train':
            mat = train
        elif mode == 'test':
            mat = test
        else:
            raise ValueError('invalid mode: %s', mode)
        rows, cols, nums = get_triplets(mat)
        user_side_info = get_user_side_info()
        item_side_info = get_item_side_info()

        def produce_x_from_batch(uid_batch, pid_batch, nid_batch):
            # Always include the cf part.
            x_to_yield = {
                'user_id': np.array(uid_batch).reshape(-1, 1),
                'positive_item_id': np.array(pid_batch).reshape(-1, 1),
                'negative_item_id': np.array(nid_batch).reshape(-1, 1)
            }
            if include_user_side_info:
                user_content_vecs = np.array([user_side_info[uid] for uid in uid_batch])
                x_to_yield['user_content'] = user_content_vecs
            if include_item_side_info:
                pos_item_content_vecs = np.array([item_side_info[pid] for pid in pid_batch])
                x_to_yield['positive_item_content'] = pos_item_content_vecs
                neg_item_content_vecs = np.array([item_side_info[nid] for nid in nid_batch])
                x_to_yield['negative_item_content'] = neg_item_content_vecs
            return x_to_yield

        while True:
            uid_batch = []
            pid_batch = []
            nid_batch = []
            for uid, pid, nid in zip(rows, cols, nums):
                uid_batch.append(uid)
                pid_batch.append(pid)
                nid_batch.append(nid)
                if len(uid_batch) >= batch_size:
                    x_to_yield = produce_x_from_batch(uid_batch, pid_batch, nid_batch)
                    uid_batch = []
                    pid_batch = []
                    nid_batch = []
                    yield (
                        x_to_yield,
                        np.ones(shape=(len(x_to_yield['user_id']), 1))
                    )
            if len(uid_batch) > 0:
                x_to_yield = produce_x_from_batch(uid_batch, pid_batch, nid_batch)
                yield (
                    x_to_yield,
                    np.ones(shape=(len(x_to_yield['user_id']), 1))
                )




def get_files():
    return {
        'main': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'local_path': 'movielens/movielens.zip'
        }
    }


def _get_raw_movielens_data(name='main'):
    files = get_files()
    local_path = get_data_absolute_path(files[name]['local_path'])
    for _, metadata in files.items():
        download_file(
            url=metadata.get('url'),
            dest_path=get_data_absolute_path(metadata.get('local_path')),
            overwrite=False
        )
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
    local_path = get_data_absolute_path(files['main']['local_path'])

    with zipfile.ZipFile(local_path) as datafile:
        return datafile.read('ml-100k/u.item').decode(errors='ignore').split('\n')


def get_item_side_info():
    item_side_info = {}
    for line in _get_movie_raw_metadata():
        if not line:
            continue
        fields = line.split('|')
        item_id = int(fields[0])
        genres_vec = [int(val) for val in fields[5:]]
        item_side_info[item_id] = genres_vec
    return item_side_info


def get_occupations():
    files = get_files()
    local_path = get_data_absolute_path(files['main']['local_path'])
    with zipfile.ZipFile(local_path) as datafile:
        return datafile.read('ml-100k/u.occupation').decode(errors='ignore').split('\n')


def _get_user_raw_metadata():
    files = get_files()
    local_path = get_data_absolute_path(files['main']['local_path'])
    with zipfile.ZipFile(local_path) as datafile:
        return datafile.read('ml-100k/u.user').decode(errors='ignore').split('\n')


def get_user_side_info():
    """
    Build a matrix of user features (num_users, num_features).
    """
    occupations = dict(
        [(occ, i) for (i, occ) in enumerate(get_occupations()) if len(occ) > 0]
    )

    def vectorize_age(age):
        try:
            age = max(float(age), 1.)
        except:
            age = 1.
        return [
            int(age >= 5), int(age >= 10), int(age >= 15), int(age >= 20),
            int(age >= 30), int(age >= 50)
        ]

    def vectorize_gender(gender):
        if gender == 'M':
            return [0, 0]
        elif gender == 'F':
            return [1, 0]
        else:
            return [0, 1]

    def vectorize_occupation(occupation):
        # One-hot encoding.
        idx = occupations.get(occupation, len(occupations))
        vec = [0] * (len(occupations) + 1)
        vec[idx] = 1
        return vec

    def generate_row(age, gender, occupation):
        return vectorize_age(age) + vectorize_gender(gender) + vectorize_occupation(occupation)

    user_side_info = {}
    for line in _get_user_raw_metadata():
        if not line:
            continue
        user_id, age, gender, occupation, zip_code = tuple(line.split('|'))
        user_side_info[int(user_id)] = generate_row(age, gender, occupation)
    return user_side_info


def get_dense_triplets(uids, pids, nids, num_users, num_items):

    user_identity = np.identity(num_users)
    item_identity = np.identity(num_items)

    return user_identity[uids], item_identity[pids], item_identity[nids]


def get_triplets(mat):
    # TODO: replace random negative sampling with importance sampling based on 
    #   predictions of current model.
    np.random.seed(42)
    item_side_info = get_item_side_info()
    all_item_ids = set([int(x) for x in item_side_info.keys()])
    # item_ids_in_this_mat = set(mat.col.tolist())
    # # Make sure item_id has associated side info. This should be guaranteed on MovieLens dataset.
    # item_ids_to_sample_from = [iid for iid in item_ids_in_this_mat if iid in all_item_ids]
    # return mat.row, mat.col, np.random.choice(item_ids_to_sample_from, size=len(mat.row))
    return mat.row, mat.col, np.random.choice(range(1, mat.shape[1]), size=len(mat.row))



def train_matrix_shape():
    train, _ = get_movielens_data()
    return train.shape


def test_matrix(include_user_side_info=False, include_item_side_info=False):
    mat_cf = get_movielens_data()[1]  # matrix[uid, iid]
    user_side_info = get_user_side_info()
    item_side_info = get_item_side_info()
    raise NotImplementedError


def get_movielens_data():
    """
    Return (train_interactions, test_interactions) as two sparse matrices.
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

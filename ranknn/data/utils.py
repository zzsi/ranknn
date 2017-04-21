"""Utility functions for file download."""
from __future__ import absolute_import
from __future__ import print_function
import requests
import errno
import os
from os import path
from tqdm import tqdm


DATA_ROOT = '%s/data/rankpy' % path.expanduser('~')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


mkdir_p(DATA_ROOT)


def download_file(url, dest_path, overwrite=False):
    if not overwrite and path.isfile(dest_path):
        return
    res = requests.get(url, stream=True)

    print('Downloading data file: %s' % dest_path)

    mkdir_p(path.dirname(dest_path))

    # TODO: display download progress.
    content_length = int(res.headers.get('content-length', 1))
    chunk_size = 1024
    with tqdm(total=content_length, unit='B', unit_scale=True) as pbar:
        with open(dest_path, 'wb') as outfile:
            for chunk in res.iter_content(chunk_size=chunk_size):
                outfile.write(chunk)
                pbar.update(len(chunk))


def get_data_absolute_path(relative_path):
    return path.join(DATA_ROOT, relative_path)

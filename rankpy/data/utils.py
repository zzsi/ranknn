"""Utility functions for file download."""
from __future__ import absolute_import
from __future__ import print_function
import requests
import errno
import os
from os import path


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
    req = requests.get(url, stream=True)

    print('Downloading data file: %s' % dest_path)

    mkdir_p(path.dirname(dest_path))

    # TODO: display download progress.
    with open(dest_path, 'wb') as outfile:
        for chunk in req.iter_content():
            outfile.write(chunk)


def get_data_absolute_path(relative_path):
    return path.join(DATA_ROOT, relative_path)

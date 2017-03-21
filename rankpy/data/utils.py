"""Utility functions for file download."""
from __future__ import absolute_import
from __future__ import print_function
import requests
from os import path


CWD = path.dirname(path.abspath(__file__))


def download_file(url, dest_path, overwrite=False):
    if not overwrite and path.isfile(dest_path):
        return
    req = requests.get(url, stream=True)

    print('Downloading MovieLens data')

    with open(dest_path, 'wb') as outfile:
        for chunk in req.iter_content():
            outfile.write(chunk)


def get_data_path(relative_path):
    return path.join(CWD, '../../data', relative_path)

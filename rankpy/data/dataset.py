"""
Base Dataset class.
"""

class BaseDataset(object):

    def triplet_batches(self, mode='train', batch_size=100):
        raise NotImplementedError

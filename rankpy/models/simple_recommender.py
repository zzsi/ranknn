"""
Borrowed from: https://github.com/maciejkula/triplet_recommendations_keras
"""
from __future__ import absolute_import

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from .. import metrics


def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss


def build_model(num_users, num_items, latent_dim):

    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(
        num_items, latent_dim, name='item_embedding', input_length=1)

    user_input = Input((1, ), name='user_input')

    positive_item_embedding = Flatten()(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = Flatten()(Embedding(
        num_users, latent_dim, name='user_embedding', input_length=1)(
            user_input))

    loss = merge(
        [positive_item_embedding, negative_item_embedding, user_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam())

    return model


class SimpleRecommender(object):

    def __init__(self, latent_dim=100, n_epochs=10):
        self.__latent_dim = latent_dim
        self.__n_epochs = n_epochs

    def fit(self, loader):

        # Read data
        train, test = loader.get_movielens_data()
        num_users, num_items = train.shape

        # Prepare the test triplets
        test_uid, test_pid, test_nid = loader.get_triplets(test)

        model = build_model(num_users, num_items, self.__latent_dim)

        # Print the model structure
        print(model.summary())

        # Sanity check, should be around 0.5
        print('AUC before training %s' % metrics.full_auc(model, test))

        for epoch in range(self.__n_epochs):

            print('Epoch %s' % epoch)

            # Sample triplets from the training data
            uid, pid, nid = loader.get_triplets(train)

            X = {
                'user_input': uid,
                'positive_item_input': pid,
                'negative_item_input': nid
            }

            model.fit(X,
                      np.ones(len(uid)),
                      batch_size=64,
                      nb_epoch=1,
                      verbose=0,
                      shuffle=True)

            print('AUC %s' % metrics.full_auc(model, test))
